import gc

import logging
import time
import math
import os
import warnings

import torch

import torch.utils.checkpoint
from torch.utils.tensorboard import SummaryWriter
import transformers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from tqdm.auto import tqdm
import diffusers
from diffusers.optimization import get_scheduler

from config import parse_args
from tools import generate_pp_images
from models import DreamDiffusionLoRA
from utils import compute_text_embeddings, Logger
from datasets import DreamBoothDataset, DB_collate_fn
from engines import (train_one_epoch, validation, save_model_hook, load_model_hook)

def main(args):
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.output_dir = os.path.join(args.output_dir, timestamp)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='all',
        project_dir=args.output_dir,
        project_config=accelerator_project_config,
    )

    Logger.init(args.output_dir, 'log')
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    logging.info("Generating class images for prior preservation.")
    if args.with_prior_preservation:
        generate_pp_images(
            accelerator=accelerator,
            class_img_root=args.class_data_dir,
            num_images=args.num_class_images,
            prompt=args.class_prompt,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            precision=args.prior_generation_precision,
            sample_batch_size=args.sample_batch_size,
            revision=args.revision,
        )

    # Creat working directory
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(
                os.path.join(args.output_dir, 'validation_images'),
                exist_ok=True)
    tb_writer = SummaryWriter(
        log_dir=os.path.join(args.output_dir, 'tensorboard'))
    # Define the model

    model = DreamDiffusionLoRA(
        tokenizer_name=args.tokenizer_name,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        enable_xformers_memory_efficient_attention=args.enable_xformers_memory_efficient_attention,
        train_text_encoder=args.train_text_encoder,
        with_prior_preservation=args.with_prior_preservation,
        prior_loss_weight=args.prior_loss_weight,
        revision=args.revision)

    # Change the model's dtype to the one specified by the user.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model = model.to(accelerator.device, dtype=weight_dtype)
    # For LoRA Layers, we need to convert them to float32
    model.unet_lora_layers.to(accelerator.device, dtype=torch.float32)
    if args.train_text_encoder:
        model.text_encoder_lora_layers.to(
            accelerator.device, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices  # noqa
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.train_batch_size * accelerator.num_processes)

    # Define Optimizer, only for parameters that are not frozen
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay)

    # Precompute prompt embeddings if requested. In this case, the text encoder
    # will not be kept in memory during training, which is aslo incompatible
    # with training the text encoder.

    if args.pre_compute_text_embeddings:
        pre_computed_instance_prompt_embeddings = compute_text_embeddings(
            args.instance_prompt)
        validation_prompt_negative_prompt_embeds = compute_text_embeddings("")
        if args.validation_prompt is not None:
            validation_prompt_encoder_hidden_states = compute_text_embeddings(
                args.validation_prompt)
        else:
            validation_prompt_encoder_hidden_states = None
        if args.class_prompt is not None:
            pre_computed_class_prompt_embeddings = compute_text_embeddings(
                args.class_prompt)
        else:
            pre_computed_class_prompt_embeddings = None
        model._del_tokenizer_text_encoder()
        gc.collect()
        torch.cuda.empty_cache()
    else:
        pre_computed_instance_prompt_embeddings = None
        validation_prompt_negative_prompt_embeds = None
        validation_prompt_encoder_hidden_states = None
        pre_computed_class_prompt_embeddings = None

    # Build Dataset and Dataloaer
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir
        if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        class_num=args.num_class_images,
        tokenizer=model._get_tokenizer(),
        img_size=args.resolution,
        center_crop=args.center_crop,
        instance_prompt_encoder_hidden_states=  # noqa
        pre_computed_instance_prompt_embeddings,
        class_prompt_encoder_hidden_states=  # noqa
        pre_computed_class_prompt_embeddings,  # noqa
        tokenizer_max_length=args.tokenizer_max_length,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=DB_collate_fn(args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch  # noqa
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps *
        args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps *
        args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        _, _, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model.unet_lora_layers, model.text_encoder_lora_layers, optimizer,
            train_dataloader, lr_scheduler)
    else:
        _, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model.unet_lora_layers, optimizer, train_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the
    # training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch  # noqa
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # ========================= Train! =========================
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps  # noqa
    logging.info("{}".format(args).replace(', ', ',\n'))
    logging.info(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataset)}")
    logging.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logging.info(f"  Num Epochs = {args.num_train_epochs}")
    logging.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logging.info( f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logging.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logging.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."  # noqa
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)
    else:
        resume_step = 0
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process)

    progress_bar.set_description("Steps")
    for epoch in range(first_epoch, args.num_train_epochs):
        global_step = train_one_epoch(
            accelerator=accelerator,
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            tb_writer=tb_writer,
            epoch=epoch,
            first_epoch=first_epoch,
            resume_step=resume_step,
            global_step=global_step,
            progress_bar=progress_bar,
            args=args,
            weight_dtype=weight_dtype,
        )

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:  # noqa
                logging.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"  # noqa
                    f" {args.validation_prompt}.")
                validation(
                    model=model,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    epoch=epoch,
                    global_step=global_step,
                    tb_writer=tb_writer,
                    args=args,
                    validation_prompt_encoder_hidden_states=  # noqa
                    validation_prompt_encoder_hidden_states,
                    validation_prompt_negative_prompt_embeds=  # noqa
                    validation_prompt_negative_prompt_embeds)


if __name__ == "__main__":
    args = parse_args()
    main(args)
