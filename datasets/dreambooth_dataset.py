import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from utils import tokenize_prompt

class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class DreamBoothDataset(Dataset):
    """A dataset to prepare the instance and class images with the prompts for
    fine-tuning the model. It pre-processes the images and the tokenizes
    prompts.

    Args:
        instance_data_root (str): Path to the directory containing the
            instance images.
        instance_prompt (str): The prompt with identifier specifying the
            instance. E.g.: A photo of a idada cat. idada is the identifier.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be used
            for tokenizing the prompts.
        class_data_root (str, optional): A folder containing the training data
            of class images. This is used to compute prior_preservation loss.
            For example, if the instance class is cat, then this folder should
            contain images of cats. Defaults to None. If not provided, we can
            also generate the class images on the fly with Stable Diffusion.
        class_prompt (str, optional):  The prompt to specify images in the same
            class as provided instance images.
        class_num (int, optional): Minimal class images for prior preservation
            loss. If there are not enough images already present in
            class_data_root, additional images will be sampled with
            class_prompt.
        img_size (int, optional): The size of the image to be used for
            training. Defaults to 512.
        center_crop (bool, optional): Whether to use center crop or random
            crop. Defaults to False.
        instance_prompt_encoder_hidden_states (torch.Tensor, optional): The id
            of the prombt text for the instance. If not provided, it will be
            generated with the tokenizer. Defaults to None.
        class_prompt_encoder_hidden_states (torch.Tensor, optional): The
            embedding of the prompt text for the instance. If not provided, it
            will be generated with the tokenizer. Defaults to None.
        tokenizer_max_length (int, optional): The maximum length of the
            tokenizer. Defaults to None.
    """

    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str,
        tokenizer,
        class_data_root: str = None,
        class_prompt: str = None,
        class_num: int = 100,
        img_size: int = 512,
        center_crop: bool = False,
        instance_prompt_encoder_hidden_states: torch.Tensor = None,
        class_prompt_encoder_hidden_states: torch.Tensor = None,
        tokenizer_max_length: int = None,
    ) -> None:
        self.img_size = img_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.instance_prompt_encoder_hidden_states = instance_prompt_encoder_hidden_states  # noqa
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states  # noqa
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(
                    len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose([
            transforms.Resize(
                img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size)
            if center_crop else transforms.RandomCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.instance_prompt_encoder_hidden_states is not None:
            example[
                "instance_prompt_ids"] = self.instance_prompt_encoder_hidden_states  # noqa
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer,
                self.instance_prompt,
                tokenizer_max_length=self.tokenizer_max_length)
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if self.class_prompt_encoder_hidden_states is not None:
                example[
                    "class_prompt_ids"] = self.class_prompt_encoder_hidden_states  # noqa
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer,
                    self.class_prompt,
                    tokenizer_max_length=self.tokenizer_max_length)
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example[
                    "class_attention_mask"] = class_text_inputs.attention_mask

        return example


class DB_collate_fn(object):
    """Collate function for DreamBoothDataset

    Args:
        with_prior_preservation (bool, optional): Whether to compute
            prior_preservation loss. Defaults to False.
    """

    def __init__(self, with_prior_preservation: bool = False) -> None:
        self.with_prior_preservation = with_prior_preservation

    def __call__(self, examples):
        has_attention_mask = "instance_attention_mask" in examples[0]

        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        if has_attention_mask:
            attention_mask = [
                example["instance_attention_mask"] for example in examples
            ]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if self.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            if has_attention_mask:
                attention_mask += [
                    example["class_attention_mask"] for example in examples
                ]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(
            memory_format=torch.contiguous_format).float()

        input_ids = torch.cat(input_ids, dim=0)

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

        if has_attention_mask:
            batch["attention_mask"] = attention_mask

        return batch
