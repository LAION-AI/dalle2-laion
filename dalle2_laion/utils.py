from pathlib import Path
from typing import List
from PIL import Image as PILImage
import torch
import re
import numpy as np

def is_image_file(file: Path) -> bool:
    return file.suffix == '.png' or file.suffix == '.jpg' or file.suffix == '.jpeg'

def is_text_file(file: Path) -> bool:
    return file.suffix == '.txt'

def is_json_file(file: Path) -> bool:
    return file.suffix == '.json'

def get_images_in_dir(dir: Path) -> List[Path]:
    assert dir.is_dir()
    return [file for file in dir.iterdir() if is_image_file(file)]

def get_images_from_paths(paths: List[Path]) -> List[PILImage.Image]:
    return [PILImage.open(path) for path in paths]

def get_prompt_from_filestem(filestem: str) -> str:
    """
    Converts the filename to a prompt with the first letter capitalized and spaces between words.
    We assume the stem is either in snake case or camel case.
    """
    # First, we replace all "_" with " "
    prompt = filestem.replace("_", " ")
    # Then we insert a space before every capital letter that does not already have a space
    prompt = re.sub(r'([A-Z])', r' \1', prompt)
    # Then we capitalize the first letter
    prompt = prompt[0].upper() + prompt[1:]
    return prompt

def get_mask_from_image(image: PILImage.Image) -> torch.Tensor:
    """
    Returns a boolean tensor of the same size as the image.
    Where the red channel of the image is greater than 128, the mask is True.
    """
    mask = torch.zeros(list(reversed(image.size)), dtype=torch.bool)
    mask[np.array(image.getchannel('R')) > 128] = True
    return mask

def center_crop_to_square(image: PILImage.Image) -> PILImage.Image:
    """
    Crops the pill image into a square with the center staying in the same location
    """
    width, height = image.size
    if width > height:
        left = (width - height) // 2
        right = left + height
        return image.crop((left, 0, right, height))
    else:
        top = (height - width) // 2
        bottom = top + width
        return image.crop((0, top, width, bottom))
