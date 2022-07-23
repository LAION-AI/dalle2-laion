from pathlib import Path
from typing import List
from PIL import Image as PILImage
import torch

def is_image_file(file: Path) -> bool:
    return file.suffix == '.png' or file.suffix == '.jpg' or file.suffix == '.jpeg'

def get_images_in_dir(dir: Path) -> List[Path]:
    assert dir.is_dir()
    return [file for file in dir.iterdir() if is_image_file(file)]

def get_images_from_paths(paths: List[Path]) -> List[PILImage.Image]:
    return [PILImage.open(path) for path in paths]

def get_mask_from_image(image: PILImage.Image) -> torch.Tensor:
    """
    Returns a boolean tensor of the same size as the image.
    Where the red channel of the image is less than 128, the mask is True.
    """
    mask = torch.zeros(image.size, dtype=torch.bool)
    mask[image.getchannel('R') < 128] = True
    return mask