from json import decoder
from pathlib import Path
from dalle2_pytorch.train_configs import AdapterConfig as ClipConfig
from typing import List, Optional, Union
from enum import Enum
from pydantic import BaseModel, root_validator, ValidationError
from contextlib import contextmanager
import tempfile
import urllib.request
import json

class LoadLocation(str, Enum):
    """
    Enum for the possible locations of the data.
    """
    local = "local"
    url = "url"

class File(BaseModel):
    load_type: LoadLocation
    path: str
    cache_dir: Optional[Path] = None
    filename_override: Optional[str] = None

    def download_to(self, path: str):
        """
        Downloads the file to the given path
        """
        assert self.load_type == LoadLocation.url
        urllib.request.urlretrieve(self.path, path)

    @property
    def filename(self):
        if self.filename_override is not None:
            return self.filename_override
        # The filename is everything after the last '/' but before the '?' if it exists
        filename = self.path.split('/')[-1]
        if '?' in filename:
            filename = filename.split('?')[0]
        return filename

    @contextmanager
    def as_local_file(self):
        if self.load_type == LoadLocation.local:
            yield self.path
        elif self.cache_dir is not None:
            # Then we are caching the data in a local directory
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            file_path = self.cache_dir / self.filename
            if not file_path.exists():
                print(f"Downloading {self.path} to {file_path}")
                self.download_to(file_path)
            else:
                print(f'{file_path} already exists. Skipping download. If you think this file should be re-downloaded, delete it and try again.')
            yield file_path
        else:
            # Then we are not caching and the file should be stored in a temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfile = tmpdir + "/" + self.filename
                self.download_to(tmpfile)
                yield tmpfile

class SingleDecoderLoadConfig(BaseModel):
    """
    Configuration for the single decoder load.
    """
    unet_numbers: List[int]
    load_model_from: File
    load_config_from: Optional[File]  # The config may be defined within the model file if the version is high enough

class DecoderLoadConfig(BaseModel):
    """
    Configuration for the decoder load.
    """
    unet_sources: List[SingleDecoderLoadConfig]

    final_unet_number: int

    @root_validator(pre=True)
    def compute_num_unets(cls, values):
        """
        Gets the final unet number
        """
        unet_numbers = []
        assert "unet_sources" in values, "No unet sources defined. Make sure `unet_sources` is defined in the decoder config."
        for value in values["unet_sources"]:
            unet_numbers.extend(value["unet_numbers"])
        final_unet_number = max(unet_numbers)
        values["final_unet_number"] = final_unet_number
        return values

    @root_validator
    def verify_unet_numbers_valid(cls, values):
        """
        The unets must go from 1 to some positive number not skipping any and not repeating any.
        """
        unet_numbers = []
        for value in values["unet_sources"]:
            unet_numbers.extend(value.unet_numbers)
        unet_numbers.sort()
        if len(unet_numbers) != len(set(unet_numbers)):
            raise ValidationError("The decoder unet numbers must not repeat.")
        if unet_numbers[0] != 1:
            raise ValidationError("The decoder unet numbers must start from 1.")
        differences = [unet_numbers[i] - unet_numbers[i - 1] for i in range(1, len(unet_numbers))]
        if any(diff != 1 for diff in differences):
            raise ValidationError("The decoder unet numbers must not skip any.")
        return values

class PriorLoadConfig(BaseModel):
    """
    Configuration for the prior load.
    """
    load_model_from: File
    load_config_from: Optional[File]  # The config may be defined within the model file if the version is high enough

class ModelLoadConfig(BaseModel):
    """
    Configuration for the model load.
    """
    decoder: Optional[DecoderLoadConfig] = None
    prior: Optional[PriorLoadConfig] = None
    clip: Optional[ClipConfig] = None

    devices: Union[List[str], str] = 'cuda:0'  # The device(s) to use for model inference. If a list, the first device is used for loading.
    load_on_cpu: bool = True  # Whether to load the state_dict on the first device or on the cpu
    strict_loading: bool = True  # Whether to error on loading if the model is not compatible with the current version of the code

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path) as f:
            config = json.load(f)
        return cls(**config)