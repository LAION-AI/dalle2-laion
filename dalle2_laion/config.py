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
    checksum_file_path: Optional[str] = None
    cache_dir: Optional[Path] = None
    filename_override: Optional[str] = None

    @root_validator(pre=True)
    def add_default_checksum(cls, values):
        """
        When loading from url, the checksum is the best way to see if there is an update to the model.
        If we are loading from specific places, we know it is already storing a checksum and we can read and compare those to check for updates.
        Sources we can do this with:
        1. Huggingface: If model is at https://huggingface.co/[ORG?]/[REPO]/resolve/main/[PATH_TO_MODEL.pth] we know the checksum is at https://huggingface.co/[ORG?]/[REPO]/raw/main/[PATH_TO_MODEL.pth]
        """
        if values["load_type"] == LoadLocation.url:
            filepath = values["path"]
            existing_checksum = values["checksum_file_path"] if "checksum_file_path" in values else None
            if filepath.startswith("https://huggingface.co/") and "resolve" in filepath and existing_checksum is None:
                values["checksum_file_path"] = filepath.replace("resolve/main/", "raw/main/")
        return values

    def download_to(self, path: Path):
        """
        Downloads the file to the given path
        """
        assert self.load_type == LoadLocation.url
        urllib.request.urlretrieve(self.path, path)
        if self.checksum_file_path is not None:
            urllib.request.urlretrieve(self.checksum_file_path, str(path) + ".checksum")

    def download_checksum_to(self, path: Path):
        """
        Downloads the checksum to the given path
        """
        assert self.load_type == LoadLocation.url
        assert self.checksum_file_path is not None, "No checksum file path specified"
        urllib.request.urlretrieve(self.checksum_file_path, path)

    def get_remote_checksum(self):
        """
        Downloads the remote checksum as a tempfile and returns its content
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            self.download_checksum_to(tmpdir + "/checksum")
            with open(tmpdir + "/checksum", "r") as f:
                checksum = f.read()
        return checksum

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
    def as_local_file(self, check_update: bool = True):
        """
        Loads the file as a local file.
        If check_update is True, it will download a new version if the checksum is different.
        """
        if self.load_type == LoadLocation.local:
            yield self.path
        elif self.cache_dir is not None:
            # Then we are caching the data in a local directory
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            file_path = self.cache_dir / self.filename
            cached_checksum_path = self.cache_dir / (self.filename + ".checksum")
            if not file_path.exists():
                print(f"Downloading {self.path} to {file_path}")
                self.download_to(file_path)
            else:
                # Then we should download and compare the checksums
                if self.checksum_file_path is None:
                    print(f'{file_path} already exists. Skipping download. No checksum found so if you think this file should be re-downloaded, delete it and try again.')
                elif not cached_checksum_path.exists():
                    # Then we don't know if the file is up to date so we should download it
                    if check_update:
                        print(f"Checksum not found for {file_path}. Downloading it again.")
                        self.download_to(file_path)
                    else:
                        print(f"Checksum not found for {file_path}, but updates are disabled. Skipping download.")
                else:
                    new_checksum = self.get_remote_checksum()
                    with open(cached_checksum_path, "r") as f:
                        old_checksum = f.read()
                    should_update = new_checksum != old_checksum
                    if should_update:
                        if check_update:
                            print(f"Checksum mismatch. Deleting {file_path} and downloading again.")
                            file_path.unlink()
                            self.download_to(file_path)  # This automatically overwrites the checksum file
                        else:
                            print(f"Checksums mismatched, but updates are disabled. Skipping download.")
                    
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
    default_sample_timesteps: Optional[List[int]] = None
    default_cond_scale: Optional[List[float]] = None
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
    default_sample_timesteps: Optional[int] = None
    default_cond_scale: Optional[float] = None
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