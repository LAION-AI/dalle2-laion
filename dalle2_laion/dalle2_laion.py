from dataclasses import dataclass
from typing import Any, Tuple, Optional, NamedTuple, TypeVar, Generic, List
from dalle2_laion.config import DecoderLoadConfig, SingleDecoderLoadConfig, PriorLoadConfig, ModelLoadConfig
from dalle2_pytorch import __version__ as Dalle2Version, Decoder, DiffusionPrior, Unet
from dalle2_pytorch.train_configs import TrainDecoderConfig, TrainDiffusionPriorConfig, DecoderConfig, UnetConfig, DiffusionPriorConfig
import torch
import torch.nn as nn
from packaging import version

def exists(obj: Any) -> bool:
    return obj is not None

@dataclass
class DataRequirements:
    image_embedding: bool
    text_encoding: bool
    image: bool
    text: bool
    can_generate_embedding: bool
    image_size: int

    def has_clip(self):
        self.can_generate_embedding = True

    def is_valid(
        self,
        has_image_emb: bool = False, has_text_encoding: bool = False,
        has_image: bool = False, has_text: bool = False,
        image_size: Optional[int] = None
    ):
        # The image size must be equal to or greater than the required size
        # Verify that the text input is valid
        errors = []
        is_valid = True
        if self.text_encoding:
            # Then we need to some way to get the text encoding
            if not (has_text_encoding or (self.can_generate_embedding and has_text)):
                errors.append('Text encoding is required, but no text encoding or text was provided')
                is_valid = False
        if self.text:
            # Then this requires text be passed in explicitly
            if not has_text:
                errors.append('Text is required, but no text was provided')
                is_valid = False

        # Verify that the image input is valid
        image_size_greater = exists(image_size) and image_size >= self.image_size
        if self.image_embedding:
            # Then we need to some way to get the image embedding
            # In this case, we also need to make sure that the image size is big enough to generate the embedding
            if not (has_image_emb or (self.can_generate_embedding and has_image and image_size_greater)):
                errors.append('Image embedding is required, but no image embedding or image was provided or the image was too small')
                is_valid = False
        if self.image:
            # Then this requires an image be passed in explicitly
            # In this case we also need to make sure the image is big enough to be used
            if not (has_image and image_size_greater):
                errors.append('Image is required, but no image was provided or the image was too small')
                is_valid = False
        return is_valid, errors

    def __add__(self, other: 'DataRequirements') -> 'DataRequirements':
        return DataRequirements(
            image_embedding=self.image_embedding or other.image_embedding,  # If either needs an image embedding, the combination needs one
            text_embedding=self.text_embedding or other.text_embedding,  # If either needs a text embedding, the combination needs one
            image=self.image or other.image,  # If either needs an image, the combination needs it  
            text=self.text or other.text,  # If either needs a text, the combination needs it
            can_generate_embedding=self.can_generate_embedding and other.can_generate_embedding,  # If either cannot generate an embedding, we know that trying to replace an embedding with raw data will not work
            image_size=max(self.image_size, other.image_size)  # We can downsample without loss of information, so we use the larger image size
        )

ModelType = TypeVar('ModelType', Decoder, DiffusionPrior)
class ModelInfo(NamedTuple, Generic[ModelType]):
    model: ModelType
    model_version: Optional[version.Version]
    requires_clip: bool
    data_requirements: DataRequirements

class DalleModelManager:
    """
    Used to load priors and decoders and to provide a simple interface to run general scripts against
    """
    def __init__(self, model_load_config: ModelLoadConfig):
        self.current_version = version.parse(Dalle2Version)
        self.single_device = isinstance(model_load_config.devices, str)
        self.devices = [torch.device(model_load_config.devices)] if self.single_device else [torch.device(d) for d in model_load_config.devices]
        self.load_device = torch.device('cpu') if model_load_config.load_on_cpu else self.devices[0]
        self.strict_loading = model_load_config.strict_loading

        if model_load_config.decoder is not None:
            self.decoder_info = self.load_decoder(model_load_config.decoder)
        else:
            self.decoder_info = None

        if model_load_config.prior is not None:
            self.prior_info = self.load_prior(model_load_config.prior)
        else:
            self.prior_info = None

        if (exists(self.decoder_info) and self.decoder_info.requires_clip) or (exists(self.prior_info) and self.prior_info.requires_clip):
            assert model_load_config.clip is not None, 'Your model requires clip to be loaded. Please provide a clip config.'
            self.clip = model_load_config.clip.create().to(self.devices[0])
            # Update the data requirements to include the clip model
            if exists(self.decoder_info):
                self.decoder_info.data_requirements.has_clip()
            if exists(self.prior_info):
                self.prior_info.data_requirements.has_clip()
        else:
            if model_load_config.clip is not None:
                print(f'WARNING: Your model does not require clip, but you provided a clip config. This will be ignored.')

    def _get_decoder_data_requirements(self, decoder_config: DecoderConfig, min_unet_number: int = 1) -> DataRequirements:
        """
        Returns the data requirements for a decoder
        """
        return DataRequirements(
            image_embedding=True,
            text_encoding=any(unet_config.cond_on_text_encodings for unet_config in decoder_config.unets[min_unet_number - 1:]),
            image=min_unet_number > 1,  # If this is an upsampler we need an image
            text=False,  # Text is never required for anything
            can_generate_embedding=False,  # This might be added later if clip is being used
            image_size=decoder_config.image_sizes[min_unet_number - 1]  # The input image size is the input to the first unet we are using
        )

    def _load_single_decoder(self, load_config: SingleDecoderLoadConfig) -> Tuple[Decoder, DecoderConfig, Optional[version.Version], bool]:
        """
        Loads a single decoder from a model and a config file
        """
        with load_config.load_model_from.as_local_file() as model_file:
            model_state_dict = torch.load(model_file, map_location=self.load_device)
            if 'version' in model_state_dict:
                model_version = model_state_dict['version']
                if model_version != self.current_version:
                    print(f'WARNING: This decoder was trained on version {model_version} but the current version is {self.current_version}. This may result in the model failing to load.')
            else:
                print(f'WARNING: This decoder was trained on an old version of Dalle2. This may result in the model failing to load or it may lead to producing garbage results.')
                model_version = None  # No version info in the model
            
            requires_clip = False
            if 'config' in model_state_dict:
                # Then we define the decoder config from this object
                decoder_config = TrainDecoderConfig(**model_state_dict['config']).decoder
                if decoder_config.clip is not None:
                    # We don't want to load clip with the model
                    requires_clip = True
                    decoder_config.clip = None
                decoder = decoder_config.create().to(self.devices[0]).eval()
                decoder.load_state_dict(model_state_dict['model'], strict=self.strict_loading)  # If the model has a config included, then we know the model_state_dict['model'] is the actual model
            else:
                # In this case, the state_dict is the model itself. This means we also must load the config from an external file
                assert load_config.load_config_from is not None
                with load_config.load_config_from.as_local_file() as config_file:
                    decoder_config = TrainDecoderConfig.from_json_path(config_file).decoder
                    if decoder_config.clip is not None:
                        # We don't want to load clip with the model
                        requires_clip = True
                        decoder_config.clip = None
                decoder = decoder_config.create().to(self.devices[0]).eval()
                decoder.load_state_dict(model_state_dict, strict=self.strict_loading)

            return decoder, decoder_config, model_version, requires_clip

    def load_decoder(self, load_config: DecoderLoadConfig) -> 'ModelInfo[Decoder]':
        """
        Loads a decoder from a model and a config file
        """
        if len(load_config.unet_sources) == 1:
            # Then we are loading only one model
            decoder, decoder_config, decoder_version, requires_clip = self._load_single_decoder(load_config.unet_sources[0])
            decoder_data_requirements = self._get_decoder_data_requirements(decoder_config)
            decoder.to(torch.float32)
            return ModelInfo(decoder, decoder_version, requires_clip, decoder_data_requirements)
        else:
            true_unets: List[Unet] = [None] * load_config.final_unet_number  # Stores the unets that will replace the ones in the true decoder
            true_unet_configs: List[UnetConfig] = [None] * load_config.final_unet_number  # Stores the unet configs that will replace the ones in the true decoder config
            true_upsampling_sizes: List[Tuple[int, int]] = [None] * load_config.final_unet_number  # Stores the progression of upsampling sizes for each unet so that we can validate these unets actually work together
            true_train_timesteps: List[int] = [None] * load_config.final_unet_number  # Stores the number of timesteps that each unet trained with
            true_beta_schedules: List[str] = [None] * load_config.final_unet_number  # Stores the beta scheduler that each unet used
            true_uses_learned_variance: List[bool] = [None] * load_config.final_unet_number  # Stores whether each unet uses learned variance

            requires_clip = False
            for source in load_config.unet_sources:
                decoder, decoder_config, decoder_version, unets_requires_clip = self._load_single_decoder(source)
                if unets_requires_clip:
                    requires_clip = True
                for unet_number in source.unet_numbers:
                    print(f"Loading unet {unet_number}")
                    unet_index = unet_number - 1
                    # Now we need to insert the unet into the true unets and the unet config into the true config
                    true_unets[unet_index] = decoder.unets[unet_index]
                    true_unet_configs[unet_index] = decoder_config.unets[unet_index]
                    true_upsampling_sizes[unet_index] = None if unet_index == 0 else decoder_config.image_sizes[unet_index - 1], decoder_config.image_sizes[unet_index]
                    true_train_timesteps[unet_index] = decoder_config.timesteps
                    true_beta_schedules[unet_index] = decoder_config.beta_schedule[unet_index]
                    true_uses_learned_variance[unet_index] = decoder_config.learned_variance if isinstance(decoder_config.learned_variance, bool) else decoder_config.learned_variance[unet_index]

            true_decoder_config_obj = {}
            # Insert the true configs into the true decoder config
            true_decoder_config_obj['unets'] = true_unet_configs
            true_image_sizes = []
            for i in range(load_config.final_unet_number):
                if i == 0:
                    true_image_sizes.append(true_upsampling_sizes[i][1])
                else:
                    assert true_upsampling_sizes[i - 1][1] == true_upsampling_sizes[i][0], f"The upsampling sizes for unet {i} are not compatible with unet {i - 1}."
                    true_image_sizes.append(true_upsampling_sizes[i][1])
            true_decoder_config_obj['image_sizes'] = true_image_sizes
            # All unets must have been trained with the same number of sampling timesteps in order to be compatible
            assert all(true_train_timesteps[0] == t for t in true_train_timesteps), f"All unets must have been trained with the same number of sampling timesteps in order to be compatible."
            true_decoder_config_obj['timesteps'] = true_train_timesteps[0]
            true_decoder_config_obj['beta_schedule'] = true_beta_schedules
            true_decoder_config_obj['learned_variance'] = true_uses_learned_variance

            # Now we can create the decoder and substitute the unets
            true_decoder_config = DecoderConfig(**true_decoder_config_obj)
            decoder_data_requirements = self._get_decoder_data_requirements(true_decoder_config)
            decoder = true_decoder_config.create().to(self.devices[0]).eval()
            decoder.unets = nn.ModuleList(true_unets)
            decoder.to(torch.float32)
            return ModelInfo(decoder, decoder_version, requires_clip, decoder_data_requirements)
            
    def _get_prior_data_requirements(self, config: DiffusionPriorConfig) -> DataRequirements:
        """
        Returns the data requirements for a diffusion prior
        """
        return DataRequirements(
            image_embedding=False,  # This is kinda the whole point
            text_encoding=True,  # This is also kinda the whole point
            image=False,  # The prior is never conditioned on the image
            text=False,  # Text is never required for anything
            can_generate_embedding=False,  # This might be added later if clip is being used
            image_size=[-1, -1]  # This is not used
        )

    def load_prior(self, load_config: PriorLoadConfig) -> 'ModelInfo[DiffusionPrior]':
        """
        Loads a prior from a model and a config file
        """
        with load_config.load_model_from.as_local_file() as model_file:
            model_state_dict = torch.load(model_file, map_location=self.load_device)
            if 'version' in model_state_dict:
                model_version = model_state_dict['version']
                if model_version != self.current_version:
                    print(f'WARNING: This prior was trained on version {model_version} but the current version is {self.current_version}. This may result in the model failing to load.')
            else:
                print('WARNING: This prior was trained on an old version of Dalle2. This may result in the model failing to load or it may produce garbage results.')
                model_version = None

            requires_clip = False
            if 'config' in model_state_dict:
                # Then we define the prior config from this object
                prior_config = TrainDiffusionPriorConfig(**model_state_dict['config']).prior
                if prior_config.clip is not None:
                    # We don't want to load clip with the model
                    prior_config.clip = None
                    requires_clip = True
                prior = prior_config.create().to(self.devices[0]).eval()
                prior.load_state_dict(model_state_dict['model'], strict=self.strict_loading)
            else:
                # In this case, the state_dict is the model itself. This means we also must load the config from an external file
                assert load_config.load_config_from is not None
                with load_config.load_config_from.as_local_file() as config_file:
                    prior_config = TrainDiffusionPriorConfig.from_json_path(config_file).prior
                    if prior_config.clip is not None:
                        # We don't want to load clip with the model
                        prior_config.clip = None
                        requires_clip = True
                prior = prior_config.create().to(self.devices[0]).eval()
                prior.load_state_dict(model_state_dict, strict=self.strict_loading)

            data_requirements = self._get_prior_data_requirements(prior_config)
            prior.to(torch.float32)
            return ModelInfo(prior, model_version, requires_clip, data_requirements)