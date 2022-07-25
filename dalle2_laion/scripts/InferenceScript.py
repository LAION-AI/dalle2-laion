"""
This module contains an abstract class for inference scripts.
"""

from typing import List, Tuple, Union, Dict, TypeVar
from dalle2_pytorch.tokenizer import tokenizer
from dalle2_laion import DalleModelManager, ModelLoadConfig
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image as PILImage
import torch
from contextlib import contextmanager

ClassType = TypeVar('ClassType')

class InferenceScript:
    def __init__(self, model_manager: DalleModelManager, verbose: bool = False):
        self.model_manager = model_manager
        self.verbose = verbose
        self.device = model_manager.devices[0] if model_manager is not None else 'cpu'

    @classmethod
    def create(cls: ClassType, config: Union[str, ModelLoadConfig], *args, verbose: bool = False, **kwargs) -> ClassType:
        """
        Creates an instance of the inference script directly from a config.
        Useful if only one inference script will be run at a time.
        """
        if isinstance(config, str):
            config = ModelLoadConfig.from_json_path(config)
        model_manager = DalleModelManager(config)
        return cls(model_manager, *args, **kwargs, verbose=verbose)
    
    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    @contextmanager
    def _clip_in_decoder(self):
        assert self.model_manager.decoder_info is not None, "Cannot use the decoder without a decoder model."
        decoder = self.model_manager.decoder_info.model
        clip = self.model_manager.clip
        decoder.clip = clip
        yield decoder
        decoder.clip = None
        
    @contextmanager
    def _clip_in_prior(self):
        assert self.model_manager.prior_info is not None, "Cannot use the prior without a prior model."
        prior = self.model_manager.prior_info.model
        clip = self.model_manager.clip
        prior.clip = clip
        yield prior
        prior.clip = None

    @contextmanager
    def _decoder_in_gpu(self):
        # Moves the decoder to gpu and prior to cpu and removes both from gpu after the context is exited.
        assert self.model_manager.decoder_info is not None, "Cannot use the decoder without a decoder model."
        if self.model_manager.prior_info is not None:
            prior = self.model_manager.prior_info.model
            prior.to('cpu')
        with self._clip_in_decoder() as decoder:
            decoder.to(self.device)
            yield decoder
            decoder.to('cpu')

    @contextmanager
    def _prior_in_gpu(self):
        # Moves the prior to gpu and decoder to cpu and removes both from gpu after the context is exited.
        assert self.model_manager.prior_info is not None, "Cannot use the prior without a prior model."
        if self.model_manager.decoder_info is not None:
            decoder = self.model_manager.decoder_info.model
            decoder.to('cpu')
        with self._clip_in_prior() as prior:
            prior.to(self.device)
            yield prior
            prior.to('cpu')

    @contextmanager
    def _clip_in_gpu(self):
        # Moves the clip model to gpu and doesn't touch the others. If clip was originally on cpu, then it is moved back to cpu after
        clip = self.model_manager.clip
        assert clip is not None, "Cannot use the clip without a clip model."
        original_device = next(iter(clip.parameters())).device
        not_on_device = original_device != self.device
        if not_on_device:
            clip.to(self.device)
        yield clip
        if not_on_device:
            clip.to(original_device)
    
    def _pil_to_torch(self, image: Union[PILImage.Image, List[PILImage.Image]], resize_for_clip: bool = True):
        """
        Convert a PIL image into a torch tensor.
        Tensor is of dimension 3 if one image is passed in, and of dimension 4 if a list of images is passed in.
        """
        # If the image has an alpha channel, then we need to remove it.
        def process_image(image: PILImage.Image) -> PILImage.Image:
            if resize_for_clip:
                clip_size = self.model_manager.clip.image_size
                image = image.resize((clip_size, clip_size), resample=PILImage.LANCZOS)
            if image.mode == 'RGBA':
                return image.convert('RGB')
            else:
                return image
        
        if isinstance(image, PILImage.Image):
            return ToTensor()(process_image(image))
        else:
            return torch.stack([ToTensor()(process_image(image[i])) for i in range(len(image))])

    def _torch_to_pil(self, image: torch.tensor):
        """
        If the tensor is a batch of images, then we return a list of PIL images.
        """

        if len(image.shape) == 4:
            return [ToPILImage(image[i]) for i in range(image.shape[0])]
        else:
            return ToPILImage()(image)

    def _repeat_tensor_and_batch(self, tensor: Union[torch.Tensor, List[torch.Tensor]], repeat_num: int, batch_size: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Repeats each element of the first dimension of a tensor repeat_num times then batches the result into a list of tensors.
        Also returns a map from the repeated tensor to the index of the original tensor.
        """
        if isinstance(tensor, list):
            tensor = torch.stack(tensor, dim=0)
        batched_repeat = tensor.repeat_interleave(repeat_num, dim=0).split(batch_size, dim=0)
        batched_map = torch.arange(0, tensor.shape[0]).repeat_interleave(repeat_num, dim=0).split(batch_size, dim=0)
        return list(batched_repeat), [t.tolist() for t in batched_map]

    def _embed_images(self, images: List[PILImage.Image]) -> torch.Tensor:
        """
        Generates the clip embeddings for a list of images
        """
        assert self.model_manager.clip is not None, "Cannot generate embeddings for this model."
        images_tensor = self._pil_to_torch(images, resize_for_clip=True).to(self.device)
        with self._clip_in_gpu() as clip:
            image_embed = clip.embed_image(images_tensor)
        return image_embed.image_embed

    def _encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Generates the clip embeddings for a list of text
        """
        assert self.model_manager.clip is not None, "Cannot generate embeddings for this model."
        text_tokens = self._tokenize_text(text)
        with self._clip_in_gpu() as clip:
            text_embed = clip.embed_text(text_tokens.to(self.device))
        return text_embed.text_encodings

    def _tokenize_text(self, text: List[str]) -> torch.Tensor:
        """
        Tokenizes a list of text
        """
        return tokenizer.tokenize(text)

    def _sample_decoder(
        self,
        images: List[PILImage.Image] = None, image_embed: List[torch.Tensor] = None,
        text: List[str] = None, text_encoding: List[torch.Tensor] = None,
        inpaint_images: List[PILImage.Image] = None, inpaint_image_masks: List[torch.Tensor] = None,
        cond_scale: float = None, sample_count: int = 1, batch_size: int = 10,
    ):
        """
        Samples images from the decoder
        Capable of doing basic generation with a list of image embeddings (possibly also conditioned with a list of strings or text embeddings)
        Also capable of two more advanced generation techniques:
        1. Variation generation: If images are passed in the image embeddings will be generated based on those.
        2. In-painting generation: If images and masks are passed in, the images will be in-painted using the masks and the image embeddings.
        """
        if cond_scale is None:
            # Then we use the default scale
            load_config = self.model_manager.model_config.decoder
            unet_configs = load_config.unet_sources
            cond_scale = [1.0] * load_config.final_unet_number
            for unet_config in unet_configs:
                if unet_config.default_cond_scale is not None:
                    for unet_number, new_cond_scale in zip(unet_config.unet_numbers, unet_config.default_cond_scale):
                        cond_scale[unet_number - 1] = new_cond_scale
            
        decoder_info = self.model_manager.decoder_info
        assert decoder_info is not None, "No decoder loaded."
        data_requirements = decoder_info.data_requirements
        min_image_size = min(min(image.size) for image in images) if images is not None else None
        is_valid, errors = data_requirements.is_valid(
            has_image_emb=image_embed is not None, has_image=images is not None,
            has_text_encoding=text_encoding is not None, has_text=text is not None,
            image_size=min_image_size
        )
        assert is_valid, f"The data requirements for the decoder are not satisfied: {errors}"

        # Prepare the data
        image_embeddings = []  # The null case where nothing is done. This should never be used in actuality, but for stylistic consistency I'm keeping it.
        if data_requirements.image_embedding:
            if image_embed is None:
                # Then we need to use clip to generate the image embedding
                image_embed = self._embed_images(images)
            # Then we need to group these tensors into batches of size batch_size such that the total number of samples is sample_count
            image_embeddings, image_embeddings_map = self._repeat_tensor_and_batch(image_embed, repeat_num=sample_count, batch_size=batch_size)
        
        if data_requirements.text_encoding:
            if text_encoding is None:
                text_encoding = self._encode_text(text)
            text_encodings, text_encodings_map = self._repeat_tensor_and_batch(text_encoding, repeat_num=sample_count, batch_size=batch_size)

        assert len(image_embeddings) > 0, "No data provided for decoder inference."
        output_image_map: Dict[int, List[PILImage.Image]] = {}
        with self._decoder_in_gpu() as decoder:
            for i in range(len(image_embeddings)):
                args = {}
                embeddings_map = []
                if data_requirements.image_embedding:
                    args["image_embed"] = image_embeddings[i].to(self.device)
                    embeddings_map = image_embeddings_map[i]
                if data_requirements.text_encoding:
                    args["text_encodings"] = text_encodings[i].to(self.device)
                    embeddings_map = text_encodings_map[i]
                output_images = decoder.sample(**args, cond_scale=cond_scale)
                for output_image, input_embedding_number in zip(output_images, embeddings_map):
                    if input_embedding_number not in output_image_map:
                        output_image_map[input_embedding_number] = []
                    output_image_map[input_embedding_number].append(self._torch_to_pil(output_image))
            return output_image_map

    def _sample_prior(self, text_or_tokens: Union[torch.Tensor, List[str]], cond_scale: float = None, sample_count: int = 1, batch_size: int = 100, num_samples_per_batch: int = 2):
        """
        Samples image embeddings from the prior
        :param text_or_tokens: A list of strings to use as input to the prior or a tensor of tokens generated from strings.
        :param cond_scale: The scale of the conditioning.
        :param sample_count: The number of samples to generate for each input.
        :param batch_size: The max number of samples to run in parallel.
        :param num_samples_per_batch: The number of samples to rerank for each output sample.
        """
        if cond_scale is None:
            # Then we use the default scale
            cond_scale = self.model_manager.model_config.prior.default_cond_scale
            if cond_scale is None:
                # Fallback
                cond_scale = 1.0

        assert self.model_manager.prior_info is not None
        data_requirements = self.model_manager.prior_info.data_requirements
        is_valid, errors = data_requirements.is_valid(
            has_text_encoding=False, has_text=text_or_tokens is not None,
            has_image_emb=False, has_image=False,
            image_size=None
        )
        assert is_valid, f"The data requirements for the prior are not satisfied. {errors}"
        if isinstance(text_or_tokens, list):
            text_tokens = self._tokenize_text(text_or_tokens)
        else:
            text_tokens = text_or_tokens
        text_batches, text_batches_map = self._repeat_tensor_and_batch(text_tokens, repeat_num=sample_count, batch_size=batch_size)
        embedding_map: Dict[int, List[torch.Tensor]] = {}
        # Weirdly the prior requires clip be part of itself to work so we insert it 
        with self._prior_in_gpu() as prior:
            for text_batch, batch_map in zip(text_batches, text_batches_map):
                text_batch = text_batch.to(self.device)
                embeddings = prior.sample(text_batch, cond_scale=cond_scale, num_samples_per_batch=num_samples_per_batch)
                for embedding, embedding_number in zip(embeddings, batch_map):
                    if embedding_number not in embedding_map:
                        embedding_map[embedding_number] = []
                    embedding_map[embedding_number].append(embedding)
        return embedding_map

class CliInferenceScript(InferenceScript):
    def __init__(self, model_manager: DalleModelManager):
        super().__init__(model_manager)
        raise NotImplementedError("CliInferenceScript is not implemented cause I have no idea how I'm going to do it yet.")