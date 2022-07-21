"""
This module contains an abstract class for inference scripts.
"""

from typing import Any, List, Tuple, Union, TypeVar
from dalle2_pytorch.tokenizer import tokenizer
from dalle2_laion import DalleModelManager
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image as PILImage
import torch
import numpy as np
from contextlib import contextmanager

RepeatObject = TypeVar('RepeatObject')

class InferenceScript:
    def __init__(self, model_manager: DalleModelManager):
        self.model_manager = model_manager
        self.device = model_manager.devices[0]

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
    
    def _pil_to_torch(self, image: Union[PILImage.Image, List[PILImage.Image]]):
        """
        Convert a PIL image into a torch tensor.
        Tensor is of dimension 3 if one image is passed in, and of dimension 4 if a list of images is passed in.
        """
        if isinstance(image, PILImage.Image):
            return ToTensor()(image)
        else:
            return torch.stack([ToTensor()(image[i]) for i in range(len(image))])

    def _torch_to_pil(self, image: torch.tensor):
        """
        If the tensor is a batch of images, then we return a list of PIL images.
        """

        if len(image.shape) == 4:
            return [ToPILImage(image[i]) for i in range(image.shape[0])]
        else:
            return ToPILImage()(image)
    
    def _repeat_tensors_with_batch_size(self, tensors: List[torch.Tensor], repeat_num: int, batch_size: int) -> Tuple[List[torch.Tensor], List[List[int]]]:
        """
        Takes a list of tensors and converts it to a list of tensors of shape (<=batch_size, ...) such that the total number of the original tensors is repeat_num * len(tensors)
        Since there are multiple tensor inputs, we also return a list of indices that correspond to the original tensors.
        """
        assert repeat_num > 0
        assert batch_size > 0
        assert isinstance(tensors[0], torch.Tensor), f"Tensors must be torch tensors, not {type(tensors[0])}"
        assert all(tensors[0].shape == tensor.shape for tensor in tensors), "All tensors must have the same shape to be repeated together."
        num_dims = len(tensors[0].shape)
        current_tensor_index = 0
        num_left = repeat_num
        residual = 0
        result = []
        result_indices = []
        while current_tensor_index < len(tensors):
            if residual > 0:
                # Then we had some from the last tensor that we need to fill in before we start repeating the current tensor
                residual_tensor = tensors[current_tensor_index - 1].repeat(residual, *[1] * num_dims)
                num_to_add = min(num_left, batch_size - residual)
                add_tensor = tensors[current_tensor_index].repeat(num_to_add, *[1] * num_dims)
                result.append(torch.cat([residual_tensor, add_tensor], dim=0))
                result_indices.append([current_tensor_index - 1] * residual + [current_tensor_index] * num_to_add)
                num_left -= num_to_add
            # Expand the current tensor until we have too few to fill another batch
            while num_left >= batch_size:
                result.append(tensors[current_tensor_index].repeat(batch_size, *[1] * num_dims))
                result_indices.append([current_tensor_index] * batch_size)
                num_left -= batch_size
            # Now we need to add the remaining tensors to the next batch and then move on to the next tensor
            residual = num_left
            current_tensor_index += 1
            num_left = repeat_num
        # Take care of the final residual
        if residual > 0:
            residual_tensor = tensors[current_tensor_index - 1].repeat(residual, *[1] * num_dims)
            result.append(residual_tensor)
            result_indices.append([current_tensor_index - 1] * residual)
        return result, result_indices

    def _repeat_object_with_batch_size(self, objects: List[RepeatObject], repeat_num: int, batch_size: int) -> Tuple[List[List[RepeatObject]], List[List[int]]]:
        """
        Takes a list of objects and converts it to a list of objects of shape (<=batch_size, ...) such that the total number of the original objects is repeat_num * len(objects)
        Since there are multiple object inputs, we also return a list of indices that correspond to the original objects.
        """
        assert repeat_num > 0
        assert batch_size > 0
        current_object_index = 0
        num_left = repeat_num
        residual = 0
        result = []
        result_indices = []
        while current_object_index < len(objects):
            if residual > 0:
                # Then we had some from the last object that we need to fill in before we start repeating the current object
                residual_objects = [objects[current_object_index - 1]] * residual
                num_to_add = min(num_left, batch_size - residual)
                add_objects = [objects[current_object_index]] * num_to_add
                result.append(residual_objects + add_objects)
                result_indices.append([current_object_index - 1] * residual + [current_object_index] * num_to_add)
                num_left -= num_to_add
            # Expand the current object until we have too few to fill another batch
            while num_left >= batch_size:
                result.append([objects[current_object_index]] * batch_size)
                result_indices.append([current_object_index] * batch_size)
                num_left -= batch_size
            # Now we need to add the remaining objects to the next batch and then move on to the next object
            residual = num_left
            current_object_index += 1
            num_left = repeat_num
        # Take care of the final residual
        if residual > 0:
            residual_objects = [objects[current_object_index - 1]] * residual
            result.append(residual_objects)
            result_indices.append([current_object_index - 1] * residual)
        return result, result_indices

    def _embed_images(self, images: List[PILImage.Image]) -> torch.Tensor:
        """
        Generates the clip embeddings for a list of images
        """
        assert self.model_manager.decoder_info.data_requirements.can_generate_embedding, "Cannot generate embeddings for this model."
        clip = self.model_manager.clip
        image_embed = clip.embed_image(self._pil_to_torch(images))
        return image_embed.image_embed

    def _encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Generates the clip embeddings for a list of text
        """
        assert self.model_manager.prior_info.data_requirements.can_generate_embedding, "Cannot generate embeddings for this model."
        text_tokens = self._tokenize_text(text)
        clip = self.model_manager.clip
        text_embed = clip.embed_text(text_tokens.to(self.device))
        return text_embed.text_encodings

    def _tokenize_text(self, text: List[str]) -> torch.Tensor:
        """
        Tokenizes a list of text
        """
        return tokenizer.tokenize(text)

class CliInferenceScript(InferenceScript):
    def __init__(self, model_manager: DalleModelManager):
        super().__init__(model_manager)
        raise NotImplementedError("CliInferenceScript is not implemented cause I have no idea how to do it yet.")

if __name__ == "__main__":
    i = InferenceScript(None)
    # t = torch.randn(10)
    # r = i._repeat_with_batch_size(t, 20, 15)
    # print([tens.shape for tens in r])

    # t1 = torch.tensor([1] * 4)
    # t2 = torch.tensor([2] * 4)
    # t1 = torch.randn(4, 4)
    # t2 = torch.randn(4, 4)
    # r = i._repeat_tensors_with_batch_size([t1, t2], 5, 7)
    # print(r[0])
    # print([(tens.shape, tens.min(), tens.max()) for tens in r[0]])
    # print(r[1])

    t1 = 1
    t2 = "asdf"
    r = i._repeat_object_with_batch_size([t1, t2], 5, 7)
    print(r[0])
    print(r[1])