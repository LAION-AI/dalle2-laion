"""
This inference script is used to do basic inference without any bells and whistles.
Pass in text, get out image.
"""

from dalle2_laion.scripts import InferenceScript
from typing import Dict, List, Union
from PIL import Image as PILImage
import numpy as np
import torch

class BasicInference(InferenceScript):
    def sample_decoder_with_image_embedding(
        self,
        images: List[PILImage.Image] = None, image_embed: List[torch.Tensor] = None,
        text: List[str] = None, text_encoding: List[torch.Tensor] = None,
        cond_scale: float = 1.0, sample_count: int = 1, batch_size: int = 10,
    ):
        decoder_info = self.model_manager.decoder_info
        assert decoder_info is not None, "No decoder loaded."
        data_requirements = decoder_info.data_requirements
        min_image_size = min(min(image.size) for image in images) if images is not None else None
        assert data_requirements.is_valid(
            has_image_emb=image_embed is not None, has_image=images is not None,
            has_text_encoding=text_encoding is not None, has_text=text is not None,
            image_size=min_image_size
        ), "The data requirements for the decoder are not satisfied."

        # Prepare the data
        image_embeddings = []  # The null case where nothing is done. This should never be used in actuality, but for stylistic consistency I'm keeping it.
        if data_requirements.image_embedding:
            if image_embed is None:
                # Then we need to use clip to generate the image embedding
                image_embed = self._embed_images(images)
            # Then we need to group these tensors into batches of size batch_size such that the total number of samples is sample_count
            image_embeddings, image_embeddings_map = self._repeat_tensors_with_batch_size(image_embed, repeat_num=sample_count, batch_size=batch_size)
            print(f'Batched {torch.stack(image_embed).shape} to {torch.stack(image_embeddings).shape} with batch size {batch_size} and repeat num {sample_count}')
        
        if data_requirements.text_encoding:
            if text_encoding is None:
                text_encoding = self._encode_text(text)
            text_encodings, text_encodings_map = self._repeat_tensors_with_batch_size(text_encoding, repeat_num=sample_count, batch_size=batch_size)

        assert len(image_embeddings) > 0, "No data provided for decoder inference."
        output_image_map: Dict[int, List[PILImage.Image]] = {}
        for i in range(len(image_embeddings)):
            args = {}
            embeddings_map = []
            if data_requirements.image_embedding:
                args["image_embed"] = image_embeddings[i].to(self.device)
                embeddings_map = image_embeddings_map[i]
            if data_requirements.text_encoding:
                args["text_encodings"] = text_encodings[i].to(self.device)
                embeddings_map = text_encodings_map[i]
            output_images = decoder_info.model.sample(**args, cond_scale=cond_scale)
            for output_image, input_embedding_number in zip(output_images, embeddings_map):
                if input_embedding_number not in output_image_map:
                    output_image_map[input_embedding_number] = []
                output_image_map[input_embedding_number].append(self._torch_to_pil(output_image))
        return output_image_map


    def sample_prior_with_text_encoding(self, text: List[str], cond_scale: float = 1.0, sample_count: int = 1, batch_size: int = 100, num_samples_per_batch: int = 2):
        assert self.model_manager.prior_info is not None
        data_requirements = self.model_manager.prior_info.data_requirements
        assert data_requirements.is_valid(
            has_text_encoding=False, has_text=text is not None,
            has_image_emb=False, has_image=False,
            image_size=None
        ), "The data requirements for the prior are not satisfied."
        text_tokens = self._tokenize_text(text)
        text_batches, text_batches_map = self._repeat_tensors_with_batch_size(text_tokens, repeat_num=sample_count, batch_size=batch_size)
        embedding_map: Dict[int, List[torch.Tensor]] = {}
        # Weirdly the prior requires clip be part of itself to work so we insert it 
        with self._clip_in_prior() as prior:
            for text_batch, batch_map in zip(text_batches, text_batches_map):
                text_batch = text_batch.to(self.device)
                embeddings = prior.sample(text_batch, cond_scale=cond_scale, num_samples_per_batch=num_samples_per_batch)
                for embedding, embedding_number in zip(embeddings, batch_map):
                    if embedding_number not in embedding_map:
                        embedding_map[embedding_number] = []
                    embedding_map[embedding_number].append(embedding)
        return embedding_map

    def dream(
        self,
        text: Union[str, List[str]],
        prior_cond_scale: float = 1.0, decoder_cond_scale: float = 1.0,
        prior_sample_count: int = 1, decoder_sample_count: int = 1,
        prior_batch_size: int = 100, decoder_batch_size: int = 10,
        prior_num_samples_per_batch: int = 2
    ):
        if isinstance(text, str):
            text = [text]
        image_embedding_map = self.sample_prior_with_text_encoding(text, cond_scale=prior_cond_scale, sample_count=prior_sample_count, batch_size=prior_batch_size, num_samples_per_batch=prior_num_samples_per_batch)
        # This is a map between the text index and the generated image embeddings
        # In order to 
        image_embeddings: List[torch.Tensor] = []
        for i in range(len(text)):
            image_embeddings.extend(image_embedding_map[i])
        # In order to get the original text from the image embeddings, we need to reverse the map
        image_embedding_index_reverse_map = {i: [] for i in range(len(text))}
        current_count = 0
        texts = []
        for i in range(len(text)):
            for _ in range(len(image_embedding_map[i])):
                texts.append(text[i])
                image_embedding_index_reverse_map[i].append(current_count)
                current_count += 1
        # Now we can use the image embeddings to generate the images
        image_map = self.sample_decoder_with_image_embedding(text=texts, image_embed=image_embeddings, cond_scale=decoder_cond_scale, sample_count=decoder_sample_count, batch_size=decoder_batch_size)
        # Now we will reconstruct a map from text to a map of img_embedding indices to list of images
        output_map: Dict[int, Dict[int, List[PILImage.Image]]] = {}
        for i, text in enumerate(text):
            output_map[text] = {}
            embedding_indices = image_embedding_index_reverse_map[i]
            for embedding_index in embedding_indices:
                output_map[text][embedding_index] = image_map[embedding_index]
        return output_map