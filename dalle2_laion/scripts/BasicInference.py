"""
This inference script is used to do basic inference without any bells and whistles.
Pass in text, get out image.
"""

from dalle2_laion.scripts import InferenceScript
from typing import Dict, List, Union
from PIL import Image as PILImage
import torch

class BasicInference(InferenceScript):
    def run(
        self,
        text: Union[str, List[str]],
        prior_cond_scale: float = None, decoder_cond_scale: float = None,  # Use defaults from config by default
        prior_sample_count: int = 1, decoder_sample_count: int = 1,
        prior_batch_size: int = 100, decoder_batch_size: int = 10,
        prior_num_samples_per_batch: int = 2
    ) -> Dict[str, Dict[int, List[PILImage.Image]]]:
        """
        Takes text and generates images.
        Returns a map from the text index to the image embedding index to a list of images generated for that embedding.
        """
        if isinstance(text, str):
            text = [text]
        self.print(f"Generating images for texts: {text}")
        self.print("Generating prior embeddings...")
        image_embedding_map = self._sample_prior(text, cond_scale=prior_cond_scale, sample_count=prior_sample_count, batch_size=prior_batch_size, num_samples_per_batch=prior_num_samples_per_batch)
        self.print("Finished generating prior embeddings.")
        # image_embedding_map is a map between the text index and the generated image embeddings
        image_embeddings: List[torch.Tensor] = []
        decoder_text = []  # The decoder also needs the text, but since we have repeated the text embeddings, we also need to repeat the text
        for i, original_text in enumerate(text):
            decoder_text.extend([original_text] * len(image_embedding_map[i]))
            image_embeddings.extend(image_embedding_map[i])
        # In order to get the original text from the image embeddings, we need to reverse the map
        image_embedding_index_reverse_map = {i: [] for i in range(len(text))}
        current_count = 0
        for i in range(len(text)):
            for _ in range(len(image_embedding_map[i])):
                image_embedding_index_reverse_map[i].append(current_count)
                current_count += 1
        # Now we can use the image embeddings to generate the images
        self.print(f"Grouped {len(text)} texts into {len(image_embeddings)} embeddings.")
        self.print("Sampling from decoder...")
        image_map = self._sample_decoder(text=decoder_text, image_embed=image_embeddings, cond_scale=decoder_cond_scale, sample_count=decoder_sample_count, batch_size=decoder_batch_size)
        self.print("Finished sampling from decoder.")
        # Now we will reconstruct a map from text to a map of img_embedding indices to list of images
        output_map: Dict[str, Dict[int, List[PILImage.Image]]] = {}
        for i, prompt in enumerate(text):
            output_map[prompt] = {}
            embedding_indices = image_embedding_index_reverse_map[i]
            for embedding_index in embedding_indices:
                output_map[prompt][embedding_index] = image_map[embedding_index]
        return output_map