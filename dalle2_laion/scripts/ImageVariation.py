"""
This script generate image embeddings directly with clip instead of using the prior.
Put image in, get image out... but different.
"""

from dalle2_laion.scripts import InferenceScript
from typing import List, Dict, Optional
from PIL import Image as PILImage

class ImageVariation(InferenceScript):
    def run(
        self,
        images: List[PILImage.Image],
        text: Optional[List[str]],
        cond_scale: float = None,  # Use defaults from config by default
        sample_count: int = 1,
        batch_size: int = 10
    ) -> Dict[int, List[PILImage.Image]]:
        self.print("Running decoder...")
        image_map = self._sample_decoder(images=images, text=text, cond_scale=cond_scale, sample_count=sample_count, batch_size=batch_size)
        self.print("Finished running decoder.")
        return image_map

