# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import re
import tempfile
import typing

import numpy as np
import torch
from dalle2_pytorch import (DALLE2, DiffusionPrior, DiffusionPriorNetwork,
                            OpenAIClipAdapter, train_configs)
from dalle2_pytorch.tokenizer import tokenizer
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from cog import BaseModel, BasePredictor, Input, Path


class Output(BaseModel):
    image: Path
    prompt: str


def load_decoder(decoder_state_dict_path, config_file_path="config.json"):
    config = train_configs.TrainDecoderConfig.from_json_path(config_file_path)
    decoder_text_conditioned = config.decoder.condition_on_text_encodings
    clip_config = config.decoder.clip
    config.decoder.clip = None
    print("Decoder conditioned on text", decoder_text_conditioned)
    decoder = config.decoder.create()
    decoder_state_dict = torch.load(decoder_state_dict_path, map_location="cpu")
    decoder.load_state_dict(decoder_state_dict, strict=False)
    del decoder_state_dict
    decoder.eval()
    decoder = decoder.to(device)
    return decoder, clip_config, decoder_text_conditioned


def similarity(image_embedding, text_embedding):
    image_embedding = image_embedding / np.linalg.norm(image_embedding)
    text_embedding = text_embedding / np.linalg.norm(text_embedding)
    return np.inner(image_embedding, text_embedding)


def rerank_and_sample(image_embeddings, text_embedding, samples=None, strategy="top"):
    """
    Here we take the prompt, generate n number of embeddings and rerank them by cosine similarity to the text embedding,
    then take a linspace of N and sample the decoder with those embeddings to see the variation in the performance of the prior
    """
    if samples is None:
        samples = len(image_embeddings)
    reranked = sorted(
        list(image_embeddings), key=lambda img_emb: similarity(img_emb, text_embedding)
    )
    if strategy == "top":
        sampled_embeddings = np.array(reranked[-samples:])
    elif strategy == "even":
        sample_indices = np.linspace(0, len(reranked) - 1, num=samples, dtype=int)
        sampled_embeddings = np.array([reranked[i] for i in sample_indices])
    rankings = [similarity(emb, text_embedding) for emb in sampled_embeddings]
    print(rankings, rankings[0], rankings[-1])
    return sampled_embeddings


def load_prior(model_path):
    """
    Loads the prior model and returns it.
    **Note** - this is a modified version of the original function to allow for the use of slim fp16 checkpoints.
    """
    prior_network = DiffusionPriorNetwork(
        dim=768,
        depth=24,
        dim_head=64,
        heads=32,
        normformer=True,
        attn_dropout=5e-2,
        ff_dropout=5e-2,
        num_time_embeds=1,
        num_image_embeds=1,
        num_text_embeds=1,
        num_timesteps=1000,
        ff_mult=4,
    )

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        clip=OpenAIClipAdapter("ViT-L/14"),
        image_embed_dim=768,
        timesteps=1000,
        cond_drop_prob=0.1,
        loss_type="l2",
        condition_on_text_encodings=True,
    )
    state_dict = torch.load(model_path, map_location="cpu")
    diffusion_prior.load_state_dict(state_dict, strict=True)
    diffusion_prior.eval()
    diffusion_prior.to(device)
    return diffusion_prior


def np_images_to_cog_outputs(
    np_images, prior_repeat, decoder_repeat, prompts, upscale=4
) -> typing.List[Path]:
    curr_index = 0
    final_outputs = []
    # temp_dir = Path(tempfile.mkdtemp())
    temp_dir = Path("outputs")
    temp_dir.mkdir(exist_ok=True)

    for prompt in prompts:
        # clean caption = remove punctuation and lowercase
        clean_prompt = re.sub(r"[^\w\s]", "", prompt.lower())
        clean_prompt = re.sub(r"\s+", "_", clean_prompt)
        for prior_index in range(prior_repeat):
            for decoder_index in range(decoder_repeat):
                img = np_images[curr_index]
                image = Image.fromarray(np.uint8(img * 255))
                image = image.resize([dim * upscale for dim in image.size])
                image_path = temp_dir.joinpath(
                    f"{clean_prompt}_{prior_index:03}_{decoder_index:03}_{curr_index:03}.png"
                )
                image.save(image_path)
                final_outputs.append(image_path)
                curr_index += 1
    return final_outputs


def parse_prompts(prompts: str, delimiter="|") -> typing.List[str]:
    return list(filter(lambda v: len(v) > 0, prompts.split(delimiter)))


class Predictor(BasePredictor):
    @torch.inference_mode()
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.decoder, self.clip_config, self.decoder_text_conditioned = load_decoder(
            "decoder_ema_fp16.pth", "config.json"
        )

        self.diffusion_prior = load_prior("./prior_ema_fp16.pth")

        self.clip = None
        if self.clip_config is not None:
            self.clip = self.clip_config.create()

    @torch.inference_mode()
    @torch.cuda.amp.autocast(enabled=True)
    def predict(
        self,
        text_input: str = Input(
            description="Text you want to visualize", default=""
        ),
        prior_num_candidates: int = Input(ge=1, le=4, description="Number of candidate embeds from prior model to generate", default=3),
        prior_guidance_scale: float = Input(
            ge=0.0, le=10.0, description="Prior Cond Scale", default=4.0
        ),
        img_decoder_num_generations: int = Input(ge=1, le=5, description="Number of final images to generate from embeddings.", default=1),
        decoder_guidance_scale: float = Input(
            ge=0.0, le=10.0, description="Decoder Cond Scale", default=4.0
        ),
    ) -> typing.List[Path]:
        prompts = text_input.split("|")
        if len(prompts) == 0:
            prompts = [text_input]

        prior_text_input = []
        for prompt in prompts:
            for _ in range(prior_num_candidates):
                prior_text_input.append(prompt)

        tokens = tokenizer.tokenize(prior_text_input).to(device)

        # Prior
        print("Running prior")
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            image_embed = self.diffusion_prior.sample(
                tokens, cond_scale=prior_guidance_scale
            )
        image_embed = image_embed.detach().cpu().numpy()
        np.save("img_emb_prior.npy", image_embed)

        embeddings = np.repeat(image_embed, img_decoder_num_generations, axis=0)
        embeddings = torch.from_numpy(embeddings).float().to(device)

        # Decoder
        print("Running decoder")
        if self.decoder_text_conditioned:
            print("Generating clip embeddings")
            _, text_encoding, text_mask = self.clip.embed_text(tokens)
            text_encoding = text_encoding.to(device)
            text_mask = text_mask.to(device)

            images = self.decoder.sample(
                embeddings,
                text_encodings=text_encoding,
                text_mask=text_mask,
                cond_scale=decoder_guidance_scale,
            )
        else:
            print("Not generating clip embeddings")
            images = self.decoder.sample(
                embeddings, text=None, cond_scale=decoder_guidance_scale
            )

        np_images = images.cpu().permute(0, 2, 3, 1)
        np.save("images_decoder.npy", np_images)
        return np_images_to_cog_outputs(
            np_images, prior_num_candidates, img_decoder_num_generations, prompts, upscale=4
        )