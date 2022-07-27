from dalle2_laion import DalleModelManager, ModelLoadConfig, utils
from dalle2_laion.scripts import BasicInference, ImageVariation, BasicInpainting
from typing import List
import os
import click
from pathlib import Path
import json
import torch

@click.group()
@click.option('--verbose', '-v', is_flag=True, default=False, help='Print verbose output.')
@click.pass_context
def inference(ctx, verbose):
    ctx.obj['verbose'] = verbose

@inference.command()
@click.option('--model-config', default='./configs/upsampler.example.json', help='Path to model config file')
@click.option('--output-path', default='./output/basic/', help='Path to output directory')
@click.option('--decoder-batch-size', default=10, help='Batch size for decoder')
@click.pass_context
def dream(ctx, model_config: str, output_path: str, decoder_batch_size: int):
    verbose = ctx.obj['verbose']
    prompts = []
    print("Enter your prompts one by one. Enter an empty prompt to finish.")
    while True:
        prompt = click.prompt(f'Prompt {len(prompts)+1}', default='', type=str, show_default=False)
        if prompt == '':
            break
        prompt_file = Path(prompt)
        if utils.is_text_file(prompt_file):
            # Then we can read the prompts line by line
            with open(prompt_file, 'r') as f:
                for line in f:
                    prompts.append(line.strip())
        elif utils.is_json_file(prompt_file):
            # Then we assume this is an array of prompts
            with open(prompt_file, 'r') as f:
                prompts.extend(json.load(f))
        else:
            prompts.append(prompt)
    num_prior_samples = click.prompt('How many samples would you like to generate for each prompt?', default=1, type=int)

    dreamer: BasicInference = BasicInference.create(model_config, verbose=verbose)
    output_map = dreamer.run(prompts, prior_sample_count=num_prior_samples, decoder_batch_size=decoder_batch_size)
    os.makedirs(output_path, exist_ok=True)
    for text in output_map:
        for embedding_index in output_map[text]:
            for image in output_map[text][embedding_index]:
                image.save(os.path.join(output_path, f"{text}_{embedding_index}.png"))

@inference.command()
@click.option('--model-config', default='./configs/variation.example.json', help='Path to model config file')
@click.option('--output-path', default='./output/variations/', help='Path to output directory')
@click.option('--decoder-batch-size', default=10, help='Batch size for decoder')
@click.pass_context
def variation(ctx, model_config: str, output_path: str, decoder_batch_size: int):
    verbose = ctx.obj['verbose']
    variation: ImageVariation = ImageVariation.create(model_config, verbose=verbose)
    decoder_data_requirements = variation.model_manager.decoder_info.data_requirements
    image_filepaths: List[Path] = []
    text_prompts: List[str] = [] if decoder_data_requirements.text_encoding else None

    print("Enter paths to your images. If you specify a directory all images within will be added. Enter an empty line to finish.")
    if decoder_data_requirements.text_encoding:
        print("This decoder was also conditioned on text. You will need to enter a prompt for each image you use.")

    while True:
        image_filepath: Path = click.prompt(f'File {len(image_filepaths)+1}', default=Path(), type=Path, show_default=False)
        if image_filepath == Path():
            break
        if image_filepath.is_dir():
            new_image_paths = utils.get_images_in_dir(image_filepath)
        elif utils.is_image_file(image_filepath):
            new_image_paths = [image_filepath]
        else:
            print(f"{image_filepath} is not a valid image file.")
            continue

        if decoder_data_requirements.text_encoding:
            for image_path in new_image_paths:
                text_prompt = click.prompt(f'Prompt for {image_path.name}', default=utils.get_prompt_from_filestem(image_path.stem), type=str, show_default=True)
                text_prompts.append(text_prompt)
        image_filepaths.extend(new_image_paths)

    print(f"Found {len(image_filepaths)} images.")
    images = utils.get_images_from_paths(image_filepaths)
    num_samples = click.prompt('How many samples would you like to generate for each image?', default=1, type=int)

    output_map = variation.run(images, text=text_prompts, sample_count=num_samples, batch_size=decoder_batch_size)
    os.makedirs(output_path, exist_ok=True)
    for file_index, generation_list in output_map.items():
        file = image_filepaths[file_index].stem
        for i, image in enumerate(generation_list):
            image.save(os.path.join(output_path, f"{file}_{i}.png"))

@inference.command()
@click.option('--model-config', default='./configs/upsampler.example.json', help='Path to model config file')
@click.option('--output-path', default='./output/inpaint/', help='Path to output directory')
@click.pass_context
def inpaint(ctx, model_config: str, output_path: str):
    verbose = ctx.obj['verbose']
    inpainting: BasicInpainting = BasicInpainting.create(model_config, verbose=verbose)
    image_filepaths: List[Path] = []
    mask_filepaths: List[Path] = []
    text_prompts: List[str] = []
    print("You will be entering the paths to your images and masks one at a time. Enter an empty image path to continue")
    while True:
        image_filepath: Path = click.prompt(f'File {len(image_filepaths)+1}', default=Path(), type=Path, show_default=False)
        if image_filepath == Path():
            break
        if not utils.is_image_file(image_filepath):
            print(f"{image_filepath} is not a valid image file.")
            continue
        mask_filepath: Path = click.prompt(f'Mask for {image_filepath.name}', default=Path(), type=Path, show_default=False)
        if not utils.is_image_file(mask_filepath):
            print(f"{mask_filepath} is not a valid image file.")
            continue
        text_prompt = click.prompt(f'Prompt for {image_filepath.name}', default=utils.get_prompt_from_filestem(image_filepath.stem), type=str, show_default=True)

        image_filepaths.append(image_filepath)
        mask_filepaths.append(mask_filepath)
        text_prompts.append(text_prompt)
            
    print(f"Found {len(image_filepaths)} images.")
    images = utils.get_images_from_paths(image_filepaths)
    mask_images = utils.get_images_from_paths(mask_filepaths)
    min_image_size = float('inf')
    for i, image, mask_image, filepath in zip(range(len(images)), images, mask_images, image_filepaths):
        assert image.size == mask_image.size, f"Image {filepath.name} has different dimensions than mask {mask_filepaths[i].name}"
        if min(image.size) < min_image_size:
            min_image_size = min(image.size)
        if image.size[1] != image.size[0]:
            print(f"{filepath.name} is not a square image. It will be center cropped into a square.")
            images[i] = utils.center_crop_to_square(image)
            mask_images[i] = utils.center_crop_to_square(mask_image)
    print(f"Minimum image size is {min_image_size}. All images will be resized to this size for inference.")
    images = [image.resize((min_image_size, min_image_size)) for image in images]
    mask_images = [mask_image.resize((min_image_size, min_image_size)) for mask_image in mask_images]

    masks = [utils.get_mask_from_image(mask_image) for mask_image in mask_images]
    num_samples = click.prompt('How many samples would you like to generate for each image?', default=1, type=int)
    output_map = inpainting.run(images, masks, text=text_prompts, sample_count=num_samples)
    os.makedirs(output_path, exist_ok=True)
    for file_index, generation_list in output_map.items():
        file = image_filepaths[file_index].stem
        for i, image in enumerate(generation_list):
            image.save(os.path.join(output_path, f"{file}_{i}.png"))


if __name__ == "__main__":
    inference(obj={})