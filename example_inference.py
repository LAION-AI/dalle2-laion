from dalle2_laion import DalleModelManager, ModelLoadConfig, utils
from dalle2_laion.scripts import BasicInference, ImageVariation, BasicInpainting
from typing import List
import os
import click
from pathlib import Path
import torch

@click.group()
@click.option('--verbose', '-v', is_flag=True, default=False, help='Print verbose output.')
@click.pass_context
def inference(ctx, verbose):
    ctx.obj['verbose'] = verbose

@inference.command()
@click.option('--model-config', default='./configs/upsampler.example.json', help='Path to model config file')
@click.option('--output-path', default='./output/basic/', help='Path to output directory')
@click.pass_context
def dream(ctx, model_config: str, output_path: str):
    verbose = ctx.obj['verbose']
    prompts = []
    print("Enter your prompts one by one. Enter an empty prompt to finish.")
    while True:
        prompt = click.prompt(f'Prompt {len(prompts)+1}', default='', type=str, show_default=False)
        if prompt == '':
            break
        prompts.append(prompt)
    num_prior_samples = click.prompt('How many samples would you like to generate for each prompt?', default=1, type=int)

    dreamer: BasicInference = BasicInference.create(model_config, verbose=verbose)
    output_map = dreamer.run(prompts, prior_sample_count=num_prior_samples)
    os.makedirs(output_path, exist_ok=True)
    for text in output_map:
        for embedding_index in output_map[text]:
            for image in output_map[text][embedding_index]:
                image.save(os.path.join(output_path, f"{text}_{embedding_index}.png"))

@inference.command()
@click.option('--model-config', default='./configs/variation.example.json', help='Path to model config file')
@click.option('--output-path', default='./output/variations/', help='Path to output directory')
@click.pass_context
def variation(ctx, model_config: str, output_path: str):
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
                text_prompt = click.prompt(f'Prompt for {image_path.name}', default=image_path.stem, type=str, show_default=True)
                text_prompts.append(text_prompt)
        image_filepaths.extend(new_image_paths)

    print(f"Found {len(image_filepaths)} images.")
    images = utils.get_images_from_paths(image_filepaths)
    num_samples = click.prompt('How many samples would you like to generate for each image?', default=1, type=int)

    output_map = variation.run(images, text=text_prompts, sample_count=num_samples)
    os.makedirs(output_path, exist_ok=True)
    for file_index, generation_list in output_map.items():
        file = image_filepaths[file_index].stem
        for i, image in enumerate(generation_list):
            image.save(os.path.join(output_path, f"{file}_{i}.png"))

@inference.command()
@click.option('--model-config', default='./configs/variation.example.json', help='Path to model config file')
@click.option('--output-path', default='./output/variations/', help='Path to output directory')
@click.pass_context
def inpaint(ctx, model_config: str, output_path: str):
    raise NotImplementedError('Image inpainting is not implemented in the decoder yet.')
    verbose = ctx.obj['verbose']
    inpainting: BasicInpainting = BasicInpainting.create(model_config, verbose=verbose)
    requires_prompts = variation.model_manager.decoder_info.data_requirements.text_encoding
    image_filepaths: List[Path] = []
    mask_filepaths: List[Path] = []
    text_prompts: List[str] = [] if requires_prompts else None
    print("You will be entering the paths to your images and masks one at a time. Enter an empty image path to continue")
    if requires_prompts:
        print("This decoder also requires text. You will need to enter a prompt for each image you use.")
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
        image_filepaths.append(image_filepath)
        mask_filepaths.append(mask_filepath)
        if requires_prompts:
            text_prompt = click.prompt(f'Prompt for {image_filepath.name}', default=image_filepath.stem, type=str, show_default=True)
            text_prompts.append(text_prompt)
    print(f"Found {len(image_filepaths)} images.")
    images = utils.get_images_from_paths(image_filepaths)
    mask_images = utils.get_images_from_paths(mask_filepaths)
    masks = torch.stack([utils.get_mask_from_image(mask_image) for mask_image in mask_images])
    num_samples = click.prompt('How many samples would you like to generate for each image?', default=1, type=int)
    output_map = inpainting.run(images, masks, text=text_prompts, sample_count=num_samples)
    os.makedirs(output_path, exist_ok=True)
    for file_index, generation_list in output_map.items():
        file = image_filepaths[file_index].stem
        for i, image in enumerate(generation_list):
            image.save(os.path.join(output_path, f"{file}_{i}.png"))


if __name__ == "__main__":
    inference(obj={})