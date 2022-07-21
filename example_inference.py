from dalle2_laion import DalleModelManager, ModelLoadConfig
from dalle2_laion.scripts import BasicInference
import os
import click

@click.command()
@click.option('--model-config', default='./configs/upsampler.example.json', help='Path to model config file')
@click.option('--output-path', default='./output/', help='Path to output directory')
def run_basic_inference(model_config: str, output_path: str):
    prompts = []
    print("Enter your prompts one by one. Enter an empty prompt to finish.")
    while True:
        prompt = click.prompt(f'Prompt {len(prompts)+1} ', default='', type=str)
        if prompt == '':
            break
        prompts.append(prompt)
    num_prior_samples = click.prompt('How many samples would you like to generate for each prompt?', default=1, type=int)
    
    print(f"Generating image for prompts: {prompts}")
    config = ModelLoadConfig.from_json_path(model_config)
    model_manager = DalleModelManager(config)
    dreamer = BasicInference(model_manager)
    output_map = dreamer.dream(prompts, prior_sample_count=num_prior_samples)
    os.makedirs(output_path, exist_ok=True)
    for text in output_map:
        for embedding_index in output_map[text]:
            for image in output_map[text][embedding_index]:
                # Save the image
                image.save(os.path.join(output_path, f"{text}_{embedding_index}.png"))

if __name__ == "__main__":
    run_basic_inference()