try:
    import gradio as gr
except ImportError:
    print("Please install gradio: `pip install gradio`")
    exit(1)
from pathlib import Path
from typing import Dict
from PIL import Image as PILImage
from dalle2_laion import ModelLoadConfig, DalleModelManager, utils
from dalle2_laion.scripts import BasicInference, ImageVariation, BasicInpainting

config_path = Path(__file__).parent / 'configs/gradio.example.json'
model_config = ModelLoadConfig.from_json_path(config_path)
model_manager = DalleModelManager(model_config)

output_path = Path(__file__).parent / 'output/gradio'
output_path.mkdir(parents=True, exist_ok=True)

def dream(text: str, samples_per_prompt: int):
    prompts = text.split('\n')[:8]

    script = BasicInference(model_manager, verbose=True)
    output = script.run(prompts, prior_sample_count=samples_per_prompt, decoder_batch_size=40)
    all_outputs = []
    for text, embedding_outputs in output.items():
        for index, embedding_output in embedding_outputs.items():
            all_outputs.extend(embedding_output)
    return all_outputs
dream_interface = gr.Interface(
    dream,
    inputs=[
        gr.Textbox(placeholder="A corgi wearing a top hat...", lines=8),
        gr.Slider(minimum=1, maximum=4, step=1, label="Samples per prompt", value=1)
    ],
    outputs=[
        gr.Gallery()
    ],
    title="Dalle2 Dream",
    description="Generate images from text. You can give a maximum of 8 prompts at a time. Any more will be ignored. Generation takes around 5 minutes so be patient.",
)

def variation(image: PILImage.Image, text: str, num_generations: int):
    print("Variation using text:", text)
    img = utils.center_crop_to_square(image)

    script = ImageVariation(model_manager, verbose=True)
    output = script.run([img], [text], sample_count=num_generations)
    all_outputs = []
    for index, embedding_output in output.items():
        all_outputs.extend(embedding_output)
    return all_outputs
variation_interface = gr.Interface(
    variation,
    inputs=[
        gr.Image(value="https://www.thefarmersdog.com/digest/wp-content/uploads/2021/12/corgi-top-1400x871.jpg", source="upload", interactive=True, type="pil"),
        gr.Text(),
        gr.Slider(minimum=1, maximum=6, label="Number to generate", value=2, step=1)
    ],
    outputs=[
        gr.Gallery()
    ],
    title="Dalle2 Variation",
    description="Generates images similar to the input image.\nGeneration takes around 5 minutes so be patient.",
)

def inpaint(image: Dict[str, PILImage.Image], text: str, num_generations: int):
    print("Inpainting using text:", text)
    img, mask = image['image'], image['mask']
    # Remove alpha from img
    img = img.convert('RGB')
    img = utils.center_crop_to_square(img)
    mask = utils.center_crop_to_square(mask)

    script = BasicInpainting(model_manager, verbose=True)
    mask = ~utils.get_mask_from_image(mask)
    output = script.run(images=[img], masks=[mask], text=[text], sample_count=num_generations)
    all_outputs = []
    for index, embedding_output in output.items():
        all_outputs.extend(embedding_output)
    return all_outputs
inpaint_interface = gr.Interface(
    inpaint,
    inputs=[
        gr.Image(value="https://www.thefarmersdog.com/digest/wp-content/uploads/2021/12/corgi-top-1400x871.jpg", source="upload", tool="sketch", interactive=True, type="pil"),
        gr.Text(),
        gr.Slider(minimum=1, maximum=6, label="Number to generate", value=2, step=1)
    ],
    outputs=[
        gr.Gallery()
    ],
    title="Dalle2 Inpainting",
    description="Fills in the details of areas you mask out.\nGeneration takes around 5 minutes so be patient.",
)

demo = gr.TabbedInterface(interface_list=[dream_interface, variation_interface, inpaint_interface], tab_names=["Dream", "Variation", "Inpaint"])

demo.launch(share=True, enable_queue=True)