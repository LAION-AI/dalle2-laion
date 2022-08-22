# DALLE2 LAION Inferencing
In order to simplify running generalized inferences against a dalle2 model, we have created a three stage process to make any inference possible.

## Simple inference
If you are not interested in the details, there are two scripts that just run text to image. In either case, you will need a powerful graphics card (with at least 16gb VRAM).

The easiest method is to use the gradio interface with you can start by navigating to the root folder and running `python gradio_inference.py`.

For a lower level platform and a place to develop your own scripts, it is easier to use the cli with `python example_inference.py`.

## Configuring the Model
Dalle2 is a multistage model constructed out of an encoder, a prior, and some number of decoders. In order to run inference, we must join together all these separately trained components into one model. To do that, we must point the model manager to the model files and tell it how to load them. This is done through a configuration `.json` file.

Generally, an end user should not attempt to write their own config unless they have trained the models themselves. This is because all models must have been trained to be compatible and trying to stitch together models where the components are not compatible will result in nonsense results.

The general structure of the config is a dictionary with the following keys:
`clip`, `prior`, and `decoder`. Each of these contains a dictionary with more specific information.

One repeating pattern in the config is the `File` type. Many of the config options are of type `File` and any using this pattern will be referenced as such.

A file has the following configuration:
| Key | Description |
| --- | --- |
| `load_type` | Either `local` or `url`. |
| `path` | The path to the file or the url of the file. |
| `checksum_file_path` | **Optional**: The path or url of the checksum of the file. This is generated automatically for files stored using huggingface repositories and does not need to be specified. |
| `cache_dir` | **Optional**: The directory to cache the file in. Should be used for `url` load type files. If not provided the file will be re-downloaded every time it is used. |
| `filename_override` | **Optional**: The name of the file to use instead of the default filename. |

### CLIP
Dalle2 uses CLIP as the encoder to turn an image or text into the encoded representation. CLIP produces embeddings for images and text in a shared representation space, but these embeddings are not equal for images and text that match.

Under the `clip` configuration, there are the following options:
| Key | Description |
| --- | --- |
| `make` | The make of the CLIP model to use. Options are `openai`, `x-clip`, or `coca`. |
| `model` | The specific model to use. These are defined by which option you choose for `make`. This should be the same model as the one used during training. |

### Diffusion Prior
The decoders will take an image embedding and turn it into an image, but CLIP has only produced a text embedding. The purpose of the prior is to take the text embeddings and convert them into image embeddings.

Under the `prior` configuration, there are the following options:
| Key | Description |
| --- | --- |
| `load_model_from` | A `File` configuration that points to the model to load. |
| `load_config_from` | **Optional**: If this is an old model, the config must be loaded separately with this `File` configuration. For the vast majority of cases this is not necessary to specify. |
| `default_sample_timesteps` | **Optional**: The number of sampling timesteps to use by default. If not specified this uses the number the prior was trained with. |
| `default_cond_scale` | **Optional**: The default conditioning scale. If not specified 1 is used. |

### Decoders
The initial decoder's purpose is to take the image embedding and turn it into a low resolution image (generally 64x64). Further decoders act as upsamplers and take the low resolution image and turn it into a higher resolution image (generally from 64x64 to 256x256 and then 256x256 to 1024x1024).

This is the most complex configuration since it involves loading multiple models. Each individual decoder has the following configuration which we will call a `SingleDecoderConfig`:

| Key | Description |
| --- | --- |
| `load_model_from` | A `File` configuration that points to the model to load. |
| `load_config_from` | **Optional**: If this is an old model, the config must be loaded separately with this `File` configuration. |
| `unet_numbers` | An array of integers that specify which unet numbers this decoder should be used for. Together, all `SingleDecoderConfig`s must include all numbers in the range [1, max unet number]. |
| `default_sample_timesteps` | **Optional**: An array of numbers that specify the sample timesteps to use for each unet being loaded from this model. |
| `default_cond_scale` | **Optional**: An array of numbers that specify the conditioning scale to use for each unet being loaded from this model. |

Under the `decoder` configuration, there are the following options:
| Key | Description |
| --- | --- |
| `unet_sources` | An array of `SingleDecoderConfig` configurations that point to the models to use for each unet. |

## Using the Configuration
The configuration is used to load the models with the `ModelManager`.

```python
from dalle2_laion import ModelLoadConfig, DalleModelManager

model_config = ModelLoadConfig.from_json_path("path/to/config.json")
model_manager = DalleModelManager(model_config)
```

This will download the requested models, check for updates using the checksums if provided, and load the model into RAM. For larger models, the ram requirements may be too large for most consumer machines to run.

## Inference Scripts
Inference scripts are convenient wrappers that make it easy to run a specific task. In the [scripts](dalle2_laion/scripts) folder there are a few basic scripts ready to run inference, but you can also make your own by implementing the `InferenceScript` abstract class.

In general, an inference script will take a model manager as the first argument to the constructor and then any other arguments that are specific to the task.

When inheriting from `InferenceScript`, the most important methods are `_sample_prior` and `_sample_decoder`.

`_sample_prior` runs the prior sampling loop and returns image embeddings.
It takes the following arguments:
| Argument | Description |
| --- | --- |
| `text_or_tokens` | A list of strings or tokenized strings to use as the conditioning. Encoding are automatically generated using the specified CLIP. |
| `cond_scale` | **Optional**: A conditioning scale to use. If not specified the default from the config is used. |
| `sample_count` | **Optional**: The number of samples to take for each text input. If not specified the default is 1. |
| `batch_size` | **Optional**: The batch size to use when sampling. If not specified the default is 100. |
| `num_samples_per_batch` | **Optional**: The number of samples to rerank when generating an image embedding. You should usually not touch this and the default is 2. |

`_sample_decoder` runs the decoder sampling loop and returns an image.
It takes the following arguments:
| Argument | Description |
| --- | --- |
| `images` or `image_embed` | Exactly one of these must be passed. `images` is an array of PIL images. If it is passed, the image embeddings will be generated using these images so variations of them will be generated. `image_embed` is an array of tensors representing precomputed image embeddings generated by the prior or by CLIP. |
| `text` or `text_encoding` | Exactly one of these must be passed if the decoder has been conditioned on text. |
| `inpaint_images` | **Optional**: If the inpainting feature is being used, this is an array of PIL images that will be masked and inpainted. |
| `inpaint_image_masks` | A list of 2D boolean tensors that indicate which pixels in the inpaint images should be inpainted. |
| `cond_scale` | **Optional**: A conditioning scale to use. If not specified the default from the config is used. |
| `sample_count` | **Optional**: The number of samples to take for each text input. If not specified the default is 1. |
| `batch_size` | **Optional**: The batch size to use when sampling. If not specified the default is 10. |

A simple implementation of a inference script is:

```python
from dalle2_laion import ModelLoadConfig, DalleModelManager
from dalle2_laion.scripts import InferenceScript

class ExampleInference(InferenceScript):
    def run(self, text: str) -> PILImage.Image:
        """
        Takes a string and returns a single image.
        """
        text = [text]
        image_embedding_map = self._sample_prior(text)
        image_embedding = image_embedding_map[0][0]
        image_map = self._sample_decoder(text=text, image_embed=image_embedding)
        return image_map[0][0]

model_config = ModelLoadConfig.from_json_path("path/to/config.json")
model_manager = DalleModelManager(model_config)
inference = ExampleInference(model_manager)
image = inference.run("Hello World")
```
