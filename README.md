<h1 align="center">🎨🤖🦁<br/>DALLE2 LAION<br/></h1>

This is a collection of resources and tools for LAION's pre-trained DALLE-2 model. The official codebase can be found in [DALLE2-PyTorch](https://github.com/lucidrains/DALLE2-pytorch).

## Awknowledgments
A big thanks to...
* **[Zion](https://twitter.com/nousr_) & [Aidan](https://github.com/veldrovive)**: For developing the training code in DALLE2-PyTorch, training the models, and writing the inference pipeline found here.
* **[Romain Beaumont](https://github.com/rom1504)**: For their project management skills and guidance throughout the project.
* **[lucidrains](https://github.com/lucidrains)**: For spearheading the DALLE2 replication.
* **[LAION](https://laion.ai/)**: For providing support, feedback, and inspiration to the open-source AI community.
* **[StabilityAI](https://stability.ai/)**: For their generous donation of compute resources, without that these models would not exist.

## How To Use
There is a colab notebook available [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LAION-AI/dalle2-laion/blob/main/notebooks/dalle2_laion_alpha.ipynb) for those who do not have access to a powerful GPU, but want to play around.

Alternatively, you can download the latest checkpoints and use any of the inference pipelines available in this repository!

🚧 inference example usage 🚧

## Pre-Trained Models

For access to the latest official checkpoints please visit the official 🤗 huggingface [repo](https://huggingface.co/laion/DALLE2-PyTorch).

If you are interested in viewing the training runs for these models, they can be found here [decoder](https://wandb.ai/veldrovive/dalle2_train_decoder/runs/2yea5t0u) and [prior](https://wandb.ai/nousr_laion/dalle2_diffusion_prior). Please note that, due to the nature of rapid prototyping, these links may become outdated--we will try out best to keep them updated, however. If you notice that is the case, and would like up-to-date reports, feel free to open an issue!

## Community
If you make anything cool, we'd love to see! Join us on Discord <a href="https://discord.gg/xBPBXfcFHd"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>

## Issues
If you have questions, run into specific problems, or have an idea that you think should be integrated to our inference pipeline, feel free to open an issue here on github.

Some things we ask you to **please** keep in mind 😄
* Do _**not**_ open issues ⚠️ in the _DALLE2-PyTorch_ repo for any problems with code found here!
* If you find a bug 🐛, please report as much information as possible in your issue.
    * Things like your environment, library versions, and method of inference are imparitave to helping identify the issue.
* If you have a feature you would like to add or request ✋, please give a brief and detailed description of its use case and why you believe it would be helpful.
* Keep questions 🤔 relevant to this repository, if you have more general questions about dalle2, diffusion, upscaling, or AI in general, please join our discord and find an appropriate channel!
---

>DISCLAIMER:
>
>*DALLE2-LAION is not affilliated in any way with OpenAI. While the architecture closely follows the original paper, modifications have been made. The output of this model, opinions of the creators, and code provided does not reflect in any way that of OpenAI's.* 
