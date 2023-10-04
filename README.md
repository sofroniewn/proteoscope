# proteoscope

Generating images of protein subcellular localization from sequences

*Status: exploratory project, under open development & unpublished. All code may undergo breaking changes, and any insights or results are subject to change as more work is done - this is open science! If you have any questions/ comments open an issue.*

# image diffusion conditioned on protein sequence

Image diffusion conditioned on natural language has allowed the creation of striking examples of complex and imaginative scenes from simple text prompts (e.g. [stable diffusion](https://github.com/Stability-AI/stablediffusion)). Diffusion methods have begun to be applied within biology to problems like protein structure generation (e.g. [rf-diffusion](https://github.com/RosettaCommons/RFdiffusion)). In this work, we explore whether image based diffusion models can be conditioned on protein sequences to generate photorealistic images of the subcellular distribution of proteins.

The ability to accurately generate images of protein subcellular localization from sequence could shed light on the function and interactivity of proteins as information about protein complex formation can be inferred from images alone (see [cytoself](https://www.nature.com/articles/s41592-022-01541-z)). Furthermore such a model could allow for investigation of how sequence motifs and mutations might drive and affect protein trafficking. Protein sequence to subcellular localization models are often studied at the level of subcellular compartments (e.g. [deeploc2](https://academic.oup.com/nar/article/50/W1/W228/6576357)), though see [CELL-E](https://www.biorxiv.org/content/10.1101/2022.05.27.493774v1) which took a transformer based approach to sequence to image generation.


### data

As in CELL-E, we make use of [OpenCell](https://opencell.czbiohub.org/) data which contains subcellular localization images of just over 1000 proteins. We use data crops available from [cytoself](https://github.com/royerlab/cytoself/tree/main#data-availability) which results in over 1M multichannel images of protein and nucleus. We split our protein sequences into 80% training, 10% validation, and 10% test.


### architecture

We follow the [latent diffusion](https://github.com/CompVis/latent-diffusion) approach made popular by stable diffusion, by first training an autoencoder on images from our training sequence. Following the original implementation, we use a loss that includes a perceptual component and a discriminator.

We then perform diffusion in the latent space of the autoencoder using the [UNet2DConditionModel](https://huggingface.co/docs/diffusers/api/models/unet2d-cond) from the huggingface diffusers library. To condition on protein sequence, we pass the protein sequence through a pre-trained protein language model [ESM2](https://github.com/facebookresearch/esm) and extract embeddings from the last layer. We also condition on a spatially downsampled image of the nuclear distance. The nuclear distance image allows us to generate protein localizations relative to the nucleus, but without having to worry about flourescent bleed through of the protein channel that can occur when using the actual nuclear image.

We also dropout the conditioned sequence with fixed probability during training, so that we can perform classifier free guidence at inference time.

### evaluation

To evaluate performance of the model we leverage a pre-trained [cytoself](https://github.com/royerlab/cytoself) model, which includes a classifier that maps images to protein sequence. Note cytoself was also trained on our validation and test sequences, so we can use it to evaluate diffusion performance for these sequences too. We take our generated images and pass them through cytoself and then compare the cytoself classifications and distances in the cytoself latent space. These losses act like the perceputal losses often used in computer vision, but leveraging a model trained on protein images, rather than natural images.

We can also train a simple subcellular localization classifier from the cytoself latent space to look at if the generated images have the right subcellular localization at the organelle level.

## results (in progress)

This section will be updated as more experiments are done. Preliminary results indicate that it is possible to generate beuatiful images of localizations from the train protein sequences, but that overfitting to those sequences is a significant concern when looking at generations from the validation set.

![Alt text](image.png)

Random selection of images from seven proteins from the validation set. Top row real image, bottom row diffused image. You can see some good matches, but also some misses (like 4th column).

To deal with overfitting we are currently exploring a range of regularization and data augmentation strategies in sequence space. We are also exploring pre-training a [CLIP](https://github.com/openai/CLIP) model and then using that to guide diffusion as it may allow us to control the degree of overfitting more easily.

## acknowledgements

Nicholas Sofroniew works and is supported by the Chan Zuckberberg Initative. Thanks to Tom Hayes for invaluable conversations on model architectures and overall approach.
