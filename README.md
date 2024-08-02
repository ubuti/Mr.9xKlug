# Mr.9xKlug

## MRI Tumor Classification Model

This project uses a ResNet18 model to classify MRI images into four categories: No Tumor, Glioma, Meningioma, and Pituitary.

### Live Demo

Try out the model with our interactive demo:

[![Gradio](https://img.shields.io/badge/Gradio-Live%20Demo-blue)](https://fe3d9890bf60d19919.gradio.live)


### How to Use

1. Click on the "Live Demo" link above.
2. Upload an MRI image or use one of the provided examples.
3. The model will classify the image and provide probabilities for each tumor type.

### Local Installation

If you want to run the model locally:

1. Clone this repository
2. Install dependencies: `conda env create -f environment.yml`
3. Run the Gradio interface: `python/python3 -u app.py`

### Model Details

- Architecture: torchvision.models.resnet18 was used and initialised with the weights and configuration of [BehradG](https://huggingface.co/BehradG/resnet-18-finetuned-MRI-Brain/tree/main).
- Training Data: Two datasets available via *kaggle* [1](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) [2] (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) where joined.
- Performance: Model exceeds $98%$ accuracy.

### Contact

Other authors contribution in form of dataset contribution and model pretraining is very much appreciated!
