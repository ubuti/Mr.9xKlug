import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import gradio as gr
import torch
from torchvision import transforms, models

# Load the trained model
model = models.resnet18()
model.load_state_dict(torch.load('Models/resnet18/resnet_mri_010824.pth'))
model.eval()

# Define prediction functions


def predict_mri(image, mode):
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[-1.3006, -1.2001, -0.9724], std=[0.8727, 0.8922, 0.8884]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)

    # Process the output based on the selected mode
    probability = torch.nn.functional.softmax(output[0], dim=0)
    if mode == "Binary Classification":
        result = {
            "No tumor": float(probability[0]),
            "Tumor": float(probability[1])
        }
    else:  # Multi-class classification
        result = {
            "No Tumor": float(probability[0]),
            "Glioma": float(probability[1]),
            "Meningioma": float(probability[2]),
            "Pituitary": float(probability[3])
        }
    return image, result


# Set the directory path
directory = "./Data/Datasets"

# Get a list of all .jpg files in the directory
file_list = [os.path.join(directory, file)
             for file in os.listdir(directory) if file.endswith(".jpg")]

# Create Gradio interface
interface = gr.Interface(
    fn=predict_mri,
    inputs=[gr.Image(type="pil"), gr.Radio(
        ["Binary Classification", "Multi-class Classification"])],
    outputs=[gr.Image(type="pil"), gr.Label()],
    examples=file_list  # Use the list of image paths as examples
)

interface.launch(share=True)
