import torch
import gradio as gr
from PIL import Image
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import os

# Initialize the model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 4)

# Load the state dictionary from the .pt file
state_dict = torch.load(
    './Models/resnet18/resnet_mri_062024.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# Define the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

def get_normalize_vals(img_path):
    """Return the mean and std for a chosen image in order to use Normalize in the transforms method of the dataset."""
    img = Image.open(img_path)
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img)
    means = torch.mean(img_tensor, dim=[1,2])
    stds = torch.std(img_tensor, dim=[1,2])
    return means, stds

path = "./Data/brain_tumor_4variants/Training/glioma_tumor/gg (1).jpg"
m, s = get_normalize_vals(path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(
        mean=m, std=s),
    transforms.Lambda(lambda x: torch.clamp(x,0,1))
])


def predict_mri(image):
    # Convert to RGB (in case the input is grayscale)
    image = image.convert("RGB")

    # Transform the image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

    # Get the predicted class and probability
    _, predicted = torch.max(probabilities, 1)
    pred_class = predicted.item()
    pred_prob = probabilities[0, pred_class].item()

    # Define class labels
    class_labels = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]

    # Create result dictionary
    result = {label: float(prob) for label, prob in zip(
        class_labels, probabilities[0].tolist())}

    return result


# Set up example images
directory = "./Data/"
file_list = []
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".jpg"):
            tumor_type = os.path.basename(root)
            file_list.append([os.path.join(root, file), tumor_type])

# Create Gradio interface
interface = gr.Interface(
    fn=predict_mri,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    examples=file_list,
    title="MRI Tumor Classification",
    description="Upload an MRI image to classify the type of tumor."
)

# Launch the interface
interface.launch(share=True)
