import os
import random
import gradio as gr
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from train import BrainTumorDataset

# Initialize the model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 4)

# Load the state dictionary from the .pt file
state_dict = torch.load(
    './Models/resnet18/ResNet_mri__62026.pth', map_location=torch.device('cpu'), weights_only=False)
model.load_state_dict(state_dict)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

m = torch.tensor([0.1834, 0.1834, 0.1835])
s = torch.tensor([0.1959, 0.1959, 0.1960])

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(
        mean=m, std=s),
    transforms.Lambda(lambda x: torch.clamp(x,0,1))
])

label_mapping = {
    "no_tumor": 0,
    "notumor": 0,
    "glioma_tumor": 1,
    "glioma": 1,
    "meningioma_tumor": 2,
    "meningioma": 2,
    "pituitary_tumor": 3,
    "pituitary": 3,
}

class_names = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]

# Prepare dataset for random img show
data = []
val_directories = ["./Data/brain_tumor_4variants2/Testing"]
for directory in val_directories:
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            file_list = [file for file in os.listdir(
                subdir_path) if file.endswith(".jpg")]
            
            data.extend([(os.path.join(subdir_path, file),
                          label_mapping[subdir]) for file in file_list])

dataset = pd.DataFrame(
    data, columns=["Image_Path", "Label"])
btd3 = BrainTumorDataset(data_frame=dataset, transform=transform)
sample_items = list(data)
sample_paths = [path for path, _ in sample_items]
sample_true_labels = [class_names[label] for _, label in sample_items]

def eval_perform(image, true_label=None):
    
    model.eval()
    if image is None:
        image_tensor, _ = btd3[random.randint(0, len(btd3) - 1)]
        image_tensor = image_tensor.unsqueeze(0).to(device)
    else:
        image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)

    probabilities = torch.softmax(logits, dim=1)[0]
    prediction_idx = torch.argmax(probabilities).item()
    prediction = class_names[prediction_idx]
    probs = {
        "No Tumor": round(float(probabilities[0].item()), 4),
        "Glioma": round(float(probabilities[1].item()), 4),
        "Meningioma": round(float(probabilities[2].item()), 4),
        "Pituitary": round(float(probabilities[3].item()), 4),
    }
    if true_label is None:
        info_text = ""
    else:
        info_text = f"True label: {true_label}"
    return probs, info_text


def pick_random_sample():
    random_index = random.randrange(len(sample_paths))
    selected_path = sample_paths[random_index]
    image = Image.open(selected_path).convert("RGB")
    true_label = sample_true_labels[random_index]
    probs, info_text = eval_perform(image, true_label=true_label)
    return image, probs, info_text


with gr.Blocks(title="MRI Tumor Classification") as interface:
    gr.Markdown("# MRI Tumor Classification")
    gr.Markdown("Upload an MRI scan and compare it with a random sample.")

    with gr.Row():
        with gr.Column():
            upload_image = gr.Image(label="Upload MRI Scan", type="pil")
            upload_output = gr.Label(num_top_classes=4, label="Upload Prediction")
            upload_info = gr.Textbox(label="Upload info", lines=2)

        with gr.Column():
            sample_image = gr.Image(label="Random btd3 Sample", type="pil")
            sample_output = gr.Label(num_top_classes=4, label="Sample Prediction")
            sample_info = gr.Textbox(label="Sample info", lines=1)

    random_btn = gr.Button("🎲 Show random btd3 image")

    random_btn.click(
        fn=pick_random_sample,
        outputs=[sample_image, sample_output, sample_info],
    )

    upload_image.change(
        fn=lambda image: eval_perform(image),
        inputs=[upload_image],
        outputs=[upload_output, upload_info],
    )

interface.launch(share=True)
