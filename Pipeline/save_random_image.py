
import os
import sys
import random
from PIL import Image

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming btd1 and btd2 are instances of BrainTumorDataset
from Pipeline.pipeline import btd1, btd2


def save_random_images(dataset, save_dir, num_images=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(num_images):
        idx = random.randint(0, len(dataset) - 1)
        image, label = dataset[idx]
        image = transforms.ToPILImage()(image)
        image.save(os.path.join(save_dir, f"example_{i}.jpg"))


# Save random images from btd1 and btd2
save_random_images(btd1, "Data/Examples/btd1.jpg")
save_random_images(btd2, "Data/Examples/btd2.jpd")
