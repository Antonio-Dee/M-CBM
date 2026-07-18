from pathlib import Path
import torch
import pandas as pd
from PIL import Image


class ISIC2018Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.image_ids = self.data.iloc[:, 0].values
        self.labels = self.data.iloc[:, 1:].values.astype(float)
        self.targets = self.labels.argmax(axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.get_image_path(idx)).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.float).argmax().long()
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_image_name(self, idx):
        return str(self.image_ids[idx])

    def get_image_path(self, idx):
        return self.resolve_image_path(self.get_image_name(idx))

    def resolve_image_path(self, image_name):
        jpg_path = Path(self.img_dir) / f"{image_name}.jpg"
        if jpg_path.exists():
            return jpg_path
        png_path = Path(self.img_dir) / f"{image_name}.png"
        if png_path.exists():
            return png_path
        raise FileNotFoundError(f"Could not find ISIC image {image_name} as .jpg or .png in {self.img_dir}.")
