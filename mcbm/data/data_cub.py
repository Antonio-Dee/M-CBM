import os
from pathlib import Path
import torch
import pandas as pd
from PIL import Image


class CUBDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform

        images = pd.read_csv(os.path.join(root_dir, "images.txt"), sep=" ", header=None, names=["img_id", "filepath"])
        labels = pd.read_csv(os.path.join(root_dir, "image_class_labels.txt"), sep=" ", header=None, names=["img_id", "label"])
        split = pd.read_csv(os.path.join(root_dir, "train_test_split.txt"), sep=" ", header=None, names=["img_id", "is_train"])

        data = images.merge(labels, on="img_id").merge(split, on="img_id")
        self.data = data[data["is_train"] == int(train)].reset_index(drop=True)
        self.targets = (self.data["label"] - 1).astype(int).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(self.get_image_path(idx)).convert("RGB")
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_image_name(self, idx):
        return str(self.data.iloc[idx]["filepath"])

    def get_image_path(self, idx):
        return self.resolve_image_path(self.get_image_name(idx))

    def resolve_image_path(self, image_name):
        return Path(self.root_dir) / "images" / image_name
