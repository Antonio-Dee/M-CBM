import os
from pathlib import Path
import torch
from PIL import Image


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split="train", transform=None, class_to_idx=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform

        classes = sorted(entry.name for entry in os.scandir(self.root_dir) if entry.is_dir())
        if class_to_idx is None:
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        self.samples = []
        self.targets = []
        for cls_name in classes:
            cls_folder = os.path.join(self.root_dir, cls_name)
            for fname in sorted(os.listdir(cls_folder)):
                if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                    img_path = os.path.join(cls_folder, fname)
                    target = self.class_to_idx[cls_name]
                    self.samples.append((img_path, target))
                    self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_image_name(self, idx):
        image_path, _ = self.samples[idx]
        return str(Path(image_path).relative_to(self.root_dir))

    def get_image_path(self, idx):
        image_path, _ = self.samples[idx]
        return Path(image_path)

    def resolve_image_path(self, image_name):
        return Path(self.root_dir) / image_name
