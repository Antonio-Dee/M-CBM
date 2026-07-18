import os
from functools import cache
from pathlib import Path

import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import models, transforms

from mcbm.data.data_cub import CUBDataset
from mcbm.data.data_imagenet import ImageNetDataset
from mcbm.data.data_isic import ISIC2018Dataset


DATASET_FOLDER = os.environ.get("DATASET_FOLDER", "datasets")
CHECKPOINT_FOLDER = Path(__file__).resolve().parents[2] / "checkpoints"

DATASET_ROOTS = {
    "imagenet": f"{DATASET_FOLDER}/imagenet/images_complete/ilsvrc/",
    "cub": f"{DATASET_FOLDER}/CUB_200_2011",
    "isic2018_train": f"{DATASET_FOLDER}/ISIC2018/ISIC2018_Task3_Training_Input/",
    "isic2018_val": f"{DATASET_FOLDER}/ISIC2018/ISIC2018_Task3_Validation_Input/",
    "isic2018_test": f"{DATASET_FOLDER}/ISIC2018/ISIC2018_Task3_Test_Input/",
}

ISIC_LABEL_FILES = {
    "isic2018_train": f"{DATASET_FOLDER}/ISIC2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv",
    "isic2018_val": f"{DATASET_FOLDER}/ISIC2018/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv",
    "isic2018_test": f"{DATASET_FOLDER}/ISIC2018/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv",
}

LABEL_FILES = {
    "imagenet": "mcbm/data/class_names/imagenet.txt",
    "cub": "mcbm/data/class_names/cub.txt",
    "isic2018": "mcbm/data/class_names/isic2018.txt",
}

DATASET_DOMAINS = {
    "cub": "bird species classification",
    "isic2018": "dermoscopic images of pigmented skin lesions",
    "imagenet": "imagenet classification",
}


def get_dataset_domain(dataset_name):
    if dataset_name not in DATASET_DOMAINS:
        raise ValueError(
            f"No domain is defined for dataset '{dataset_name}'. "
            "Add it to DATASET_DOMAINS in mcbm/data/utils.py."
        )
    return DATASET_DOMAINS[dataset_name]


def get_resnet_imagenet_preprocess():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_resnet_isic_preprocess():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class ResizeCenterCrop:
    def __init__(self, resize_size, crop_size):
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(crop_size),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


def get_llm_image_transform(target_name):
    if target_name in {"resnet18_cub", "resnet50"}:
        return ResizeCenterCrop(resize_size=256, crop_size=224)
    if target_name == "resnet50_isic2018":
        return lambda image: image
    raise ValueError(f"No LLM image transform defined for target model: {target_name}")


@cache
def get_image_dataset(dataset_name, split):
    return get_data(f"{dataset_name}_{split}", preprocess=None)


def get_image_path(dataset_name, split, image_name):
    return get_image_dataset(dataset_name, split).resolve_image_path(image_name)


def get_data(dataset_name, preprocess=None):
    if dataset_name.startswith("isic2018"):
        return ISIC2018Dataset(
            DATASET_ROOTS[dataset_name],
            ISIC_LABEL_FILES[dataset_name],
            transform=preprocess,
        )

    if dataset_name.startswith("cub"):
        if "train" in dataset_name:
            is_train = True
        elif "test" in dataset_name:
            is_train = False
        else:
            raise ValueError(f"Unknown split in CUB dataset name: {dataset_name}")
        return CUBDataset(DATASET_ROOTS["cub"], transform=preprocess, train=is_train)

    if dataset_name.startswith("imagenet"):
        if "train" in dataset_name:
            split = "train"
        elif "val" in dataset_name or "test" in dataset_name:
            split = "val"
        else:
            raise ValueError(f"Unknown split in ImageNet dataset name: {dataset_name}")
        return ImageNetDataset(DATASET_ROOTS["imagenet"], split=split, transform=preprocess)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def split_train_dataset(train_dataset, val_split, seed):
    train_size = len(train_dataset)
    if isinstance(val_split, float) and 0.0 < val_split < 1.0:
        val_size = int(round(train_size * val_split))
    else:
        val_size = int(val_split)

    if val_size <= 0 or val_size >= train_size:
        raise ValueError(f"val_split must produce between 1 and {train_size - 1} validation samples.")

    if not hasattr(train_dataset, "targets"):
        raise AttributeError("Dataset must expose targets for a class-stratified validation split.")
    if len(train_dataset.targets) != train_size:
        raise ValueError("Dataset targets must have the same length as the dataset.")

    train_indices, val_indices = train_test_split(
        range(train_size),
        test_size=val_size,
        random_state=int(seed),
        stratify=train_dataset.targets,
    )
    return Subset(train_dataset, train_indices), Subset(train_dataset, val_indices)


def get_official_data_splits(dataset_name, preprocess=None):
    if dataset_name == "imagenet":
        return {
            "train": get_data("imagenet_train", preprocess),
            "val": None,
            "test": get_data("imagenet_val", preprocess),
        }

    if dataset_name == "cub":
        return {
            "train": get_data("cub_train", preprocess),
            "val": None,
            "test": get_data("cub_test", preprocess),
        }

    if dataset_name == "isic2018":
        return {
            "train": get_data("isic2018_train", preprocess),
            "val": get_data("isic2018_val", preprocess),
            "test": get_data("isic2018_test", preprocess),
        }

    raise ValueError(f"No official data splits defined for dataset: {dataset_name}")


def get_activation_splits(dataset_name, preprocess, val_split, seed):
    splits = get_official_data_splits(dataset_name, preprocess)

    if val_split is None:
        if splits["val"] is None:
            raise ValueError(f"{dataset_name} has no official validation split. Set val_split in the config.")
        return splits

    train_dataset, val_dataset = split_train_dataset(splits["train"], val_split, seed)
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": splits["test"],
    }


def get_target_model(target_name, device):
    if target_name == "resnet18_cub":
        target_model = ptcv_get_model("resnet18_cub", pretrained=True).to(device)
        target_model.eval()
        return target_model, get_resnet_imagenet_preprocess()

    if target_name == "resnet50_isic2018":
        checkpoint_path = CHECKPOINT_FOLDER / "resnet50_ISIC2018.pth"
        target_model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        target_model.eval()
        return target_model, get_resnet_isic_preprocess()

    if target_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        target_model = models.resnet50(weights=weights).to(device)
        target_model.eval()
        return target_model, weights.transforms()

    raise ValueError(f"Unsupported target model: {target_name}")


def get_classes(dataset_name):
    with open(LABEL_FILES[dataset_name], "r") as f:
        return f.read().split("\n")
