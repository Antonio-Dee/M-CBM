import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


from mcbm.data.utils import get_activation_splits, get_target_model


def parse_args():
    parser = argparse.ArgumentParser(description="Extract target-model activations.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--output-dir", help="Directory for activation files.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for extraction.")
    parser.add_argument("--device", help="Torch device override, e.g. cuda, cuda:0, or cpu.")
    return parser.parse_args()


def clean_name(value):
    return str(value).replace(".", "_").replace("/", "_")


def get_output_dir(args, dataset_name, backbone, feature_layer):
    if args.output_dir is not None:
        return Path(args.output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"activations_{clean_name(dataset_name)}_{clean_name(backbone)}_{clean_name(feature_layer)}_{timestamp}"
    return Path("activations") / clean_name(dataset_name) / run_name


def get_nested_module(model, layer_name):
    module = model
    for part in layer_name.split("."):
        module = getattr(module, part)
    return module


def get_activation(outputs):
    def hook(model, input, output):
        if len(output.shape) == 4:
            outputs.append(output.mean(dim=[2, 3]).detach().cpu())
        elif len(output.shape) == 2:
            outputs.append(output.detach().cpu())
        else:
            raise ValueError(f"Unsupported activation shape: {tuple(output.shape)}")
    return hook


def get_image_name(dataset, index):
    if isinstance(dataset, Subset):
        return get_image_name(dataset.dataset, dataset.indices[index])

    return dataset.get_image_name(index)


def get_activation_rows(dataset, features, labels):
    rows = []
    for index, label in enumerate(labels):
        rows.append(
            {
                "image_name": get_image_name(dataset, index),
                "label": int(label.item()),
                "features": features[index],
            }
        )
    return rows


def extract_split(model, dataset, feature_layer, batch_size, device, num_workers, split_name):
    if len(dataset) == 0:
        raise ValueError("Empty dataset split.")

    features = []
    labels = []
    hook = get_nested_module(model, feature_layer).register_forward_hook(get_activation(features))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    try:
        with torch.no_grad():
            for images, batch_labels in tqdm(loader, desc=f"Extracting {split_name}", unit="batch"):
                model(images.to(device))
                if torch.is_tensor(batch_labels):
                    labels.append(batch_labels.detach().cpu())
                else:
                    labels.append(torch.as_tensor(batch_labels))
    finally:
        hook.remove()

    if not features or not labels:
        raise ValueError("No activations were extracted.")

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text())

    dataset_name = config["dataset"]
    backbone = config["backbone"]
    feature_layer = config["feature_layer"]
    val_split = config["val_split"]
    seed = config["seed"]
    num_workers = int(config["num_workers"])

    device = torch.device(args.device or config.get("device", "cuda"))
    output_dir = get_output_dir(args, dataset_name, backbone, feature_layer)
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Saving activations to {output_dir}")

    model, preprocess = get_target_model(backbone, device)
    model.eval()

    splits = get_activation_splits(dataset_name, preprocess, val_split, seed)
    for split_name, dataset in splits.items():
        print(f"Extracting {split_name} ({len(dataset)} images)")
        features, labels = extract_split(model, dataset, feature_layer, args.batch_size, device, num_workers, split_name)
        rows = get_activation_rows(dataset, features, labels)
        print(f"Saving {split_name} activations")
        torch.save(rows, output_dir / f"activations_{split_name}.pt")
        print(f"Saved {split_name}: {len(rows)} activations with feature shape {tuple(features.shape[1:])}")
        del rows, features, labels

    run_info = {
        "config": args.config,
        "dataset": dataset_name,
        "backbone": backbone,
        "feature_layer": feature_layer,
        "val_split": val_split,
        "seed": seed,
        "batch_size": args.batch_size,
        "device": str(device),
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()
