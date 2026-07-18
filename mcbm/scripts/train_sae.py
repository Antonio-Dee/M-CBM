import argparse
import copy
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sparse_autoencoder import (
    L2ReconstructionLoss,
    LearnedActivationsL1Loss,
    LossReducer,
    LossReductionType,
    SparseAutoencoder,
    SparseAutoencoderConfig,
)
from torch.optim import Adam

from mcbm.data.utils import get_target_model


MAX_POST_SELECTION_CURVE_POINTS = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Train an SAE on saved backbone activations.")
    parser.add_argument("--config", required=True, help="Path to dataset config JSON.")
    parser.add_argument("--activation-dir", required=True, help="Directory containing activation files.")
    parser.add_argument("--output-dir", help="Directory for SAE outputs.")
    parser.add_argument("--device", help="Torch device override, e.g. cuda, cuda:0, or cpu.")
    return parser.parse_args()


def require_sae_config(config):
    key_map = {
        "expansion_factor": "sae_expansion_factor",
        "l1_coefficient": "sae_l1_coefficient",
        "learning_rate": "sae_learning_rate",
        "batch_size": "sae_batch_size",
        "epochs": "sae_epochs",
        "patience": "sae_patience",
        "ce_drop_tolerance": "sae_ce_drop_tolerance",
        "min_active_images": "sae_min_active_images",
    }
    missing = [config_key for config_key in key_map.values() if config_key not in config]
    if missing:
        raise KeyError(f"Config is missing required SAE keys: {missing}")

    return {
        local_key: config[config_key]
        for local_key, config_key in key_map.items()
    }


def load_activation_rows(path):
    rows = torch.load(path, map_location="cpu")
    if not isinstance(rows, list):
        raise TypeError(f"{path} must contain a list of activation rows.")
    if len(rows) == 0:
        raise ValueError(f"{path} contains no activation rows.")

    features = torch.stack([row["features"].detach().float().cpu() for row in rows])
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    return features, labels


def load_activations(activation_dir):
    activation_dir = Path(activation_dir)
    train_features, train_labels = load_activation_rows(activation_dir / "activations_train.pt")
    val_features, val_labels = load_activation_rows(activation_dir / "activations_val.pt")
    return train_features, train_labels, val_features, val_labels


def make_autoencoder(input_dim, latent_dim, device):
    autoencoder_config = SparseAutoencoderConfig(
        n_input_features=input_dim,
        n_learned_features=latent_dim,
    )
    autoencoder = SparseAutoencoder(autoencoder_config)
    return autoencoder.to(device)


def make_optimizer(autoencoder, learning_rate):
    return Adam(autoencoder.parameters(), lr=float(learning_rate))


def make_loss(l1_coefficient):
    return LossReducer(
        LearnedActivationsL1Loss(l1_coefficient=float(l1_coefficient)),
        L2ReconstructionLoss(),
    )


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward_sae(autoencoder, features):
    learned, reconstructed = autoencoder(features)
    return learned, reconstructed


def compute_losses(loss_fn, features, learned, reconstructed):
    total_loss, loss_metrics = loss_fn.scalar_loss_with_log(
        features,
        learned,
        reconstructed,
        component_reduction=LossReductionType.MEAN,
    )
    metrics = {
        metric.postfix: metric.component_wise_values.mean()
        for metric in loss_metrics
    }
    avg_l0 = (learned > 0).float().sum(dim=-1).mean()
    return {
        "reconstruction_loss": metrics["l2_reconstruction_loss"],
        "l1_penalty": metrics["learned_activations_l1_loss_penalty"],
        "total_loss": total_loss,
        "avg_l0": avg_l0,
    }


def tensor_batches(features, batch_size, indices=None):
    for start in range(0, len(features), batch_size):
        if indices is None:
            yield features[start:start + batch_size]
        else:
            yield features[indices[start:start + batch_size]]


def run_epoch(autoencoder, features, batch_size, optimizer, loss_fn, train, shuffle, generator=None):
    if train:
        autoencoder.train()
    else:
        autoencoder.eval()

    indices = None
    if shuffle:
        if generator is None:
            raise ValueError("Shuffled SAE epochs require a random generator.")
        # Match the random-number consumption and ordering of PyTorch's DataLoader.
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        indices = torch.randperm(len(features), generator=generator).to(features.device)

    totals = {
        "reconstruction_loss": 0.0,
        "l1_penalty": 0.0,
        "total_loss": 0.0,
        "avg_l0": 0.0,
    }
    n_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tensor_batches(features, batch_size, indices):
            if train:
                optimizer.zero_grad()

            learned, reconstructed = forward_sae(autoencoder, batch)
            losses = compute_losses(loss_fn, batch, learned, reconstructed)

            if train:
                losses["total_loss"].backward()
                optimizer.step()
                autoencoder.post_backwards_hook()

            n_batch_samples = batch.shape[0]
            n_samples += n_batch_samples
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * n_batch_samples

    return {key: value / n_samples for key, value in totals.items()}


def compute_unit_active_counts(autoencoder, features, batch_size):
    autoencoder.eval()
    counts = torch.zeros(autoencoder.config.n_learned_features, dtype=torch.long)
    total_l0 = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in tensor_batches(features, batch_size):
            learned, _ = forward_sae(autoencoder, batch)
            active = learned > 0
            counts += active.sum(dim=0).cpu()
            total_l0 += float(active.float().sum(dim=-1).mean().cpu()) * batch.shape[0]
            n_samples += batch.shape[0]

    return counts, total_l0 / n_samples


def get_classifier_head(config, device):
    target_model, _ = get_target_model(config["backbone"], device)
    target_model.eval()

    if hasattr(target_model, "fc"):
        return target_model.fc.to(device).eval()
    if hasattr(target_model, "output"):
        return target_model.output.to(device).eval()

    raise ValueError(
        f"Cannot find classifier head for backbone {config['backbone']}. "
        "Expected model.fc or model.output."
    )


def reconstruct_with_kept_units(autoencoder, features, kept_mask):
    learned, _ = forward_sae(autoencoder, features)
    masked = learned * kept_mask.to(features.device).float()
    decoded = autoencoder.decoder(masked)
    reconstructed = autoencoder.post_decoder_bias(decoded)
    return reconstructed


def compute_recovered_metrics(autoencoder, classifier_head, features, labels, kept_mask, batch_size):
    autoencoder.eval()
    classifier_head.eval()
    total_original_loss = 0.0
    total_reconstructed_loss = 0.0
    total_zero_loss = 0.0
    total_original_correct = 0.0
    total_reconstructed_correct = 0.0
    total_zero_correct = 0.0
    n_samples = 0

    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch_features = features[start:start + batch_size]
            batch_labels = labels[start:start + batch_size]

            original_logits = classifier_head(batch_features)
            original_loss = F.cross_entropy(original_logits, batch_labels)
            original_correct = (original_logits.argmax(dim=1) == batch_labels).float().sum()

            reconstructed = reconstruct_with_kept_units(autoencoder, batch_features, kept_mask)
            reconstructed_logits = classifier_head(reconstructed)
            reconstructed_loss = F.cross_entropy(reconstructed_logits, batch_labels)
            reconstructed_correct = (reconstructed_logits.argmax(dim=1) == batch_labels).float().sum()

            zero_logits = classifier_head(torch.zeros_like(batch_features))
            zero_loss = F.cross_entropy(zero_logits, batch_labels)
            zero_correct = (zero_logits.argmax(dim=1) == batch_labels).float().sum()

            total_original_loss += float(original_loss.cpu()) * batch_features.shape[0]
            total_reconstructed_loss += float(reconstructed_loss.cpu()) * batch_features.shape[0]
            total_zero_loss += float(zero_loss.cpu()) * batch_features.shape[0]
            total_original_correct += float(original_correct.cpu())
            total_reconstructed_correct += float(reconstructed_correct.cpu())
            total_zero_correct += float(zero_correct.cpu())
            n_samples += batch_features.shape[0]

    original_loss = total_original_loss / n_samples
    reconstructed_loss = total_reconstructed_loss / n_samples
    zero_loss = total_zero_loss / n_samples
    original_accuracy = total_original_correct / n_samples
    reconstructed_accuracy = total_reconstructed_correct / n_samples
    zero_accuracy = total_zero_correct / n_samples
    recovered_loss = (zero_loss - reconstructed_loss) / (zero_loss - original_loss + 1e-8)
    recovered_accuracy = (reconstructed_accuracy - zero_accuracy) / (original_accuracy - zero_accuracy + 1e-8)
    return recovered_loss, recovered_accuracy


def get_pruning_thresholds(train_counts, min_active_images):
    min_threshold = int(min_active_images) - 1
    if min_threshold < 0:
        raise ValueError("sae_min_active_images must be at least 1.")

    max_count = int(train_counts.max().item())
    thresholds = {min_threshold}
    thresholds.update(
        int(count)
        for count in train_counts.unique().tolist()
        if min_threshold <= int(count) < max_count
    )
    return sorted(thresholds)


def run_pruning(
    autoencoder,
    classifier_head,
    train_counts,
    val_features,
    val_labels,
    tolerance,
    min_active_images,
    batch_size,
):
    total_units = int(train_counts.numel())
    all_units_mask = torch.ones(total_units, dtype=torch.bool)
    baseline_loss, baseline_accuracy = compute_recovered_metrics(
        autoencoder, classifier_head, val_features, val_labels, all_units_mask, batch_size
    )

    curve = []
    selected = None
    first_failed_threshold = None
    thresholds = get_pruning_thresholds(train_counts, min_active_images)
    post_failure_indices = None
    for threshold_index, threshold in enumerate(thresholds):
        if post_failure_indices is not None and threshold_index not in post_failure_indices:
            continue

        kept_mask = train_counts > int(threshold)
        kept_units = int(kept_mask.sum().item())
        if kept_units == 0:
            continue

        recovered_loss, recovered_accuracy = compute_recovered_metrics(
            autoencoder, classifier_head, val_features, val_labels, kept_mask, batch_size
        )
        loss_drop = baseline_loss - recovered_loss
        if first_failed_threshold is None and loss_drop <= float(tolerance):
            selected = {
                "threshold": int(threshold),
                "kept_mask": kept_mask,
                "recovered_loss": recovered_loss,
                "recovered_accuracy": recovered_accuracy,
                "loss_drop": loss_drop,
            }
        elif first_failed_threshold is None:
            first_failed_threshold = int(threshold)
            remaining_indices = list(range(threshold_index + 1, len(thresholds)))
            if len(remaining_indices) > MAX_POST_SELECTION_CURVE_POINTS:
                sampled_positions = np.linspace(
                    0,
                    len(remaining_indices) - 1,
                    MAX_POST_SELECTION_CURVE_POINTS,
                    dtype=int,
                )
                remaining_indices = [remaining_indices[position] for position in sampled_positions]
            post_failure_indices = set(remaining_indices)

        curve.append(
            {
                "threshold": int(threshold),
                "kept_units": kept_units,
                "pruned_units": total_units - kept_units,
                "recovered_loss": recovered_loss,
                "recovered_loss_drop": loss_drop,
                "recovered_accuracy": recovered_accuracy,
            }
        )

    if selected is None:
        threshold = int(min_active_images) - 1
        kept_mask = train_counts > threshold
        recovered_loss, recovered_accuracy = compute_recovered_metrics(
            autoencoder, classifier_head, val_features, val_labels, kept_mask, batch_size
        )
        selected = {
            "threshold": threshold,
            "kept_mask": kept_mask,
            "recovered_loss": recovered_loss,
            "recovered_accuracy": recovered_accuracy,
            "loss_drop": baseline_loss - recovered_loss,
        }

    kept_indices = torch.nonzero(selected["kept_mask"], as_tuple=False).squeeze(1).cpu()
    pruned_units = int((~selected["kept_mask"]).sum().item())
    summary = {
        "total_units": total_units,
        "kept_units": int(kept_indices.numel()),
        "pruned_units": pruned_units,
        "selected_activation_count_threshold": int(selected["threshold"]),
        "min_active_images": int(min_active_images),
        "ce_drop_tolerance": float(tolerance),
        "selected_recovered_loss": float(selected["recovered_loss"]),
        "recovered_loss_drop": float(selected["loss_drop"]),
        "selected_recovered_accuracy": float(selected["recovered_accuracy"]),
        "recovered_accuracy_drop": float(baseline_accuracy - selected["recovered_accuracy"]),
        "fraction_units_kept": float(kept_indices.numel() / total_units),
    }
    recovered_metrics = {
        "val_recovered_loss": float(baseline_loss),
        "val_recovered_accuracy": float(baseline_accuracy),
    }
    return kept_indices, curve, summary, recovered_metrics


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2))


def sae_config_snapshot(config):
    keys = [
        "dataset",
        "backbone",
        "feature_layer",
        "val_split",
        "seed",
        "sae_expansion_factor",
        "sae_l1_coefficient",
        "sae_learning_rate",
        "sae_batch_size",
        "sae_epochs",
        "sae_patience",
        "sae_ce_drop_tolerance",
        "sae_min_active_images",
    ]
    return {key: config[key] for key in keys if key in config}


def save_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_run_name(config):
    dataset = config["dataset"]
    backbone = config["backbone"]
    feature_layer = config["feature_layer"].replace(".", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"sae_{dataset}_{backbone}_{feature_layer}_{timestamp}"


def plot_training_curves(path, history):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, [row["train_reconstruction_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_reconstruction_loss"] for row in history], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Reconstruction Loss")
    axes[0].set_title("SAE Reconstruction Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, [row["train_avg_l0"] for row in history], label="Train")
    axes[1].plot(epochs, [row["val_avg_l0"] for row in history], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Average L0")
    axes[1].set_title("SAE Average L0")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_feature_density(path, unit_counts, n_train_samples, selected_threshold):
    counts = unit_counts.cpu().numpy()
    density = counts / float(n_train_samples)
    log_density = np.log10(np.clip(density, 1e-8, None))
    ultra_low_log_threshold = -7.5

    mask_dead = log_density < ultra_low_log_threshold
    mask_low_usage = np.logical_and(
        counts <= int(selected_threshold),
        log_density >= ultra_low_log_threshold,
    )
    mask_high_usage = counts > int(selected_threshold)

    data_to_plot = [
        log_density[mask_dead],
        log_density[mask_low_usage],
        log_density[mask_high_usage],
    ]
    colors = ["purple", "orange", "green"]
    labels = [
        "Dead Neurons",
        f"Pruned/Low Density",
        "High Density",
    ]

    plt.figure(figsize=(10, 6))
    plt.hist(
        data_to_plot,
        bins=100,
        stacked=True,
        color=colors,
        edgecolor="black",
        label=labels,
    )
    plt.xlabel("log10(Feature density)")
    plt.ylabel("Counts")
    plt.title("Feature Density (log scale)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_pruning_curve(path, curve, selected_threshold):
    valid = [row for row in curve if row["recovered_loss"] is not None]
    thresholds = [row["threshold"] for row in valid]
    recovered_losses = [row["recovered_loss"] for row in valid]
    recovered_accuracies = [row["recovered_accuracy"] for row in valid]
    kept_units = [row["kept_units"] for row in valid]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(thresholds, recovered_losses, zorder=2)
    axes[0].set_xlabel("Activation-count pruning threshold")
    axes[0].set_ylabel("Recovered Loss")
    axes[0].set_title("Recovered CE Loss vs Pruning Threshold")
    axes[0].grid(True)

    axes[1].plot(thresholds, recovered_accuracies, zorder=2)
    axes[1].set_xlabel("Activation-count pruning threshold")
    axes[1].set_ylabel("Recovered Accuracy")
    axes[1].set_title("Recovered Accuracy vs Pruning Threshold")
    axes[1].grid(True)

    axes[2].plot(thresholds, kept_units, zorder=2)
    axes[2].set_xlabel("Activation-count pruning threshold")
    axes[2].set_ylabel("Number of SAE units kept")
    axes[2].set_title("SAE Units Kept vs Pruning Threshold")
    axes[2].grid(True)

    if selected_threshold in thresholds:
        idx = thresholds.index(selected_threshold)
        for axis, values in zip(axes, [recovered_losses, recovered_accuracies, kept_units]):
            axis.scatter(
                [selected_threshold],
                [values[idx]],
                color="red",
                edgecolor="black",
                linewidth=0.9,
                marker="*",
                s=120,
                zorder=5,
            )

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train_sae(config, config_path, activation_dir, output_dir, device):
    sae_config = require_sae_config(config)
    set_seed(config["seed"])
    if output_dir is None:
        output_dir = Path("outputs") / "sae" / config["dataset"] / make_run_name(config)
    else:
        output_dir = Path(output_dir)
    pruning_dir = output_dir / "pruning"
    plots_dir = output_dir / "plots"
    pruning_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_features, _train_labels, val_features, val_labels = load_activations(activation_dir)
    train_features = train_features.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    input_dim = int(train_features.shape[1])
    latent_dim = int(input_dim * float(sae_config["expansion_factor"]))
    if latent_dim <= 0:
        raise ValueError("sae_expansion_factor produces zero learned features.")

    train_generator = torch.Generator().manual_seed(int(config["seed"]))
    autoencoder = make_autoencoder(input_dim, latent_dim, device)
    optimizer = make_optimizer(autoencoder, sae_config["learning_rate"])
    loss_fn = make_loss(sae_config["l1_coefficient"])
    batch_size = int(sae_config["batch_size"])

    history = []
    best_state_dict = None
    best_epoch = 0
    best_val_total_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, int(sae_config["epochs"]) + 1):
        train_metrics = run_epoch(
            autoencoder,
            train_features,
            batch_size,
            optimizer,
            loss_fn,
            train=True,
            shuffle=True,
            generator=train_generator,
        )
        val_metrics = run_epoch(
            autoencoder,
            val_features,
            batch_size,
            optimizer,
            loss_fn,
            train=False,
            shuffle=False,
        )

        row = {
            "epoch": epoch,
            "train_reconstruction_loss": train_metrics["reconstruction_loss"],
            "train_l1_penalty": train_metrics["l1_penalty"],
            "train_total_loss": train_metrics["total_loss"],
            "train_avg_l0": train_metrics["avg_l0"],
            "val_reconstruction_loss": val_metrics["reconstruction_loss"],
            "val_l1_penalty": val_metrics["l1_penalty"],
            "val_total_loss": val_metrics["total_loss"],
            "val_avg_l0": val_metrics["avg_l0"],
        }
        history.append(row)
        print(
            f"epoch {epoch}: train recon {row['train_reconstruction_loss']:.6f}, "
            f"train L0 {row['train_avg_l0']:.2f}, "
            f"val recon {row['val_reconstruction_loss']:.6f}, "
            f"val L0 {row['val_avg_l0']:.2f}"
        )

        if val_metrics["total_loss"] < best_val_total_loss:
            best_val_total_loss = val_metrics["total_loss"]
            best_epoch = epoch
            best_state_dict = copy.deepcopy(autoencoder.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(sae_config["patience"]):
            print(f"Stopping early after {epoch} epochs.")
            break

    if best_state_dict is None:
        raise RuntimeError("Training did not produce a best model.")

    autoencoder.load_state_dict(best_state_dict)
    best_train_metrics = run_epoch(
        autoencoder,
        train_features,
        batch_size,
        optimizer,
        loss_fn,
        train=False,
        shuffle=True,
        generator=train_generator,
    )
    best_val_metrics = run_epoch(
        autoencoder,
        val_features,
        batch_size,
        optimizer,
        loss_fn,
        train=False,
        shuffle=False,
    )
    checkpoint_name = f"{output_dir.name}.pt"
    checkpoint_path = output_dir / checkpoint_name
    torch.save(
        {
            "state_dict": best_state_dict,
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "sae_config": sae_config,
            "best_epoch": best_epoch,
            "checkpoint_name": checkpoint_name,
        },
        checkpoint_path,
    )

    unit_counts, train_avg_l0 = compute_unit_active_counts(
        autoencoder,
        train_features,
        batch_size,
    )
    _, val_avg_l0 = compute_unit_active_counts(
        autoencoder,
        val_features,
        batch_size,
    )
    classifier_head = get_classifier_head(config, device)
    kept_indices, pruning_curve, pruning_summary, recovered_metrics = run_pruning(
        autoencoder,
        classifier_head,
        unit_counts,
        val_features,
        val_labels,
        sae_config["ce_drop_tolerance"],
        sae_config["min_active_images"],
        batch_size,
    )

    torch.save(kept_indices, pruning_dir / "kept_indices.pt")
    torch.save(unit_counts, pruning_dir / "unit_active_counts.pt")
    save_csv(
        pruning_dir / "pruning_curve.csv",
        pruning_curve,
        ["threshold", "kept_units", "pruned_units", "recovered_loss", "recovered_loss_drop", "recovered_accuracy"],
    )
    save_json(pruning_dir / "pruning_summary.json", pruning_summary)

    save_json(output_dir / "history.json", history)
    plot_training_curves(plots_dir / "training_curves.png", history)
    plot_feature_density(
        plots_dir / "feature_density_histogram.png",
        unit_counts,
        len(train_features),
        pruning_summary["selected_activation_count_threshold"],
    )
    plot_pruning_curve(
        plots_dir / "pruning_curve.png",
        pruning_curve,
        pruning_summary["selected_activation_count_threshold"],
    )

    sae_summary = {
        "dataset": config["dataset"],
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "expansion_factor": sae_config["expansion_factor"],
        "l1_coefficient": sae_config["l1_coefficient"],
        "learning_rate": sae_config["learning_rate"],
        "batch_size": sae_config["batch_size"],
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_val_total_loss": best_val_total_loss,
        "final_train_reconstruction_loss": best_train_metrics["reconstruction_loss"],
        "final_val_reconstruction_loss": best_val_metrics["reconstruction_loss"],
        "final_train_avg_l0": best_train_metrics["avg_l0"],
        "final_val_avg_l0": best_val_metrics["avg_l0"],
        "number_of_sae_units": latent_dim,
    }
    metrics = {
        **sae_summary,
        **recovered_metrics,
        "pruning": pruning_summary,
    }
    save_json(output_dir / "metrics.json", metrics)
    save_json(
        output_dir / "run_info.json",
        {
            "config": str(config_path),
            "activation_dir": str(activation_dir),
            "device": str(device),
            "parameters": sae_config_snapshot(config),
        },
    )

    print(f"Saved SAE to {checkpoint_path}")
    print(
        f"Best epoch {best_epoch}: "
        f"val total loss {best_val_total_loss:.6f}"
    )
    print(
        "Pruning kept "
        f"{pruning_summary['kept_units']}/{pruning_summary['total_units']} units "
        f"at activation-count threshold {pruning_summary['selected_activation_count_threshold']}."
    )


def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    device = torch.device(args.device or config.get("device", "cuda"))
    train_sae(config, args.config, Path(args.activation_dir), args.output_dir, device)


if __name__ == "__main__":
    main()
