import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from glm_saga.elasticnet import IndexedTensorDataset, elastic_loss_and_acc_loader, maximum_reg_loader, train_saga
from torch.utils.data import DataLoader, TensorDataset

from mcbm.scripts.train_cbm import (
    ConceptLayer,
    concept_logits_for_split,
    load_all_activation_splits,
    normalize_concept_features,
)


MAX_SAGA_PATH_STEPS = 100
MIN_LAMBDA_COEFFICIENT = 1e-5
DEFAULT_TARGET_NCCS = "5,10,15,20,25,30"
DEFAULT_NCC_THRESHOLD = 0.95
DEFAULT_NCC_TOLERANCE = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate M-CBM at target NCC levels.")
    parser.add_argument("--cbm-dir", required=True, help="Directory containing a trained CBM run.")
    parser.add_argument("--output-dir", help="Where to save NCC outputs. Defaults to <cbm-dir>/ncc.")
    parser.add_argument("--device", help="Torch device override, e.g. cuda or cuda:0.")
    parser.add_argument("--lam", type=float, help="Maximum glm_saga regularization value.")
    parser.add_argument("--target-nccs", default=DEFAULT_TARGET_NCCS, help="Comma-separated target NCC values.")
    parser.add_argument("--ncc-threshold", type=float, default=DEFAULT_NCC_THRESHOLD, help="Contribution fraction used to compute NCC.")
    parser.add_argument("--ncc-tolerance", type=float, default=DEFAULT_NCC_TOLERANCE, help="Relative matching tolerance for target NCCs.")
    return parser.parse_args()


def load_config_for_cbm(cbm_dir):
    run_info_path = cbm_dir / "run_info.json"
    if not run_info_path.exists():
        raise FileNotFoundError(f"{run_info_path} not found.")

    run_info = json.loads(run_info_path.read_text())
    required_keys = {"config", "device", "parameters"}
    missing_keys = sorted(required_keys - run_info.keys())
    if missing_keys:
        raise KeyError(f"{run_info_path} is missing: {', '.join(missing_keys)}.")

    config = dict(run_info["parameters"])
    config["device"] = run_info["device"]
    return config, Path(run_info["config"]), run_info


def parse_target_nccs(text):
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("--target-nccs must contain at least one value.")
    return values


def load_cbl_state(path):
    return torch.load(path, map_location="cpu")


def build_concept_logits(cbm_dir, config, run_info, device):
    activation_dir = run_info.get("activation_dir")
    if activation_dir is None:
        raise KeyError("run_info.json must contain activation_dir to rebuild concept features.")

    train_split, val_split, test_split = load_all_activation_splits(activation_dir)
    train_features, _, _ = train_split
    input_dim = int(train_features.shape[1])
    cbl_state = load_cbl_state(cbm_dir / "cbl.pt")
    num_concepts = int(cbl_state["model.0.weight"].shape[0])

    cbl = ConceptLayer(
        in_features=input_dim,
        out_features=num_concepts,
        num_hidden=int(config["cbl_hidden_layers"]),
    ).to(device)
    cbl.load_state_dict(cbl_state)

    batch_size = int(config["saga_batch_size"])

    train_concepts, train_labels = concept_logits_for_split(cbl, train_split, batch_size, device)
    val_concepts, val_labels = concept_logits_for_split(cbl, val_split, batch_size, device)
    test_concepts, test_labels = concept_logits_for_split(cbl, test_split, batch_size, device)
    return train_concepts, train_labels, val_concepts, val_labels, test_concepts, test_labels


def build_concept_features(cbm_dir, config, run_info, device):
    train_concepts, train_labels, val_concepts, val_labels, test_concepts, test_labels = build_concept_logits(
        cbm_dir, config, run_info, device
    )
    normalized = normalize_concept_features(train_concepts, val_concepts, test_concepts)[2]
    train_concepts, val_concepts, test_concepts = normalized
    return train_concepts, train_labels, val_concepts, val_labels, test_concepts, test_labels


def make_train_loader(features, labels, batch_size, saga_auto_weight, device):
    sample_weights = None
    if saga_auto_weight:
        unique_classes, class_counts = torch.unique(labels, return_counts=True)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
        sample_weights = class_weights[labels].to(device)
    dataset = IndexedTensorDataset(features, labels, sample_weight=sample_weights)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def make_eval_loader(features, labels, batch_size):
    return DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=False)


def balanced_accuracy(preds, labels, num_classes):
    confusion_ids = labels.cpu() * num_classes + preds.cpu()
    confusion = torch.bincount(confusion_ids, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    true_positive = confusion.diag()
    support = confusion.sum(dim=1)
    recall = true_positive.float() / support.clamp_min(1).float()
    return float(recall[support > 0].mean().item())


def compute_ncc_for_batch(features, weights, ncc_threshold):
    contributions = torch.abs(features.unsqueeze(1) * weights)
    sorted_values, _ = torch.sort(contributions, dim=-1, descending=True)
    sorted_values = sorted_values.float()
    cumulative = sorted_values.cumsum(dim=-1)
    total = cumulative[..., -1:].contiguous()
    threshold = ncc_threshold * total
    prefix = torch.cat([torch.zeros_like(total), cumulative], dim=-1)
    counts = torch.searchsorted(prefix, threshold, right=False).squeeze(-1)
    counts = counts.clamp_max(sorted_values.size(-1))
    counts[total.squeeze(-1) == 0] = 0
    return counts.float().mean(dim=-1)


def evaluate_path_point(final_layer, loader, ncc_threshold, device):
    correct = []
    all_preds = []
    all_labels = []
    ncc_values = []

    final_layer.eval()
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = final_layer(features)
            preds = logits.argmax(dim=-1)

            correct.append(preds == labels)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            ncc_values.append(compute_ncc_for_batch(features, final_layer.weight, ncc_threshold).cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return {
        "accuracy": float(torch.cat(correct).float().mean().item()),
        "balanced_accuracy": balanced_accuracy(preds, labels, final_layer.weight.shape[0]),
        "actual_ncc": float(torch.cat(ncc_values).mean().item()),
    }


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def get_ncc_lam(config, args):
    if args.lam is not None:
        return args.lam
    if "saga_lam" not in config:
        raise KeyError("Missing required config key: saga_lam")
    return config["saga_lam"]


def make_csv_rows(path_results):
    return [
        {key: value for key, value in row.items() if key not in {"weight", "bias"}}
        for row in path_results
    ]


def train_and_evaluate_ncc_path(
    train_loader,
    val_loader,
    test_loader,
    num_concepts,
    num_classes,
    config,
    lam,
    ncc_threshold,
    stop_ncc,
    output_dir,
    device,
):
    linear = torch.nn.Linear(num_concepts, num_classes).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    metadata = {"max_reg": {"nongrouped": float(lam)}}
    alpha = 0.99
    max_lam = maximum_reg_loader(train_loader, group=False, metadata=metadata, family="multinomial") / max(0.001, alpha)
    min_lam = MIN_LAMBDA_COEFFICIENT * max_lam
    lams = torch.logspace(math.log10(float(max_lam)), math.log10(float(min_lam)), MAX_SAGA_PATH_STEPS)
    lrs = torch.full_like(lams, float(config["saga_step_size"]))

    state = None
    final_layer = torch.nn.Linear(num_concepts, num_classes).to(device)
    path_results = []
    n_ex = len(train_loader.dataset)

    for index, (lambda_value, lr) in enumerate(zip(lams, lrs)):
        start = time.time()
        state = train_saga(
            linear,
            train_loader,
            float(lr),
            int(config["saga_n_iters"]),
            lambda_value,
            alpha,
            table_device=None,
            preprocess=None,
            group=False,
            verbose=None,
            state=state,
            n_ex=n_ex,
            n_classes=num_classes,
            tol=1e-4,
            lookbehind=None,
            family="multinomial",
            logger=print,
        )

        with torch.no_grad():
            train_loss, train_accuracy = elastic_loss_and_acc_loader(
                linear, train_loader, lambda_value, alpha, family="multinomial"
            )
            val_loss, _ = elastic_loss_and_acc_loader(
                linear, val_loader, lambda_value, alpha, family="multinomial"
            )
            test_loss, _ = elastic_loss_and_acc_loader(
                linear, test_loader, lambda_value, alpha, family="multinomial"
            )

        final_layer.load_state_dict(
            {
                "weight": linear.weight.detach().clone(),
                "bias": linear.bias.detach().clone(),
            }
        )
        val_metrics = evaluate_path_point(final_layer, val_loader, ncc_threshold, device)
        test_metrics = evaluate_path_point(final_layer, test_loader, ncc_threshold, device)
        nnz = int((linear.weight.detach().abs() > 1e-5).sum().item())
        total = int(linear.weight.numel())
        row = {
            "path_index": index,
            "lam": float(lambda_value),
            "val_actual_ncc": val_metrics["actual_ncc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "test_actual_ncc": test_metrics["actual_ncc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "ncc_threshold": ncc_threshold,
            "train_loss": float(train_loss.item()),
            "train_accuracy": float(train_accuracy.item()),
            "val_loss": float(val_loss.item()),
            "test_loss": float(test_loss.item()),
            "nonzero_weights": nnz,
            "total_weights": total,
            "fraction_nonzero": nnz / total,
            "seconds": time.time() - start,
            "weight": linear.weight.detach().cpu().clone(),
            "bias": linear.bias.detach().cpu().clone(),
        }
        path_results.append(row)
        write_csv(output_dir / "ncc_all_models.csv", make_csv_rows(path_results))
        print(
            f"({index}) lambda {float(lambda_value):.4f}, "
            f"val_acc {val_metrics['accuracy']:.4f}, val_ncc {val_metrics['actual_ncc']:.2f}, "
            f"test_acc {test_metrics['accuracy']:.4f}, "
            f"sparsity {row['fraction_nonzero']:.4f} [{nnz}/{total}], "
            f"time {row['seconds']:.1f}s"
        )
        if val_metrics["actual_ncc"] >= stop_ncc:
            print(f"Stopping NCC path because validation NCC reached {val_metrics['actual_ncc']:.2f} >= {stop_ncc:.2f}.")
            break

    return path_results


def select_target_models(path_results, target_nccs, tolerance):
    selected_rows = []
    selected = {}
    val_nccs = np.array([row["val_actual_ncc"] for row in path_results])
    val_accuracies = np.array([row["val_accuracy"] for row in path_results])

    for target in target_nccs:
        lower = target * (1.0 - tolerance)
        upper = target * (1.0 + tolerance)
        mask = (val_nccs >= lower) & (val_nccs <= upper)

        if np.any(mask):
            candidate_indices = np.where(mask)[0]
            best_accuracy = val_accuracies[candidate_indices].max()
            best_accuracy_indices = candidate_indices[val_accuracies[candidate_indices] == best_accuracy]
            distances = np.abs(val_nccs[best_accuracy_indices] - target)
            best_index = best_accuracy_indices[np.argmin(distances)]
            selection_rule = "within_tolerance_best_accuracy"
        else:
            best_index = int(np.argmin(np.abs(val_nccs - target)))
            selection_rule = "closest_ncc"

        best = path_results[int(best_index)]
        selected[target] = best
        selected_rows.append(
            {
                "Target_NCC": target,
                "Selected_Validation_NCC": best["val_actual_ncc"],
                "Validation_Accuracy": best["val_accuracy"],
                "Validation_Balanced_Accuracy": best["val_balanced_accuracy"],
                "Selected_Test_NCC": best["test_actual_ncc"],
                "Test_Accuracy": best["test_accuracy"],
                "Test_Balanced_Accuracy": best["test_balanced_accuracy"],
                "Lambda": best["lam"],
                "NCC_threshold": best["ncc_threshold"],
                "Selection": selection_rule,
            }
        )
        print(
            f"Target NCC {target:.2f}: selected NCC {best['val_actual_ncc']:.2f}, "
            f"val accuracy {best['val_accuracy']:.4f}, test accuracy {best['test_accuracy']:.4f}"
        )

    return selected, selected_rows


def main():
    args = parse_args()
    cbm_dir = Path(args.cbm_dir)
    config, config_path, run_info = load_config_for_cbm(cbm_dir)
    output_dir = Path(args.output_dir) if args.output_dir else cbm_dir / "ncc"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or config.get("device", "cuda"))
    lam = float(get_ncc_lam(config, args))
    target_nccs = parse_target_nccs(args.target_nccs)
    ncc_threshold = float(args.ncc_threshold)
    ncc_tolerance = float(args.ncc_tolerance)

    print("Building concept features from cbl.pt and saved activations.")
    train_features, train_labels, val_features, val_labels, test_features, test_labels = build_concept_features(
        cbm_dir,
        config,
        run_info,
        device,
    )

    num_concepts = int(train_features.shape[1])
    num_classes = int(max(train_labels.max(), val_labels.max(), test_labels.max()).item()) + 1
    batch_size = int(config["saga_batch_size"])
    train_loader = make_train_loader(
        train_features,
        train_labels,
        batch_size,
        bool(config["saga_auto_weight"]),
        device,
    )
    val_loader = make_eval_loader(val_features, val_labels, batch_size)
    test_loader = make_eval_loader(test_features, test_labels, batch_size)

    print(f"Loaded CBM concept features from {cbm_dir}")
    print(f"Training NCC path with {num_concepts} concepts and {num_classes} classes.")
    print(f"NCC settings: lam={lam}, targets={target_nccs}, threshold={ncc_threshold}, tolerance={ncc_tolerance}")
    stop_ncc = max(target_nccs)
    path_results = train_and_evaluate_ncc_path(
        train_loader,
        val_loader,
        test_loader,
        num_concepts,
        num_classes,
        config,
        lam,
        ncc_threshold,
        stop_ncc,
        output_dir,
        device,
    )

    selected, selected_rows = select_target_models(path_results, target_nccs, ncc_tolerance)
    write_csv(output_dir / "ncc_selected_models.csv", selected_rows)
    for target, row in selected.items():
        torch.save(row["weight"], output_dir / f"W_g@NCC_target={int(target):d}.pt")
        torch.save(row["bias"], output_dir / f"b_g@NCC_target={int(target):d}.pt")

    summary = {
        "dataset": config["dataset"],
        "config": str(config_path),
        "cbm_dir": str(cbm_dir),
        "num_concepts": num_concepts,
        "num_classes": num_classes,
        "lam": lam,
        "ncc_threshold": ncc_threshold,
        "ncc_tolerance": ncc_tolerance,
        "target_nccs": target_nccs,
        "average_selected_val_accuracy": float(np.nanmean([row["Validation_Accuracy"] for row in selected_rows])),
        "average_selected_val_balanced_accuracy": float(np.nanmean([row["Validation_Balanced_Accuracy"] for row in selected_rows])),
        "average_selected_test_accuracy": float(np.nanmean([row["Test_Accuracy"] for row in selected_rows])),
        "average_selected_test_balanced_accuracy": float(np.nanmean([row["Test_Balanced_Accuracy"] for row in selected_rows])),
    }
    (output_dir / "ncc_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved NCC evaluation to {output_dir}")
    print(f"Average selected validation accuracy: {summary['average_selected_val_accuracy']:.4f}")
    print(f"Average selected test accuracy: {summary['average_selected_test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
