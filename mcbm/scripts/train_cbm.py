import argparse
import heapq
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from mcbm.data.utils import get_classes
from mcbm.scripts.name_concepts import load_activation_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Train the M-CBM from saved activations and annotations.")
    parser.add_argument("--config", required=True, help="Path to dataset config JSON.")
    parser.add_argument("--activation-dir", required=True, help="Directory containing activations_train.pt, activations_val.pt, and activations_test.pt.")
    parser.add_argument("--annotation-dir", required=True, help="Annotation run directory containing annotations_train.pt and annotations_test.pt.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to outputs/cbm/<dataset>/cbm_<dataset>_<timestamp>.")
    parser.add_argument("--device", help="Torch device override, e.g. cuda or cuda:0.")
    return parser.parse_args()


class ConceptLayer(nn.Module):
    def __init__(self, in_features, out_features, num_hidden=0, bias=True):
        super().__init__()
        layers = [nn.Linear(in_features, out_features, bias=bias)]
        for _ in range(num_hidden):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(out_features, out_features, bias=bias))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def copy_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in unwrap_model(model).state_dict().items()
    }


def read_concepts_from_annotations(path):
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a dictionary keyed by concept id.")
    rows = []
    for raw_concept_id, detail in data.items():
        concept_id = int(raw_concept_id)
        name = detail.get("concept")
        if name is None or not str(name).strip():
            raise ValueError(f"Concept {concept_id} in {path} has no concept name.")
        rows.append({"concept_id": concept_id, "concept_name": str(name).strip()})
    rows.sort(key=lambda row: row["concept_id"])
    if [row["concept_id"] for row in rows] != list(range(len(rows))):
        raise ValueError(f"{path} must contain consecutive concept ids starting from 0.")
    return rows


def validate_annotation_pair(train_path, test_path, concepts):
    test_concepts = read_concepts_from_annotations(test_path)
    if test_concepts != concepts:
        train_names = {row["concept_id"]: row["concept_name"] for row in concepts}
        test_names = {row["concept_id"]: row["concept_name"] for row in test_concepts}
        mismatches = []
        for concept_id in sorted(set(train_names) | set(test_names)):
            if train_names.get(concept_id) != test_names.get(concept_id):
                mismatches.append((concept_id, train_names.get(concept_id), test_names.get(concept_id)))
        examples = "\n".join(
            f"  id {concept_id}: train='{train_name}' test='{test_name}'"
            for concept_id, train_name, test_name in mismatches[:5]
        )
        raise ValueError(
            "annotations_train.pt and annotations_test.pt do not contain the same concept ids/names.\n"
            f"Examples:\n{examples}"
        )


def load_activation_split(activation_dir, split):
    return load_activation_rows(Path(activation_dir) / f"activations_{split}.pt")


def load_all_activation_splits(activation_dir):
    train = load_activation_split(activation_dir, "train")
    val = load_activation_split(activation_dir, "val")
    test = load_activation_split(activation_dir, "test")
    return train, val, test


def concatenate_train_side(train, val):
    train_features, train_names, train_labels = train
    val_features, val_names, val_labels = val
    features = torch.cat([train_features, val_features], dim=0)
    names = train_names + val_names
    labels = torch.cat([train_labels, val_labels], dim=0)
    return features, names, labels


def read_annotation_labels(path):
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a dictionary keyed by concept id.")

    rows = {}
    for raw_concept_id, detail in data.items():
        concept_id = int(raw_concept_id)
        if "labels" not in detail:
            raise KeyError(f"Concept {concept_id} has no labels field.")
        labels = detail["labels"]
        images = detail["images"]
        if len(images) != len(labels):
            raise ValueError(f"Concept {concept_id} has {len(images)} images and {len(labels)} labels.")
        rows[concept_id] = list(zip(images, labels))
    return rows


def build_cbl_tensors(train_side, annotations, concepts):
    features, image_names, class_labels = train_side
    image_to_index = {
        str(image_name): index
        for index, image_name in enumerate(image_names)
    }
    concept_to_col = {row["concept_id"]: col for col, row in enumerate(concepts)}

    concept_labels = torch.full((len(image_names), len(concepts)), -1, dtype=torch.int8)
    used_labels = 0
    known_image_indices = set()

    for concept_id, pairs in annotations.items():
        if concept_id not in concept_to_col:
            continue
        col = concept_to_col[concept_id]
        for image_name, label in pairs:
            if label is None:
                continue
            key = str(image_name)
            if key not in image_to_index:
                continue
            row = image_to_index[key]
            value = int(label)
            if concept_labels[row, col] != -1 and concept_labels[row, col] != value:
                raise ValueError(
                    f"Conflicting labels for concept {concept_id} on image {image_name}: "
                    f"{int(concept_labels[row, col])} and {value}."
                )
            concept_labels[row, col] = value
            used_labels += 1
            known_image_indices.add(row)

    if not known_image_indices:
        raise ValueError("No annotation image names matched activations_train.pt or activations_val.pt.")

    return (
        features.float(),
        concept_labels,
        class_labels.long(),
        image_names,
        {
            "annotation_images_with_labels": len(known_image_indices),
            "annotation_labels_used": used_labels,
        },
    )


def stratified_concept_split(
    concept_labels,
    val_frac=0.10,
    seed=42,
    ensure_pos_left_in_train=True,
    round_mode="round",
    min_val_pos_if_tot_ge2=True,
):
    assert concept_labels.ndim == 2
    N, C = concept_labels.shape
    rng = np.random.default_rng(seed)

    r_pos_t, c_pos_t = torch.nonzero(concept_labels == 1, as_tuple=True)
    r_neg_t, c_neg_t = torch.nonzero(concept_labels == 0, as_tuple=True)
    r_pos, c_pos = r_pos_t.cpu().numpy(), c_pos_t.cpu().numpy()
    r_neg, c_neg = r_neg_t.cpu().numpy(), c_neg_t.cpu().numpy()

    Apos = sp.csr_matrix((np.ones_like(r_pos, dtype=np.uint8), (r_pos, c_pos)), shape=(N, C))
    Aneg = sp.csr_matrix((np.ones_like(r_neg, dtype=np.uint8), (r_neg, c_neg)), shape=(N, C))
    Lpos = Apos.T.tocsr()
    Lneg = Aneg.T.tocsr()

    pos_tot = np.asarray(Apos.sum(axis=0)).ravel().astype(np.int64)
    neg_tot = np.asarray(Aneg.sum(axis=0)).ravel().astype(np.int64)
    known_tot = pos_tot + neg_tot

    def rnd(x):
        if round_mode == "floor":
            return np.floor(x)
        if round_mode == "ceil":
            return np.ceil(x)
        return np.rint(x)

    t_pos = rnd(val_frac * pos_tot).astype(np.int64)
    t_neg = rnd(val_frac * neg_tot).astype(np.int64)
    t_known = rnd(val_frac * known_tot).astype(np.int64)

    if min_val_pos_if_tot_ge2:
        bump = (pos_tot >= 2) & (t_pos == 0)
        t_pos[bump] = 1
        t_pos = np.minimum(t_pos, np.maximum(pos_tot - 1, 0))

    pos_per_row = Apos.getnnz(axis=1)
    neg_per_row = Aneg.getnnz(axis=1)
    unk_per_row = (C - (pos_per_row + neg_per_row)).astype(np.int64)
    inv = np.zeros(C, dtype=np.float32)
    nonzero = pos_tot > 0
    inv[nonzero] = 1.0 / pos_tot[nonzero]
    rarity = (Apos @ inv).astype(np.float32)

    K = int(round(val_frac * N))
    K = max(0, min(N, K))
    val = np.zeros(N, dtype=bool)
    cur_pos = np.zeros(C, dtype=np.int64)
    cur_neg = np.zeros(C, dtype=np.int64)
    cur_known = np.zeros(C, dtype=np.int64)
    val_size = 0

    def row_pos_labels(i):
        start, end = Apos.indptr[i], Apos.indptr[i + 1]
        return Apos.indices[start:end]

    def row_neg_labels(i):
        start, end = Aneg.indptr[i], Aneg.indptr[i + 1]
        return Aneg.indices[start:end]

    def row_known_labels(i):
        pos_labels = row_pos_labels(i)
        neg_labels = row_neg_labels(i)
        if pos_labels.size == 0:
            return neg_labels
        if neg_labels.size == 0:
            return pos_labels
        return np.union1d(pos_labels, neg_labels)

    def affected_rows(i):
        chunks = []
        for col in row_pos_labels(i):
            start, end = Lpos.indptr[col], Lpos.indptr[col + 1]
            chunks.append(Lpos.indices[start:end])
        for col in row_neg_labels(i):
            start, end = Lneg.indptr[col], Lneg.indptr[col + 1]
            chunks.append(Lneg.indices[start:end])
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(chunks))

    def safe_to_take(i):
        if not ensure_pos_left_in_train:
            return True
        labels = row_pos_labels(i)
        if labels.size == 0:
            return True
        return not np.any((pos_tot[labels] - (cur_pos[labels] + 1)) <= 0)

    def violates_caps(i, relax_known_for_needed_pos=False):
        pos_labels = row_pos_labels(i)
        neg_labels = row_neg_labels(i)
        known_labels = row_known_labels(i)
        if pos_labels.size and np.any(cur_pos[pos_labels] >= t_pos[pos_labels]):
            return True
        if neg_labels.size and np.any(cur_neg[neg_labels] >= t_neg[neg_labels]):
            return True
        if relax_known_for_needed_pos and pos_labels.size and np.any(cur_pos[pos_labels] < t_pos[pos_labels]):
            return False
        if known_labels.size and np.any(cur_known[known_labels] >= t_known[known_labels]):
            return True
        return False

    def apply_row(i):
        nonlocal val_size
        if val[i]:
            return
        val[i] = True
        val_size += 1
        pos_labels = row_pos_labels(i)
        neg_labels = row_neg_labels(i)
        known_labels = row_known_labels(i)
        if pos_labels.size:
            cur_pos[pos_labels] += 1
        if neg_labels.size:
            cur_neg[neg_labels] += 1
        if known_labels.size:
            cur_known[known_labels] += 1

    pos_rows = np.flatnonzero(pos_per_row > 0)
    noise = rng.random(pos_rows.size)
    order1 = pos_rows[
        np.lexsort((noise, unk_per_row[pos_rows], -neg_per_row[pos_rows], -rarity[pos_rows]))
    ]
    for i in order1:
        if val_size >= K:
            break
        pos_labels = row_pos_labels(i)
        if pos_labels.size == 0 or not safe_to_take(i):
            continue
        if np.any(cur_pos[pos_labels] < t_pos[pos_labels]) and not violates_caps(i, relax_known_for_needed_pos=True):
            apply_row(i)

    if val_size >= K:
        return np.flatnonzero(~val).astype(np.int64).tolist(), np.flatnonzero(val).astype(np.int64).tolist()

    remain = np.flatnonzero(~val)
    if remain.size and np.any(cur_neg < t_neg):
        versions = np.zeros(N, dtype=np.int64)
        heap = []

        def step2_key(i):
            need_neg = cur_neg < t_neg
            neg_labels = row_neg_labels(i)
            help_neg = int(np.sum(need_neg[neg_labels])) if neg_labels.size else 0
            pos_labels = row_pos_labels(i)
            over_pos = int(np.sum(cur_pos[pos_labels] >= t_pos[pos_labels])) if pos_labels.size else 0
            return (over_pos, unk_per_row[i], -help_neg, i)

        for i in remain:
            key = step2_key(i)
            if key[2] < 0:
                heapq.heappush(heap, (key, versions[i], i))

        def push_rows(rows):
            for j in rows:
                if val[j]:
                    continue
                versions[j] += 1
                heapq.heappush(heap, (step2_key(j), versions[j], j))

        while val_size < K and heap:
            _, version, i = heapq.heappop(heap)
            if version != versions[i] or not safe_to_take(i) or violates_caps(i):
                continue
            need_neg = cur_neg < t_neg
            neg_labels = row_neg_labels(i)
            if neg_labels.size == 0 or np.sum(need_neg[neg_labels]) <= 0:
                continue
            apply_row(i)
            push_rows(affected_rows(i))

    if val_size >= K:
        return np.flatnonzero(~val).astype(np.int64).tolist(), np.flatnonzero(val).astype(np.int64).tolist()

    rest = np.flatnonzero(~val)
    if rest.size:
        versions = np.zeros(N, dtype=np.int64)
        heap = []

        def step3_key(i):
            pos_labels = row_pos_labels(i)
            neg_labels = row_neg_labels(i)
            known_labels = row_known_labels(i)
            v_pos = int(np.sum(cur_pos[pos_labels] >= t_pos[pos_labels])) if pos_labels.size else 0
            v_neg = int(np.sum(cur_neg[neg_labels] >= t_neg[neg_labels])) if neg_labels.size else 0
            v_known = int(np.sum(cur_known[known_labels] >= t_known[known_labels])) if known_labels.size else 0
            return (5 * v_pos + 2 * v_known + v_neg, unk_per_row[i], i)

        for i in rest:
            heapq.heappush(heap, (step3_key(i), versions[i], i))

        need = K - val_size
        taken = 0

        def push_rows3(rows):
            for j in rows:
                if val[j]:
                    continue
                versions[j] += 1
                heapq.heappush(heap, (step3_key(j), versions[j], j))

        while taken < need and heap:
            key, version, i = heapq.heappop(heap)
            if version != versions[i]:
                continue
            if key[0] != 0:
                break
            if not safe_to_take(i) or violates_caps(i):
                continue
            apply_row(i)
            taken += 1
            push_rows3(affected_rows(i))

        need = K - val_size
        if need > 0:
            heap = []
            for i in np.flatnonzero(~val):
                heapq.heappush(heap, (step3_key(i), versions[i], i))
            while need > 0 and heap:
                _, version, i = heapq.heappop(heap)
                if version != versions[i]:
                    continue
                pos_labels = row_pos_labels(i)
                if pos_labels.size and np.any(cur_pos[pos_labels] >= t_pos[pos_labels]):
                    continue
                if not safe_to_take(i):
                    continue
                apply_row(i)
                need -= 1
                push_rows3(affected_rows(i))

    if val_size < K:
        rest_set = set(np.flatnonzero(~val).tolist())

        def delta_loss(i):
            pos_labels = row_pos_labels(i)
            neg_labels = row_neg_labels(i)
            known_labels = row_known_labels(i)
            value = 0.0
            if pos_labels.size:
                value += 5.0 * np.sum(np.abs((cur_pos[pos_labels] + 1) - t_pos[pos_labels]) - np.abs(cur_pos[pos_labels] - t_pos[pos_labels]))
            if neg_labels.size:
                value += np.sum(np.abs((cur_neg[neg_labels] + 1) - t_neg[neg_labels]) - np.abs(cur_neg[neg_labels] - t_neg[neg_labels]))
            if known_labels.size:
                value += 2.0 * np.sum(np.abs((cur_known[known_labels] + 1) - t_known[known_labels]) - np.abs(cur_known[known_labels] - t_known[known_labels]))
            return value, unk_per_row[i], i

        for _ in range(K - val_size):
            best_i = None
            best_key = (np.inf, None, None)
            for i in list(rest_set):
                if val[i]:
                    rest_set.discard(i)
                    continue
                if not safe_to_take(i):
                    continue
                key = delta_loss(i)
                if key < best_key:
                    best_key = key
                    best_i = i
            if best_i is None:
                for i in list(rest_set):
                    if not val[i] and row_pos_labels(i).size == 0:
                        best_i = i
                        break
            if best_i is None:
                break
            apply_row(best_i)
            rest_set.discard(best_i)

    val_idx = np.flatnonzero(val).astype(np.int64)
    train_idx = np.flatnonzero(~val).astype(np.int64)
    assert val_idx.size == K, f"VAL size {val_idx.size} != K {K}"
    return train_idx.tolist(), val_idx.tolist()


def make_cbl_split(concept_labels, config):
    val_frac = float(config["cbl_val_split"])
    seed = int(config["seed"])
    if bool(config["cbl_stratification"]):
        return stratified_concept_split(
            concept_labels,
            val_frac=val_frac,
            seed=seed,
            ensure_pos_left_in_train=True,
            round_mode="round",
            min_val_pos_if_tot_ge2=True,
        )

    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(concept_labels.shape[0], generator=rng)
    val_size = int(round(val_frac * concept_labels.shape[0]))
    val_idx = order[:val_size].tolist()
    train_idx = order[val_size:].tolist()
    return train_idx, val_idx


def keep_annotated_indices(indices, concept_labels):
    annotated = (concept_labels != -1).any(dim=1)
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    return index_tensor[annotated[index_tensor]].tolist()


def print_split_summary(concept_labels, train_idx, val_idx):
    train_idx = torch.as_tensor(train_idx, dtype=torch.long)
    val_idx = torch.as_tensor(val_idx, dtype=torch.long)

    train_concept_labels = concept_labels.index_select(0, train_idx)
    val_concept_labels = concept_labels.index_select(0, val_idx)

    known_train = (train_concept_labels != -1).sum(dim=0)
    known_val = (val_concept_labels != -1).sum(dim=0)
    pos_train = (train_concept_labels == 1).sum(dim=0)
    pos_val = (val_concept_labels == 1).sum(dim=0)
    pos_total = (concept_labels == 1).sum(dim=0)

    zero_known_val = int((known_val == 0).sum().item())
    zero_pos_val = int(((pos_val == 0) & (pos_total > 0)).sum().item())
    zero_pos_val_with_known = int(((pos_val == 0) & (pos_total > 0) & (known_val > 0)).sum().item())

    valid = (known_train > 0) & (known_val > 0)
    train_rate = torch.zeros(concept_labels.shape[1], dtype=torch.float64)
    val_rate = torch.zeros(concept_labels.shape[1], dtype=torch.float64)
    train_rate[valid] = pos_train[valid].double() / known_train[valid].double()
    val_rate[valid] = pos_val[valid].double() / known_val[valid].double()

    eps = 1e-8
    relative_deviation = torch.zeros(concept_labels.shape[1], dtype=torch.float64)
    relative_deviation[valid] = (
        (train_rate[valid] - val_rate[valid]).abs()
        / (0.5 * (train_rate[valid] + val_rate[valid]) + eps)
    )
    valid_deviation = relative_deviation[valid].numpy()
    mean_relative_deviation = float(np.mean(valid_deviation)) if valid_deviation.size else float("nan")
    p90_relative_deviation = float(np.percentile(valid_deviation, 90)) if valid_deviation.size else float("nan")

    print(
        "CBL split quality: "
        f"mean pos-rate drift={mean_relative_deviation * 100:.2f}%, "
        f"p90={p90_relative_deviation * 100:.2f}%, "
        f"concepts without val labels={zero_known_val}, "
        f"concepts without val positives={zero_pos_val} "
        f"({zero_pos_val_with_known} with val labels)."
    )


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        return loss[targets != -1].mean()


def compute_pos_weight(train_concept_labels, device):
    pos = (train_concept_labels == 1).sum(dim=0)
    neg = (train_concept_labels == 0).sum(dim=0)
    return (neg / (pos + 1e-8)).to(device)


def make_optimizer(model, config):
    name = config["cbl_optimizer"]
    lr = float(config["cbl_lr"])
    weight_decay = float(config["cbl_weight_decay"])
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown cbl_optimizer: {name}")


def run_cbl_epoch(cbl, loader, loss_fn, optimizer, device):
    cbl.train()
    total_loss = 0.0
    for features, concept_targets, _ in loader:
        features = features.to(device)
        concept_targets = concept_targets.to(device)
        logits = cbl(features)
        loss = loss_fn(logits, concept_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def cbl_roc_auc_summary(logits_by_batch, targets_by_batch, tail_frac=0.10):
    scores = torch.cat(logits_by_batch, dim=0).numpy()
    targets = torch.cat(targets_by_batch, dim=0).numpy()
    per_concept_auc = []

    for concept_idx in range(targets.shape[1]):
        known = targets[:, concept_idx] != -1
        if not known.any():
            continue
        y = targets[known, concept_idx]
        if (y == 1).sum() == 0 or (y == 0).sum() == 0:
            continue
        per_concept_auc.append(roc_auc_score(y, scores[known, concept_idx]))

    if not per_concept_auc:
        return float("nan"), float("nan")

    aucs = np.asarray(per_concept_auc, dtype=float)
    k = max(1, int(np.floor(len(aucs) * tail_frac)))
    worst_auc = float(np.partition(aucs, k - 1)[:k].mean())
    return float(aucs.mean()), worst_auc


def validate_cbl(cbl, loader, loss_fn, device):
    cbl.eval()
    total_loss = 0.0
    logits_by_batch = []
    targets_by_batch = []
    with torch.no_grad():
        for features, concept_targets, _ in loader:
            features = features.to(device)
            concept_targets = concept_targets.to(device)
            logits = cbl(features)
            loss = loss_fn(logits, concept_targets)
            total_loss += loss.item()

            logits_by_batch.append(logits.cpu())
            targets_by_batch.append(concept_targets.cpu())

    macro_auc, worst_auc = cbl_roc_auc_summary(logits_by_batch, targets_by_batch)
    return total_loss / len(loader), macro_auc, worst_auc


def train_cbl(cbl, train_loader, val_loader, config, device):
    optimizer = make_optimizer(cbl, config)
    if bool(config["cbl_auto_weight"]):
        print("Using automatic positive weights.")
        train_concept_labels = train_loader.dataset.tensors[1]
        pos_weight = compute_pos_weight(train_concept_labels, device)
        print(
            "Positive weight stats: "
            f"mean={pos_weight.mean().item():.4f}, std={pos_weight.std().item():.4f}, "
            f"min={pos_weight.min().item():.4f}, max={pos_weight.max().item():.4f}"
        )
    else:
        print("Using no positive weights.")
        pos_weight = None

    loss_train = MaskedBCEWithLogitsLoss(pos_weight=pos_weight)
    loss_val = MaskedBCEWithLogitsLoss()
    patience = int(config["cbl_patience"])
    no_improve_epochs = 0
    best_val_loss = float("inf")
    best_epoch = None
    best_state = None
    history = []

    for epoch in range(int(config["cbl_epochs"])):
        train_loss = run_cbl_epoch(cbl, train_loader, loss_train, optimizer, device)
        val_loss, val_auc, val_worst_auc = validate_cbl(cbl, val_loader, loss_val, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_roc_auc": val_auc,
            "val_worst_10pct_roc_auc": val_worst_auc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"CBL epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
            f"val_roc_auc={val_auc:.4f}, val_worst10_roc_auc={val_worst_auc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy_state_dict(cbl)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print(f"Stopping CBL early at epoch {epoch}.")
            break

    unwrap_model(cbl).load_state_dict(best_state)
    return history, best_epoch, best_val_loss


def concept_logits_for_tensors(cbl, features, labels, batch_size, device):
    loader = DataLoader(TensorDataset(features.float(), labels.long()), batch_size=batch_size, shuffle=False)
    chunks = []
    label_chunks = []
    cbl.eval()
    with torch.no_grad():
        for feature_batch, label_batch in loader:
            chunks.append(cbl(feature_batch.to(device)).cpu())
            label_chunks.append(label_batch)
    return torch.cat(chunks, dim=0), torch.cat(label_chunks, dim=0)


def concept_logits_for_split(cbl, split, batch_size, device):
    features, _, labels = split
    return concept_logits_for_tensors(cbl, features, labels, batch_size, device)


def normalize_concept_features(train_features, *other_features):
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0)
    normalized = [(train_features - mean) / std]
    normalized.extend((features - mean) / std for features in other_features)
    return mean, std, normalized


def make_indexed_loader(features, labels, batch_size, saga_auto_weight, device):
    sample_weights = None
    if saga_auto_weight:
        unique_classes, class_counts = torch.unique(labels, return_counts=True)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
        sample_weights = class_weights[labels].to(device)
    dataset = IndexedTensorDataset(features, labels, sample_weight=sample_weights)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_sparse_final(linear, train_loader, val_loader, config):
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    metadata = {"max_reg": {"nongrouped": float(config["saga_lam"])}}
    return glm_saga(
        linear,
        train_loader,
        float(config["saga_step_size"]),
        int(config["saga_n_iters"]),
        0.99,
        epsilon=1,
        k=1,
        val_loader=val_loader,
        do_zero=False,
        metadata=metadata,
        n_ex=len(train_loader.dataset),
        n_classes=linear.weight.shape[0],
        verbose=True,
    )


def accuracy(final_layer, features, labels, device):
    final_layer.eval()
    with torch.no_grad():
        logits = final_layer(features.to(device))
        preds = logits.argmax(dim=1).cpu()
    return float((preds == labels.cpu()).float().mean().item())


def class_accuracy_summary(final_layer, features, labels, class_names, device):
    final_layer.eval()
    with torch.no_grad():
        logits = final_layer(features.to(device))
        preds = logits.argmax(dim=1).cpu()

    labels = labels.cpu()
    correct = torch.zeros(len(class_names), dtype=torch.float32)
    total = torch.zeros(len(class_names), dtype=torch.float32)
    for pred, target in zip(preds, labels):
        total[target] += 1
        if pred == target:
            correct[target] += 1

    per_class = torch.zeros_like(correct)
    nonzero = total > 0
    per_class[nonzero] = correct[nonzero] / total[nonzero]
    overall = correct.sum() / total.sum()
    balanced = per_class[nonzero].mean()
    return {
        "Per class accuracy": {
            class_names[i]: f"{per_class[i].item() * 100.0:.2f}"
            for i in range(len(class_names))
        },
        "Balanced accuracy": f"{balanced.item() * 100.0:.2f}",
        "Overall accuracy": f"{overall.item() * 100.0:.2f}",
        "Datapoints": f"{int(total.sum().item())}",
    }


def prepare_output_dir(config, output_dir):
    if output_dir:
        return Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "cbm" / config["dataset"] / f"cbm_{config['dataset']}_{timestamp}"


def save_concepts(path, concepts):
    with Path(path).open("w") as f:
        for row in concepts:
            f.write(row["concept_name"] + "\n")


def cbm_config_snapshot(config):
    fixed_keys = [
        "dataset",
        "backbone",
        "feature_layer",
        "val_split",
        "seed",
        "data_parallel",
    ]
    prefix_keys = sorted(
        key for key in config
        if key.startswith("cbl_") or key.startswith("saga_")
    )
    keys = fixed_keys + [key for key in prefix_keys if key not in fixed_keys]
    return {
        key: config[key]
        for key in keys
        if key in config
    }


def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device or config.get("device", "cuda"))
    annotation_dir = Path(args.annotation_dir)
    activation_dir = Path(args.activation_dir)
    annotations_path = annotation_dir / "annotations_train.pt"
    test_annotations_path = annotation_dir / "annotations_test.pt"
    output_dir = prepare_output_dir(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    concepts = read_concepts_from_annotations(annotations_path)
    validate_annotation_pair(annotations_path, test_annotations_path, concepts)
    annotations = read_annotation_labels(annotations_path)
    train_split, val_split, test_split = load_all_activation_splits(activation_dir)
    train_side = concatenate_train_side(train_split, val_split)
    activation_features, concept_labels, class_y, image_names, annotation_stats = build_cbl_tensors(
        train_side,
        annotations,
        concepts,
    )

    split_kind = "stratified" if bool(config["cbl_stratification"]) else "random"
    print(
        f"Splitting annotated concept labels into CBL train/val sets ({split_kind}, "
        f"val_split={config['cbl_val_split']}).",
        flush=True,
    )
    train_idx_full, cbl_val_idx_full = make_cbl_split(concept_labels, config)
    train_idx = keep_annotated_indices(train_idx_full, concept_labels)
    cbl_val_idx = keep_annotated_indices(cbl_val_idx_full, concept_labels)
    train_dataset = TensorDataset(
        activation_features[train_idx],
        concept_labels[train_idx].float(),
        class_y[train_idx],
    )
    val_dataset = TensorDataset(
        activation_features[cbl_val_idx],
        concept_labels[cbl_val_idx].float(),
        class_y[cbl_val_idx],
    )
    train_loader = DataLoader(train_dataset, batch_size=int(config["cbl_batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config["cbl_batch_size"]), shuffle=False)

    print(
        "Annotation data: "
        f"{annotation_stats['annotation_images_with_labels']} images with labels, "
        f"{annotation_stats['annotation_labels_used']} known concept labels."
    )
    cbl = ConceptLayer(
        in_features=activation_features.shape[1],
        out_features=len(concepts),
        num_hidden=int(config["cbl_hidden_layers"]),
    ).to(device)
    if bool(config["data_parallel"]) and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs for CBL.")
        cbl = nn.DataParallel(cbl)

    print(
        "CBL train/val images: "
        f"{len(train_idx)}/{len(cbl_val_idx)} annotated "
        f"({len(train_idx_full)}/{len(cbl_val_idx_full)} total)."
    )
    print_split_summary(concept_labels, train_idx_full, cbl_val_idx_full)
    history, best_epoch, best_val_loss = train_cbl(cbl, train_loader, val_loader, config, device)
    torch.save(unwrap_model(cbl).state_dict(), output_dir / "cbl.pt")
    print(f"Saved best CBL from epoch {best_epoch}.")

    test_annotations = read_annotation_labels(test_annotations_path)
    test_activation_features, test_concept_labels, test_class_labels, _, test_annotation_stats = build_cbl_tensors(
        test_split,
        test_annotations,
        concepts,
    )
    test_cbl_loader = DataLoader(
        TensorDataset(test_activation_features.float(), test_concept_labels.float(), test_class_labels),
        batch_size=int(config["cbl_batch_size"]),
        shuffle=False,
    )
    test_cbl_loss, test_cbl_auc, test_cbl_worst_auc = validate_cbl(
        cbl,
        test_cbl_loader,
        MaskedBCEWithLogitsLoss(),
        device,
    )
    print(
        "CBL test: "
        f"loss={test_cbl_loss:.6f}, "
        f"roc_auc={test_cbl_auc:.4f}, "
        f"worst10_roc_auc={test_cbl_worst_auc:.4f}"
    )

    saga_batch_size = int(config["saga_batch_size"])
    train_concepts, train_labels = concept_logits_for_split(cbl, train_split, saga_batch_size, device)
    val_concepts, val_labels = concept_logits_for_split(cbl, val_split, saga_batch_size, device)
    test_concepts, test_labels = concept_logits_for_split(cbl, test_split, saga_batch_size, device)
    _, _, normalized = normalize_concept_features(train_concepts, val_concepts, test_concepts)
    train_concepts, val_concepts, test_concepts = normalized

    num_classes = len([name for name in get_classes(config["dataset"]) if name])
    final_layer = nn.Linear(len(concepts), num_classes).to(device)
    indexed_train_loader = make_indexed_loader(
        train_concepts,
        train_labels,
        saga_batch_size,
        bool(config["saga_auto_weight"]),
        device,
    )
    final_val_loader = DataLoader(TensorDataset(val_concepts, val_labels), batch_size=saga_batch_size, shuffle=False)
    output_proj = train_sparse_final(final_layer, indexed_train_loader, final_val_loader, config)
    path = output_proj["path"][0]
    final_layer.weight.data = path["weight"].to(device)
    final_layer.bias.data = path["bias"].to(device)
    torch.save(final_layer.state_dict(), output_dir / "final.pt")

    class_names = [name for name in get_classes(config["dataset"]) if name]
    test_accuracy = accuracy(final_layer, test_concepts, test_labels, device)
    val_accuracy = accuracy(final_layer, val_concepts, val_labels, device)
    test_class_summary = class_accuracy_summary(final_layer, test_concepts, test_labels, class_names, device)
    nnz = int((final_layer.weight.detach().abs() > 1e-5).sum().item())
    total_weights = int(final_layer.weight.numel())
    solver_metrics = dict(path.get("metrics", {}))
    solver_metrics.pop("loss_test", None)
    solver_metrics.pop("acc_test", None)
    solver_metrics["test_accuracy"] = test_accuracy
    metrics = {
        "dataset": config["dataset"],
        "num_concepts": len(concepts),
        "input_dim": int(activation_features.shape[1]),
        "cbl_train_images": len(train_idx),
        "cbl_val_images": len(cbl_val_idx),
        "cbl_full_train_images": len(train_idx_full),
        "cbl_full_val_images": len(cbl_val_idx_full),
        "best_cbl_epoch": best_epoch,
        "best_cbl_val_loss": best_val_loss,
        "test_cbl_loss": test_cbl_loss,
        "test_cbl_macro_roc_auc": test_cbl_auc,
        "test_cbl_worst_10pct_roc_auc": test_cbl_worst_auc,
        "test_cbl_images": int(test_activation_features.shape[0]),
        "test_cbl_known_labels": test_annotation_stats["annotation_labels_used"],
        "final_val_accuracy": val_accuracy,
        "final_test_accuracy": test_accuracy,
        "final_test_balanced_accuracy": float(test_class_summary["Balanced accuracy"]) / 100.0,
        "final_nonzero_weights": nnz,
        "final_total_weights": total_weights,
        "final_fraction_nonzero": nnz / total_weights,
        "saga_lam": float(path["lam"]),
        "saga_lr": float(path["lr"]),
        "saga_alpha": float(path["alpha"]),
        "saga_time": float(path["time"]),
        "saga_metrics": solver_metrics,
        "per_class_accuracies": test_class_summary,
        **annotation_stats,
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    concepts_txt_path = output_dir / "concepts.txt"
    save_concepts(concepts_txt_path, concepts)
    (output_dir / "run_info.json").write_text(
        json.dumps(
            {
                "config": args.config,
                "activation_dir": str(activation_dir),
                "annotation_dir": str(annotation_dir),
                "device": str(device),
                "parameters": cbm_config_snapshot(config),
            },
            indent=2,
        )
    )

    print(f"Saved CBM to {output_dir}")
    print(f"Final val accuracy: {val_accuracy:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Final test balanced accuracy: {float(test_class_summary['Balanced accuracy']) / 100.0:.4f}")
    print(
        "Final layer sparsity: "
        f"{nnz}/{total_weights} non-zero weights ({nnz / total_weights:.4f})"
    )


if __name__ == "__main__":
    main()
