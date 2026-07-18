import argparse
import csv
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from google import genai
from google.genai import errors, types
from openai import OpenAI
from PIL import Image

from mcbm.data.utils import get_llm_image_transform
from mcbm.scripts.train_sae import forward_sae
from mcbm.scripts.name_concepts import (
    decoder_vector_for_unit,
    get_system_prompt,
    image_to_data_url,
    image_to_jpeg_bytes,
    load_activation_rows,
    load_autoencoder,
    load_kept_indices,
    load_image,
    openai_image_detail,
    resolve_api_key,
    validate_llm_options,
)


IMAGES_PER_BATCH = 25
ACTIVE_PER_BATCH = 12
SIMILAR_PER_BATCH = 6
RANDOM_PER_BATCH = 6
MAX_ACTIVE_IMAGES = 500
ACTIVE_SELECTION_QUANTILE = 0.95
GEMINI_REQUEST_DELAY_SECONDS = 5
GEMINI_TRANSIENT_ERROR_DELAY_SECONDS = 60


class InvalidLLMOutput(ValueError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Annotate merged SAE concepts with an LLM.")
    parser.add_argument("--config", required=True, help="Path to dataset config JSON.")
    parser.add_argument("--concept-dir", required=True, help="Merged concept run directory.")
    parser.add_argument(
        "--split",
        choices=["all", "train", "test"],
        default="all",
        help="Annotation split. By default, annotate train and test into the same run folder.",
    )
    parser.add_argument("--output-dir", help="Directory for annotation outputs.")
    parser.add_argument("--device", help="Torch device override, e.g. cuda or cuda:0.")
    parser.add_argument("--llm-provider", choices=["openai", "gemini"], default="openai", help="LLM provider.")
    parser.add_argument("--llm-model", default="gpt-4.1", help="LLM model name for annotation.")
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"])
    parser.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--save-images", action="store_true", help="Save reference and annotation grids.")
    parser.add_argument("--units", help="Comma-separated merged concept ids to annotate, e.g. 1,2,3.")
    return parser.parse_args()


def parse_units(value):
    if value is None:
        return None
    units = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not units:
        raise ValueError("--units must contain at least one unit id.")
    if min(units) < 0:
        raise ValueError("--units cannot contain negative ids.")
    return set(units)


def load_run_info(run_dir):
    path = Path(run_dir) / "run_info.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return json.loads(path.read_text())


def require_run_info_path(run_info, key, run_dir):
    value = run_info.get(key)
    if value is None:
        raise KeyError(f"{Path(run_dir) / 'run_info.json'} must contain {key}.")
    return Path(value)


def read_concepts(path):
    rows = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if not {"unit", "concept_name", "merged_from"}.issubset(fieldnames):
            raise ValueError("Concept CSV must have unit, concept_name, and merged_from columns.")

        for row in reader:
            concept_name = row["concept_name"].strip()
            if not concept_name:
                continue
            merged_from = [
                int(value.strip())
                for value in row["merged_from"].split(",")
                if value.strip()
            ]
            rows.append(
                {
                    "concept_id": int(row["unit"]),
                    "concept_name": concept_name,
                    "merged_from": merged_from,
                }
            )

    if not rows:
        raise ValueError(f"{path} contains no concepts.")
    return rows


def make_grid(images, columns=5):
    if len(images) != IMAGES_PER_BATCH:
        raise ValueError(f"Expected {IMAGES_PER_BATCH} images, got {len(images)}.")

    width, height = images[0].size
    rows = math.ceil(len(images) / columns)
    grid = Image.new("RGB", (columns * width, rows * height), color=(255, 255, 255))
    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        grid.paste(image.convert("RGB"), (column * width, row * height))
    return grid


def load_grid(dataset_name, image_names, image_splits, indices, image_transform):
    images = []
    for index in indices:
        image = load_image(
            dataset_name,
            image_splits[int(index)],
            image_names[int(index)],
        )
        image = image_transform(image)
        images.append(image)
    return make_grid(images)


def parse_predictions_json(text):
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidLLMOutput(f"Could not parse structured annotation output: {text}") from exc
    predictions = parsed.get("labels")
    if not isinstance(predictions, list):
        raise InvalidLLMOutput(f"Structured output must contain a labels list: {text}")
    try:
        predictions = [int(value) for value in predictions]
    except (TypeError, ValueError) as exc:
        raise InvalidLLMOutput(f"Labels must be integers 0 or 1, got: {text}") from exc
    if len(predictions) != IMAGES_PER_BATCH:
        raise InvalidLLMOutput(f"Expected {IMAGES_PER_BATCH} labels, got {len(predictions)}.")
    if any(value not in (0, 1) for value in predictions):
        raise InvalidLLMOutput(f"Labels must contain only 0 and 1, got: {text}")
    return predictions


ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {"type": "integer", "enum": [0, 1]},
            "minItems": IMAGES_PER_BATCH,
            "maxItems": IMAGES_PER_BATCH,
        },
    },
    "required": ["labels"],
    "additionalProperties": False,
}


GEMINI_ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": IMAGES_PER_BATCH,
            "maxItems": IMAGES_PER_BATCH,
        },
    },
    "required": ["labels"],
}


def call_annotator_with_retry(args, config, concept_name, reference_grid, task_grid):
    try:
        return call_annotator(args, config, concept_name, reference_grid, task_grid)
    except InvalidLLMOutput as first_error:
        print(f"Invalid LLM output; retrying once. Error: {first_error}")
        try:
            return call_annotator(args, config, concept_name, reference_grid, task_grid)
        except InvalidLLMOutput as second_error:
            raise InvalidLLMOutput(f"LLM output was invalid twice. Last error: {second_error}") from second_error


def call_openai_annotator(args, config, concept_name, reference_grid, task_grid):
    client = OpenAI(api_key=resolve_api_key("openai"))
    image_detail = openai_image_detail(args.llm_model)
    request = {
        "model": args.llm_model,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "concept_annotation_output",
                "strict": True,
                "schema": ANNOTATION_SCHEMA,
            }
        },
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": get_system_prompt(config)}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "We're studying neurons in a neural network. Each neuron looks for a particular concept in an image.\n"
                            f"Below are 25 reference images (not to label) that strongly contain the concept '{concept_name}'.\n"
                            "Carefully review these reference examples before starting the main task."
                        ),
                    },
                    {"type": "input_image", "image_url": image_to_data_url(reference_grid), "detail": image_detail},
                    {
                        "type": "input_text",
                        "text": (
                            f"Now, here are {IMAGES_PER_BATCH} new images in a 5x5 grid. Read left to right, top to bottom.\n"
                            f"For each image, predict '1' if the concept '{concept_name}' is present, or '0' if absent.\n"
                            "Return a JSON object with one key, labels, containing the 25 labels in order."
                        ),
                    },
                    {"type": "input_image", "image_url": image_to_data_url(task_grid), "detail": image_detail},
                ],
            },
        ],
    }
    if args.reasoning_effort is not None:
        request["reasoning"] = {"effort": args.reasoning_effort}
    else:
        request["temperature"] = 0

    response = client.responses.create(**request)
    return parse_predictions_json(response.output_text)


def call_gemini_annotator(args, config, concept_name, reference_grid, task_grid):
    parts = [
        types.Part.from_text(
            text=(
                "We're studying neurons in a neural network. Each neuron looks for a particular concept in an image.\n"
                f"Below are 25 reference images (not to label) that strongly contain the concept '{concept_name}'.\n"
                "Carefully review these reference examples before starting the main task."
            )
        ),
        types.Part.from_bytes(data=image_to_jpeg_bytes(reference_grid), mime_type="image/jpeg"),
        types.Part.from_text(
            text=(
                f"Now, here are {IMAGES_PER_BATCH} new images in a 5x5 grid. Read left to right, top to bottom.\n"
                f"For each image, predict '1' if the concept '{concept_name}' is present, or '0' if absent.\n"
                "Return a JSON object with one key, labels, containing the 25 labels in order."
            )
        ),
        types.Part.from_bytes(data=image_to_jpeg_bytes(task_grid), mime_type="image/jpeg"),
    ]

    client = genai.Client(api_key=resolve_api_key("gemini"))
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=args.llm_model,
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=get_system_prompt(config),
                    temperature=0,
                    thinking_config=(
                        types.ThinkingConfig(thinking_level=args.thinking_level)
                        if args.thinking_level is not None
                        else None
                    ),
                    response_mime_type="application/json",
                    response_schema=GEMINI_ANNOTATION_SCHEMA,
                ),
            )
            time.sleep(GEMINI_REQUEST_DELAY_SECONDS)
            break
        except (errors.ClientError, errors.ServerError) as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code not in (429, 503) or attempt == 1:
                raise
            print(
                f"Gemini returned {status_code}; waiting "
                f"{GEMINI_TRANSIENT_ERROR_DELAY_SECONDS} seconds and retrying once."
            )
            time.sleep(GEMINI_TRANSIENT_ERROR_DELAY_SECONDS)
    return parse_predictions_json(response.text)


def call_annotator(args, config, concept_name, reference_grid, task_grid):
    if args.llm_provider == "openai":
        return call_openai_annotator(args, config, concept_name, reference_grid, task_grid)
    if args.llm_provider == "gemini":
        return call_gemini_annotator(args, config, concept_name, reference_grid, task_grid)
    raise ValueError(f"Unsupported LLM provider: {args.llm_provider}")


def concept_activation(autoencoder, features, unit_ids, batch_size, device):
    unit_ids_tensor = torch.tensor(unit_ids, dtype=torch.long, device=device)
    chunks = []

    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            batch = features[start:end].to(device)
            learned, _ = forward_sae(autoencoder, batch)
            chunks.append(learned[:, unit_ids_tensor].detach().cpu())

    activations = torch.cat(chunks, dim=0)
    if activations.shape[1] == 1:
        return activations[:, 0]

    scales = []
    valid_unit_indices = []
    for unit_index in range(activations.shape[1]):
        positive_values = activations[:, unit_index][activations[:, unit_index] > 0]
        if positive_values.numel() == 0:
            continue
        scales.append(torch.quantile(positive_values, 0.95))
        valid_unit_indices.append(unit_index)

    if not valid_unit_indices:
        return torch.zeros(features.shape[0])

    scales = torch.stack(scales)
    normalized = activations[:, valid_unit_indices] / scales
    return normalized.max(dim=1).values


def concept_decoder_vector(autoencoder, unit_ids, input_dim, device):
    vectors = [
        decoder_vector_for_unit(autoencoder, unit_id, input_dim, device)
        for unit_id in unit_ids
    ]
    return torch.stack(vectors).mean(dim=0)


def original_unit_ids(post_pruning_unit_ids, kept_indices):
    unit_ids = torch.tensor(post_pruning_unit_ids, dtype=torch.long)
    if unit_ids.numel() == 0:
        raise ValueError("A merged concept must contain at least one source unit.")
    if int(unit_ids.min().item()) < 0 or int(unit_ids.max().item()) >= int(kept_indices.numel()):
        raise ValueError(
            f"merged_from contains unit ids outside the post-pruning range 0..{kept_indices.numel() - 1}."
        )
    return kept_indices[unit_ids].tolist()


def stratified_indices(indices, labels, n_samples, seed, shuffle=False):
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()

    rng = random.Random(seed)
    indices = [int(index) for index in indices]
    if shuffle:
        rng.shuffle(indices)

    buckets = {}
    for index in indices:
        buckets.setdefault(int(labels[index].item()), []).append(index)

    class_ids = list(buckets)
    rng.shuffle(class_ids)
    samples_per_class = max(1, n_samples // len(class_ids)) if class_ids else n_samples

    selected = []
    for class_id in class_ids:
        bucket = buckets[class_id]
        selected.extend(bucket[: min(samples_per_class, len(bucket))])
        if len(selected) >= n_samples:
            return torch.tensor(selected[:n_samples], dtype=torch.long)

    remaining = [index for index in indices if index not in set(selected)]
    rng.shuffle(remaining)
    selected.extend(remaining[: n_samples - len(selected)])
    return torch.tensor(selected[:n_samples], dtype=torch.long)


def get_annotation_budget(config, split):
    key = f"annotation_{split}_samples"
    if key not in config:
        raise KeyError(f"Config is missing required key: {key}")
    budget = int(config[key])
    if budget < 50 or budget % 50 != 0:
        raise ValueError(f"{key} must be a multiple of 50 and at least 50.")
    return budget


def image_split_for_activation_split(config, activation_file_split):
    if activation_file_split == "val" and config["val_split"] is not None:
        return "train"
    return activation_file_split


def load_annotation_activations(config, activation_dir, split):
    activation_dir = Path(activation_dir)
    if split == "train":
        parts = []
        for source_split in ["train", "val"]:
            features, image_names, labels = load_activation_rows(
                activation_dir / f"activations_{source_split}.pt"
            )
            image_split = image_split_for_activation_split(config, source_split)
            image_splits = [image_split] * len(image_names)
            parts.append((features, image_names, labels, image_splits))

        features = torch.cat([part[0] for part in parts], dim=0)
        image_names = [name for part in parts for name in part[1]]
        labels = torch.cat([part[2] for part in parts], dim=0)
        image_splits = [split_name for part in parts for split_name in part[3]]
        return features, image_names, labels, image_splits

    features, image_names, labels = load_activation_rows(activation_dir / "activations_test.pt")
    image_splits = ["test"] * len(image_names)
    return features, image_names, labels, image_splits


def load_reference_activations(activation_dir):
    activation_dir = Path(activation_dir)
    features, image_names, _labels = load_activation_rows(activation_dir / "activations_train.pt")
    image_splits = ["train"] * len(image_names)
    return features, image_names, image_splits


def active_candidate_indices(activations):
    active_indices = torch.where(activations > 0)[0]
    if active_indices.numel() < IMAGES_PER_BATCH:
        return None

    sorted_positions = torch.argsort(activations[active_indices], descending=True)
    active_indices = active_indices[sorted_positions]

    active_values = activations[active_indices]
    quantile_threshold = torch.quantile(active_values, ACTIVE_SELECTION_QUANTILE)
    high_active_indices = active_indices[active_values > quantile_threshold]
    if high_active_indices.numel() >= MAX_ACTIVE_IMAGES:
        return high_active_indices
    return active_indices[:MAX_ACTIVE_IMAGES]


def select_annotation_indices(activations, features, labels, annotation_budget, seed):
    active_indices = active_candidate_indices(activations)
    if active_indices is None:
        return None

    reference_indices = active_indices[:IMAGES_PER_BATCH].clone()
    n_active = min(int(active_indices.numel()), annotation_budget // 2, MAX_ACTIVE_IMAGES)
    n_active = (n_active // IMAGES_PER_BATCH) * IMAGES_PER_BATCH
    if n_active < IMAGES_PER_BATCH:
        return None
    active_indices = active_indices[:n_active]

    active_indices = stratified_indices(
        active_indices,
        labels,
        n_active,
        seed=seed,
        shuffle=False,
    )

    zero_indices = torch.where(activations <= 0)[0]
    if zero_indices.numel() < n_active:
        return None

    centroid = features[active_indices].mean(dim=0, keepdim=True)
    similarities = F.cosine_similarity(centroid, features[zero_indices], dim=1)
    similar_order = torch.argsort(similarities, descending=True)
    similar_source = zero_indices[similar_order]

    n_similar = n_active // 2
    n_random = n_active - n_similar
    similar_indices = stratified_indices(
        similar_source,
        labels,
        n_similar,
        seed=seed + 1,
        shuffle=False,
    )
    similar_set = set(similar_indices.tolist())
    random_source = torch.tensor(
        [int(index) for index in zero_indices.tolist() if int(index) not in similar_set],
        dtype=torch.long,
    )
    if random_source.numel() < n_random:
        return None
    random_indices = stratified_indices(
        random_source,
        labels,
        n_random,
        seed=seed + 2,
        shuffle=True,
    )

    return reference_indices, active_indices, similar_indices, random_indices


def select_test_annotation_indices(activations, features, labels, annotation_budget, seed, fallback_vector):
    active_indices = torch.where(activations > 0)[0]
    if active_indices.numel() > 0:
        sorted_positions = torch.argsort(activations[active_indices], descending=True)
        active_indices = active_indices[sorted_positions]
        n_active = min(int(active_indices.numel()), annotation_budget // 2, MAX_ACTIVE_IMAGES)
        n_active = max(1, n_active)
        active_indices = active_indices[:n_active]

    zero_indices = torch.where(activations <= 0)[0]
    n_nonactive = (2 * IMAGES_PER_BATCH) - int(active_indices.numel())
    if zero_indices.numel() < n_nonactive:
        return None

    if active_indices.numel() > 0:
        centroid = features[active_indices].mean(dim=0, keepdim=True)
    else:
        centroid = fallback_vector.detach().cpu().view(1, -1)
    similarities = F.cosine_similarity(centroid, features[zero_indices], dim=1)
    similar_order = torch.argsort(similarities, descending=True)
    similar_source = zero_indices[similar_order]

    n_similar = n_nonactive // 2
    n_random = n_nonactive - n_similar
    similar_indices = stratified_indices(
        similar_source,
        labels,
        n_similar,
        seed=seed + 1,
        shuffle=False,
    )
    similar_set = set(similar_indices.tolist())
    random_source = torch.tensor(
        [int(index) for index in zero_indices.tolist() if int(index) not in similar_set],
        dtype=torch.long,
    )
    if random_source.numel() < n_random:
        return None
    random_indices = stratified_indices(
        random_source,
        labels,
        n_random,
        seed=seed + 2,
        shuffle=True,
    )

    return active_indices, similar_indices, random_indices


def select_reference_indices(activations):
    active_indices = torch.where(activations > 0)[0]
    if active_indices.numel() < IMAGES_PER_BATCH:
        return None

    sorted_positions = torch.argsort(activations[active_indices], descending=True)
    active_indices = active_indices[sorted_positions]
    return active_indices[:IMAGES_PER_BATCH].clone()


def build_batches(active_indices, similar_indices, random_indices, seed):
    n_active = int(active_indices.numel())
    n_nonactive = int(similar_indices.numel() + random_indices.numel())
    num_batches = (n_active + n_nonactive) // IMAGES_PER_BATCH
    active_bin_size = num_batches
    active_bins = [
        active_indices[i * active_bin_size:(i + 1) * active_bin_size].tolist()
        for i in range(ACTIVE_PER_BATCH)
    ]
    leftover_active = active_indices[ACTIVE_PER_BATCH * active_bin_size:].tolist()

    rng = random.Random(seed)
    batches = []
    used_active = set()
    used_nonactive = set()

    for batch_index in range(num_batches):
        batch_indices = []
        batch_labels = []

        for bucket in active_bins:
            candidates = [index for index in bucket if index not in used_active]
            if not candidates:
                raise RuntimeError("No active image left for an annotation batch.")
            chosen = rng.choice(candidates)
            used_active.add(chosen)
            batch_indices.append(chosen)
            batch_labels.append(1)

        similar_pool = [
            int(index)
            for index in similar_indices.tolist()
            if int(index) not in used_nonactive
        ]
        if len(similar_pool) < SIMILAR_PER_BATCH:
            raise RuntimeError("No similar non-active images left for an annotation batch.")
        similar_chosen = rng.sample(similar_pool, SIMILAR_PER_BATCH)
        for index in similar_chosen:
            used_nonactive.add(index)
            batch_indices.append(index)
            batch_labels.append(0)

        random_pool = [
            int(index)
            for index in random_indices.tolist()
            if int(index) not in used_nonactive
        ]
        if len(random_pool) < RANDOM_PER_BATCH:
            raise RuntimeError("No random non-active images left for an annotation batch.")
        random_chosen = rng.sample(random_pool, RANDOM_PER_BATCH)
        for index in random_chosen:
            used_nonactive.add(index)
            batch_indices.append(index)
            batch_labels.append(0)

        if leftover_active:
            chosen = leftover_active.pop(0)
            used_active.add(chosen)
            batch_indices.append(chosen)
            batch_labels.append(1)
        else:
            similar_remaining = [
                index for index in similar_pool if index not in used_nonactive
            ]
            random_remaining = [
                index for index in random_pool if index not in used_nonactive
            ]
            if not (similar_remaining or random_remaining):
                raise RuntimeError("No extra image left for an annotation batch.")
            if len(similar_remaining) >= len(random_remaining):
                chosen = rng.choice(similar_remaining)
            else:
                chosen = rng.choice(random_remaining)
            used_nonactive.add(chosen)
            batch_indices.append(chosen)
            batch_labels.append(0)

        paired = list(zip(batch_indices, batch_labels))
        rng.shuffle(paired)
        batch_indices, batch_labels = zip(*paired)
        batches.append((list(batch_indices), list(batch_labels)))

    return batches


def build_test_batch(active_indices, similar_indices, random_indices, seed):
    rng = random.Random(seed)
    active = active_indices.tolist()
    similar = similar_indices.tolist()
    random_indices = random_indices.tolist()

    rng.shuffle(active)
    rng.shuffle(similar)
    rng.shuffle(random_indices)

    batches = []
    active_splits = [
        active[: math.ceil(len(active) / 2)],
        active[math.ceil(len(active) / 2):],
    ]
    batch_capacities = [IMAGES_PER_BATCH - len(active_split) for active_split in active_splits]
    similar_first = min(math.ceil(len(similar) / 2), batch_capacities[0])
    similar_splits = [similar[:similar_first], similar[similar_first:]]
    random_splits = [
        random_indices[: batch_capacities[0] - len(similar_splits[0])],
        random_indices[batch_capacities[0] - len(similar_splits[0]):],
    ]

    for batch_index in range(2):
        batch_indices = active_splits[batch_index] + similar_splits[batch_index] + random_splits[batch_index]
        batch_labels = [1] * len(active_splits[batch_index]) + [0] * (
            len(similar_splits[batch_index]) + len(random_splits[batch_index])
        )
        if len(batch_indices) != IMAGES_PER_BATCH:
            raise RuntimeError(f"Expected {IMAGES_PER_BATCH} test images, got {len(batch_indices)}.")
        paired = list(zip(batch_indices, batch_labels))
        rng.shuffle(paired)
        batch_indices, batch_labels = zip(*paired)
        batches.append((list(batch_indices), list(batch_labels)))

    return batches


def read_existing_summary(path):
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return {int(row["concept_id"]): row for row in reader}


def save_summary(path, rows):
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["concept_id", "concept_name", "llm_sae_agreement", "n_samples"],
        )
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda item: int(item["concept_id"])):
            writer.writerow(row)
    tmp_path.replace(path)


def save_run_info(path, config, args, activation_dir, sae_dir, checkpoint):
    run_info = {
        "dataset": config["dataset"],
        "config": args.config,
        "concept_dir": args.concept_dir,
        "activation_dir": str(activation_dir),
        "sae_dir": str(sae_dir),
        "sae_model": checkpoint["checkpoint_path"],
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "save_images": bool(args.save_images),
        "annotation_train_samples": config["annotation_train_samples"],
        "annotation_test_samples": config["annotation_test_samples"],
    }
    if args.llm_provider == "openai":
        run_info["reasoning_effort"] = args.reasoning_effort
        run_info["image_detail"] = openai_image_detail(args.llm_model)
    else:
        run_info["thinking_level"] = args.thinking_level
    path.write_text(json.dumps(run_info, indent=2))


def make_annotation_run_name(config, args):
    model_name = f"{args.llm_provider}_{args.llm_model}".replace(".", "_").replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"annotations_{config['dataset']}_{model_name}_{timestamp}"


def prepare_run_dir(config, args):
    if args.output_dir is not None:
        return Path(args.output_dir)
    return Path("outputs") / "annotations" / config["dataset"] / make_annotation_run_name(config, args)


def save_grid_if_requested(args, output_dir, split, concept_id, batch_index, reference_grid, task_grid):
    if not args.save_images:
        return
    grid_dir = output_dir / "images" / split / f"concept_{int(concept_id):06d}"
    grid_dir.mkdir(parents=True, exist_ok=True)
    if batch_index == 0:
        reference_grid.save(grid_dir / "reference_grid.jpg")
    task_grid.save(grid_dir / f"batch_{batch_index:03d}.jpg")


def annotate_concepts(config, args, activation_dir, sae_dir, concepts_path, device):
    run_dir = prepare_run_dir(config, args)
    run_dir.mkdir(parents=True, exist_ok=True)

    reference_features, reference_image_names, reference_image_splits = load_reference_activations(activation_dir)
    autoencoder, checkpoint = load_autoencoder(sae_dir, device)
    kept_indices = load_kept_indices(sae_dir)
    batch_size = int(checkpoint["sae_config"]["batch_size"])
    concepts = read_concepts(concepts_path)
    image_transform = get_llm_image_transform(config["backbone"])
    requested_units = parse_units(args.units)
    if requested_units is not None:
        available_units = {int(concept["concept_id"]) for concept in concepts}
        missing_units = sorted(requested_units - available_units)
        if missing_units:
            raise ValueError(f"Requested units are not present in {concepts_path}: {missing_units}")

    save_run_info(run_dir / "run_info.json", config, args, activation_dir, sae_dir, checkpoint)
    print(f"Saving annotations in {run_dir}.")
    splits = ["train", "test"] if args.split == "all" else [args.split]

    for split in splits:
        features, image_names, labels, image_splits = load_annotation_activations(
            config,
            activation_dir,
            split,
        )
        annotation_budget = get_annotation_budget(config, split)
        summary_path = run_dir / f"annotation_summary_{split}.csv"
        detail_path = run_dir / f"annotations_{split}.pt"
        summary = read_existing_summary(summary_path)
        details = torch.load(detail_path, map_location="cpu") if detail_path.exists() else {}

        print(f"Loaded {features.shape[0]} {split} activations.")
        if requested_units is None:
            print(f"Annotating {len(concepts)} merged concepts with up to {annotation_budget} images each.")
        else:
            print(f"Annotating {len(requested_units)} selected merged concepts with up to {annotation_budget} images each.")

        for concept in concepts:
            concept_id = int(concept["concept_id"])
            if requested_units is not None and concept_id not in requested_units:
                continue
            if concept_id in details:
                print(f"Skipping concept {concept_id} ({split} already annotated).")
                continue

            unit_ids = original_unit_ids(concept["merged_from"], kept_indices)
            activations = concept_activation(
                autoencoder,
                features,
                unit_ids,
                batch_size,
                device,
            )
            reference_activations = concept_activation(
                autoencoder,
                reference_features,
                unit_ids,
                batch_size,
                device,
            )
            reference_indices = select_reference_indices(reference_activations)
            if reference_indices is None:
                print(f"Skipping concept {concept_id}: not enough train reference images.")
                continue

            use_small_test_batch = split == "test" and torch.where(activations > 0)[0].numel() < IMAGES_PER_BATCH
            if use_small_test_batch:
                selected = select_test_annotation_indices(
                    activations,
                    features,
                    labels,
                    annotation_budget,
                    seed=int(config["seed"]),
                    fallback_vector=concept_decoder_vector(
                        autoencoder,
                        unit_ids,
                        features.shape[1],
                        device,
                    ),
                )
            else:
                selected = select_annotation_indices(
                    activations,
                    features,
                    labels,
                    annotation_budget,
                    seed=int(config["seed"]),
                )
            if selected is None:
                print(f"Skipping concept {concept_id}: not enough active or non-active images.")
                continue

            if use_small_test_batch:
                active_indices, similar_indices, random_indices = selected
            else:
                _, active_indices, similar_indices, random_indices = selected
            reference_grid = load_grid(
                config["dataset"],
                reference_image_names,
                reference_image_splits,
                reference_indices.tolist(),
                image_transform,
            )
            if use_small_test_batch:
                batches = build_test_batch(
                    active_indices,
                    similar_indices,
                    random_indices,
                    seed=int(config["seed"]) + concept_id,
                )
            else:
                batches = build_batches(
                    active_indices,
                    similar_indices,
                    random_indices,
                    seed=int(config["seed"]) + concept_id,
                )

            all_images = []
            llm_labels = []
            expected_labels = []
            for batch_index, (batch_indices, batch_labels) in enumerate(batches):
                task_grid = load_grid(config["dataset"], image_names, image_splits, batch_indices, image_transform)
                save_grid_if_requested(args, run_dir, split, concept_id, batch_index, reference_grid, task_grid)
                predictions = call_annotator_with_retry(
                    args,
                    config,
                    concept["concept_name"],
                    reference_grid,
                    task_grid,
                )
                all_images.extend([image_names[index] for index in batch_indices])
                llm_labels.extend(predictions)
                expected_labels.extend(batch_labels)
                print(f"Concept {concept_id}, {split} batch {batch_index + 1}/{len(batches)} annotated.")

            matches = sum(
                int(expected_label == llm_label)
                for expected_label, llm_label in zip(expected_labels, llm_labels)
            )
            agreement = matches / len(expected_labels)
            summary[concept_id] = {
                "concept_id": concept_id,
                "concept_name": concept["concept_name"],
                "llm_sae_agreement": agreement,
                "n_samples": len(expected_labels),
            }
            details[concept_id] = {
                "images": all_images,
                "labels": llm_labels,
                "concept": concept["concept_name"],
            }
            details = {
                int(saved_concept_id): {
                    "images": detail["images"],
                    "labels": detail["labels"],
                    "concept": detail["concept"],
                }
                for saved_concept_id, detail in sorted(details.items(), key=lambda item: int(item[0]))
            }
            save_summary(summary_path, summary)
            torch.save(details, detail_path)
            print(f"Annotated concept {concept_id} ({split}): llm_sae_agreement={agreement:.3f}, n={len(expected_labels)}")

        print(f"Saved annotation summary to {summary_path}")
        print(f"Saved annotation details to {detail_path}")


def main():
    args = parse_args()
    validate_llm_options(args)
    config = json.loads(Path(args.config).read_text())
    concept_run_info = load_run_info(args.concept_dir)
    activation_dir = require_run_info_path(concept_run_info, "activation_dir", args.concept_dir)
    sae_dir = require_run_info_path(concept_run_info, "sae_dir", args.concept_dir)
    concepts_path = Path(args.concept_dir) / "concept_names_merged.csv"
    device = torch.device(args.device or config.get("device", "cuda"))
    annotate_concepts(config, args, activation_dir, sae_dir, concepts_path, device)


if __name__ == "__main__":
    main()
