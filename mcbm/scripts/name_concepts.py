import argparse
import base64
import csv
import io
import json
import random
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from google import genai
from google.genai import types
from openai import OpenAI
from PIL import Image

from mcbm.data.utils import get_classes, get_dataset_domain, get_image_path, get_llm_image_transform, get_target_model
from mcbm.scripts.train_sae import forward_sae, make_autoencoder


class InvalidLLMOutput(ValueError):
    pass


def openai_image_detail(model_name):
    match = re.match(r"^gpt-(\d+)(?:\.(\d+))?(?:$|-)", model_name)
    if match is None or "-mini" in model_name or "-nano" in model_name or "-codex" in model_name:
        return "high"
    version = int(match.group(1)), int(match.group(2) or 0)
    return "original" if version >= (5, 4) else "high"


def parse_args():
    parser = argparse.ArgumentParser(description="Name SAE concepts from activating and contrastive images.")
    parser.add_argument("--config", required=True, help="Path to dataset config JSON.")
    parser.add_argument("--sae-dir", required=True, help="Directory containing SAE outputs.")
    parser.add_argument("--output-dir", help="Directory for concept naming outputs.")
    parser.add_argument("--device", help="Torch device override, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--save-images", action="store_true", help="Save the 20 rendered naming images for debugging.")
    parser.add_argument("--llm-provider", choices=["openai", "gemini"], default="openai", help="LLM provider.")
    parser.add_argument("--llm-model", default="gpt-4.1", help="LLM model name for concept naming.")
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"])
    parser.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--units", help="Comma-separated post-pruning unit ids to name, e.g. 1,2,3.")
    return parser.parse_args()


def validate_llm_options(args):
    if args.llm_provider == "openai" and args.thinking_level is not None:
        raise ValueError("--thinking-level is only supported with --llm-provider gemini.")
    if args.llm_provider == "gemini" and args.reasoning_effort is not None:
        raise ValueError("--reasoning-effort is only supported with --llm-provider openai.")


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


def load_activation_rows(path):
    rows = torch.load(path, map_location="cpu")
    if not isinstance(rows, list):
        raise TypeError(f"{path} must contain a list of activation rows.")
    if len(rows) == 0:
        raise ValueError(f"{path} contains no activation rows.")

    features = torch.stack([row["features"].detach().float().cpu() for row in rows])
    image_names = [str(row["image_name"]) for row in rows]
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    return features, image_names, labels


def get_class_names(dataset):
    return [class_name for class_name in get_classes(dataset) if class_name]


def load_autoencoder(sae_dir, device):
    checkpoint_paths = sorted(Path(sae_dir).glob("*.pt"))
    if len(checkpoint_paths) != 1:
        raise ValueError(f"{sae_dir} must contain exactly one SAE checkpoint .pt file.")
    checkpoint_path = checkpoint_paths[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["checkpoint_path"] = str(checkpoint_path)
    autoencoder = make_autoencoder(
        int(checkpoint["input_dim"]),
        int(checkpoint["latent_dim"]),
        device,
    )
    autoencoder.load_state_dict(checkpoint["state_dict"])
    autoencoder.eval()
    return autoencoder, checkpoint


def load_kept_indices(sae_dir):
    path = Path(sae_dir) / "pruning" / "kept_indices.pt"
    kept_indices = torch.load(path, map_location="cpu").long()
    if kept_indices.ndim != 1:
        raise ValueError(f"{path} must contain a 1D tensor of kept SAE unit indices.")
    return kept_indices


def get_nested_module(model, layer_name):
    module = model
    for part in layer_name.split("."):
        module = getattr(module, part)
    return module


def get_parent_module(model, layer_name):
    parts = layer_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def get_feature_map_module(model, feature_layer):
    if feature_layer in {"layer1", "layer2", "layer3", "layer4"}:
        stage_number = int(feature_layer[-1])
        modules = [model.conv1, model.bn1, model.relu, model.maxpool]
        modules.extend(getattr(model, f"layer{index}") for index in range(1, stage_number + 1))
        return torch.nn.Sequential(*modules)

    parent, final_name = get_parent_module(model, feature_layer)
    if "pool" in final_name.lower() and isinstance(parent, torch.nn.Sequential):
        return torch.nn.Sequential(*list(parent.children())[:-1])
    return get_nested_module(model, feature_layer)


def load_image(dataset_name, split, image_name):
    return Image.open(get_image_path(dataset_name, split, image_name)).convert("RGB")


def select_top_indices_for_units(
    autoencoder,
    features,
    kept_indices,
    batch_size,
    top_k,
    device,
):
    n_units = int(kept_indices.numel())
    top_values = torch.full((n_units, top_k), -float("inf"))
    top_indices = torch.full((n_units, top_k), -1, dtype=torch.long)
    kept_indices_device = kept_indices.to(device)
    all_activations = []

    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            batch = features[start:end].to(device)
            learned, _ = forward_sae(autoencoder, batch)
            batch_activations = learned[:, kept_indices_device].detach().cpu()
            all_activations.append(batch_activations)
            activations = batch_activations.T
            batch_indices = torch.arange(start, end, dtype=torch.long).expand(n_units, -1)

            candidate_values = torch.cat([top_values, activations], dim=1)
            candidate_indices = torch.cat([top_indices, batch_indices], dim=1)
            top_values, top_positions = torch.topk(candidate_values, k=top_k, dim=1, largest=True)
            top_indices = candidate_indices.gather(1, top_positions)

            del batch, learned, batch_activations, activations

    return top_indices, top_values, torch.cat(all_activations, dim=0)


def select_contrastive_indices_for_unit(
    features,
    activations,
    activation_column,
    concept_id,
    active_indices,
    seed,
    device,
):
    zero_indices = torch.where(activations[:, activation_column] <= 0)[0]
    if zero_indices.numel() < 10:
        raise ValueError(f"Not enough zero-activation contrastive images for SAE unit {concept_id}.")

    centroid = features[active_indices].mean(dim=0).to(device)
    top_values = torch.full((5,), -float("inf"))
    top_indices = torch.full((5,), -1, dtype=torch.long)

    with torch.no_grad():
        for start in range(0, zero_indices.numel(), 8192):
            batch_indices = zero_indices[start:start + 8192]
            batch_features = features[batch_indices].to(device)
            similarities = F.cosine_similarity(centroid.unsqueeze(0), batch_features, dim=1).cpu()

            candidate_values = torch.cat([top_values, similarities], dim=0)
            candidate_indices = torch.cat([top_indices, batch_indices.cpu()], dim=0)
            top_values, top_positions = torch.topk(candidate_values, k=5, largest=True)
            top_indices = candidate_indices[top_positions]

            del batch_features, similarities

    similar_indices = top_indices.tolist()
    similar_set = set(similar_indices)
    remaining_indices = [int(index) for index in zero_indices.tolist() if int(index) not in similar_set]
    rng = random.Random(int(seed) + int(concept_id))
    random_indices = rng.sample(remaining_indices, 5)

    selected_indices = similar_indices + random_indices
    rng.shuffle(selected_indices)
    return torch.tensor(selected_indices, dtype=torch.long)


def decoder_vector_for_unit(autoencoder, unit, input_dim, device):
    weight = autoencoder.decoder.weight.detach()
    if weight.shape[0] == input_dim:
        vector = weight[:, int(unit)]
    elif weight.shape[1] == input_dim:
        vector = weight[int(unit)]
    else:
        raise ValueError(
            f"Cannot read decoder vector from decoder weight with shape {tuple(weight.shape)}."
        )
    return vector.to(device)


def compute_concept_map(image, model_fmaps, preprocess, concept_vector, device):
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        fmap = model_fmaps(image_tensor)
        if isinstance(fmap, (tuple, list)):
            fmap = fmap[0]
        fmap = fmap.squeeze(0)
        if fmap.ndim != 3:
            raise ValueError(f"Expected 3D feature map, got shape {tuple(fmap.shape)}.")

        concept_map = torch.sum(fmap * concept_vector.reshape(-1, 1, 1), dim=0)
        concept_map = torch.relu(concept_map)
    return concept_map


def compute_heatmap_threshold(top_images, model_fmaps, preprocess, concept_vector, device):
    max_values = []
    for image in top_images:
        concept_map = compute_concept_map(image, model_fmaps, preprocess, concept_vector, device)
        max_values.append(float(concept_map.max().detach().cpu()))
    return 0.5 * float(np.mean(max_values))


def render_heatmap_image(
    image,
    model_fmaps,
    preprocess,
    llm_image_transform,
    concept_vector,
    heatmap_threshold,
    device,
):
    concept_map = compute_concept_map(image, model_fmaps, preprocess, concept_vector, device)
    concept_map_max = concept_map.max() + 1e-8
    threshold = heatmap_threshold / concept_map_max
    concept_map = concept_map / concept_map_max

    llm_image = llm_image_transform(image)
    image_array = np.array(llm_image)
    image_height, image_width = image_array.shape[:2]
    concept_map_resized = F.interpolate(
        concept_map.unsqueeze(0).unsqueeze(0),
        size=(image_height, image_width),
        mode="bilinear",
    )
    mask = torch.where(concept_map_resized[0][0] >= threshold, 1, 0.2).cpu().numpy()
    mask_3d = np.repeat(mask, 3).reshape(list(mask.shape) + [-1])
    masked_image = image_array * mask_3d

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(llm_image)
    ax.imshow(masked_image / 255, alpha=0.7)
    ax.axis("off")

    buffer = io.BytesIO()
    plt.savefig(buffer, format="jpeg", bbox_inches="tight", pad_inches=0)
    buffer.seek(0)
    rendered = Image.open(buffer).convert("RGB")
    plt.close(fig)
    return rendered


def read_existing_names(path):
    if not path.exists():
        return {}

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return {
            int(row["unit"]): row["concept_name"]
            for row in reader
            if row.get("unit") not in (None, "")
        }


def save_names(path, names):
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["unit", "concept_name"])
        writer.writeheader()
        for unit in sorted(names):
            writer.writerow({"unit": unit, "concept_name": names[unit]})
    tmp_path.replace(path)


def save_run_info(path, config, args, checkpoint, activation_dir):
    run_info = {
        "dataset": config["dataset"],
        "config": args.config,
        "sae_dir": args.sae_dir,
        "activation_dir": str(activation_dir),
        "sae_model": checkpoint["checkpoint_path"],
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "save_images": bool(args.save_images),
        "system_prompt": get_system_prompt(config),
        "prompt_before_images": build_prompt_before_images(),
        "prompt_after_images": build_prompt_after_images("{classes_found}"),
    }
    if args.llm_provider == "openai":
        run_info["reasoning_effort"] = args.reasoning_effort
        run_info["image_detail"] = openai_image_detail(args.llm_model)
    else:
        run_info["thinking_level"] = args.thinking_level
    path.write_text(json.dumps(run_info, indent=2))


def make_unit_examples(
    dataset_name,
    split,
    image_names,
    active_indices,
    active_values,
    contrastive_indices,
    autoencoder,
    model_fmaps,
    preprocess,
    llm_image_transform,
    unit,
    input_dim,
    device,
):
    concept_vector = decoder_vector_for_unit(autoencoder, unit, input_dim, device)
    active_raw_images = [
        load_image(dataset_name, split, image_names[int(index)])
        for index in active_indices.tolist()
    ]
    heatmap_threshold = compute_heatmap_threshold(
        active_raw_images,
        model_fmaps,
        preprocess,
        concept_vector,
        device,
    )

    examples = []
    for image, activation in zip(active_raw_images, active_values.tolist()):
        examples.append(
            {
                "activation": float(activation),
                "image": render_heatmap_image(
                    image,
                    model_fmaps,
                    preprocess,
                    llm_image_transform,
                    concept_vector,
                    heatmap_threshold,
                    device,
                ),
            }
        )

    for index in contrastive_indices.tolist():
        image = load_image(dataset_name, split, image_names[int(index)])
        examples.append(
            {
                "activation": 0.0,
                "image": render_heatmap_image(
                    image,
                    model_fmaps,
                    preprocess,
                    llm_image_transform,
                    concept_vector,
                    heatmap_threshold,
                    device,
                ),
            }
        )

    return examples


def save_unit_images(output_dir, unit, examples):
    images_dir = output_dir / "images" / f"unit_{int(unit):06d}"
    images_dir.mkdir(parents=True, exist_ok=True)

    for index, example in enumerate(examples, start=1):
        image_path = images_dir / f"image_{index:02d}.png"
        if not image_path.exists():
            example["image"].save(image_path)


def image_to_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def image_to_jpeg_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def read_keys_file(path):
    keys = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Expected KEY=value format in {path}: {line}")
        key, value = line.split("=", 1)
        keys[key.strip()] = value.strip().strip('"').strip("'")
    return keys


def resolve_api_key(provider="openai"):
    key_path = Path("keys.env")
    if not key_path.exists():
        raise FileNotFoundError("Expected LLM API keys in keys.env.")
    keys = read_keys_file(key_path)
    if provider == "openai":
        key_name = "OPENAI_API_KEY"
    elif provider == "gemini":
        key_name = "GEMINI_API_KEY"
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    api_key = keys.get(key_name)
    if api_key in (None, "", "put key here"):
        raise ValueError(f"keys.env must contain {key_name}=<real key>.")
    return api_key


def get_system_prompt(config):
    return f"You are an expert in {get_dataset_domain(config['dataset'])}."


def build_prompt_before_images():
    return (
        "We're studying a neuron in a neural network. The neuron responds to a specific visual concept.\n"
        "You will be shown a series of images with their corresponding activation value for that neuron.\n"
        "The highly activating regions are highlighted. The rest of the image is intentionally darkened.\n"
        "Your task is to identify the common visual concept in the highlighted regions of the high-activation images that is absent from the zero-activation images.\n"
        "This concept should be present in every image with positive activation but not in those with zero activation.\n"
    )


def build_prompt_after_images(classes_found):
    return (
        "Now that you've analyzed the images, provide just a concept name, phrased as concisely as possible while keeping necessary distinguishing details.\n"
        "Name the visual attribute or combination of attributes that clearly appears to cause the activation, rather than describing the full images or including weak or incidental correlations. Avoid vague names like 'distinctive pattern'.\n"
        "Return a JSON object with one key, concept_name.\n"
        f"Consider that the neuron mostly activates with images of the following classes: {classes_found}. "
        "The concept name must not be equal to any of these classes.\n"
        "The darkening is only a visualization aid, so do not treat its shape as part of the concept or darkened content as evidence for activation.\n"
        "Use the bright regions in high-activation images to identify the concept, and use the zero-activation images to test whether that same concept is selective.\n"
    )


def parse_concept_name_json(content):
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise InvalidLLMOutput(f"Could not parse structured concept name output: {content}") from exc

    concept_name = parsed.get("concept_name")
    if not isinstance(concept_name, str) or not concept_name.strip():
        raise InvalidLLMOutput(f"Structured output must contain a non-empty concept_name string: {content}")
    if "\n" in concept_name:
        raise InvalidLLMOutput(f"Concept name must be a single line: {content}")
    return concept_name.lower().replace('"', "").strip()


CONCEPT_NAME_SCHEMA = {
    "type": "object",
    "properties": {
        "concept_name": {"type": "string"},
    },
    "required": ["concept_name"],
    "additionalProperties": False,
}


GEMINI_CONCEPT_NAME_SCHEMA = {
    "type": "object",
    "properties": {
        "concept_name": {"type": "string"},
    },
    "required": ["concept_name"],
}


def call_openai_namer(args, config, examples, classes_found):
    image_detail = openai_image_detail(args.llm_model)
    content = [{"type": "input_text", "text": build_prompt_before_images()}]
    for index, example in enumerate(examples, start=1):
        content.append({"type": "input_text", "text": f"Image {index}, Activation: {example['activation']:.2f}"})
        content.append({"type": "input_image", "image_url": image_to_data_url(example["image"]), "detail": image_detail})
    content.append({"type": "input_text", "text": build_prompt_after_images(classes_found)})

    client = OpenAI(api_key=resolve_api_key("openai"))
    request = {
        "model": args.llm_model,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "concept_name_output",
                "strict": True,
                "schema": CONCEPT_NAME_SCHEMA,
            }
        },
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": get_system_prompt(config)}],
            },
            {
                "role": "user",
                "content": content,
            }
        ],
    }
    if args.reasoning_effort is not None:
        request["reasoning"] = {"effort": args.reasoning_effort}
    else:
        request["temperature"] = 0

    response = client.responses.create(**request)
    return parse_concept_name_json(response.output_text)


def call_gemini_namer(args, config, examples, classes_found):
    parts = [types.Part.from_text(text=build_prompt_before_images())]
    for index, example in enumerate(examples, start=1):
        parts.append(types.Part.from_text(text=f"Image {index}, Activation: {example['activation']:.2f}"))
        parts.append(types.Part.from_bytes(data=image_to_jpeg_bytes(example["image"]), mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=build_prompt_after_images(classes_found)))

    client = genai.Client(api_key=resolve_api_key("gemini"))
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
            response_schema=GEMINI_CONCEPT_NAME_SCHEMA,
        ),
    )
    return parse_concept_name_json(response.text)


def call_namer(args, config, examples, classes_found):
    if args.llm_provider == "openai":
        return call_openai_namer(args, config, examples, classes_found)
    if args.llm_provider == "gemini":
        return call_gemini_namer(args, config, examples, classes_found)
    raise ValueError(f"Unsupported LLM provider: {args.llm_provider}")


def call_namer_with_retry(args, config, examples, classes_found):
    try:
        return call_namer(args, config, examples, classes_found)
    except InvalidLLMOutput as first_error:
        print(f"Invalid LLM output; retrying once. Error: {first_error}")
        try:
            return call_namer(args, config, examples, classes_found)
        except InvalidLLMOutput as second_error:
            raise InvalidLLMOutput(
                f"LLM output was invalid after two attempts. Last error: {second_error}"
            ) from second_error


def make_concept_run_name(config, args):
    model_name = f"{args.llm_provider}_{args.llm_model}".replace(".", "_").replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"concepts_{config['dataset']}_{model_name}_{timestamp}"


def prepare_output_dir(config, args):
    if args.output_dir is not None:
        return Path(args.output_dir)
    return Path("outputs") / "concepts" / config["dataset"] / make_concept_run_name(config, args)


def name_concepts(config, args, activation_dir, device):
    output_dir = prepare_output_dir(config, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    names_path = output_dir / "concept_names.csv"

    activation_path = Path(activation_dir) / "activations_train.pt"
    features, image_names, labels = load_activation_rows(activation_path)
    autoencoder, checkpoint = load_autoencoder(args.sae_dir, device)
    kept_indices = load_kept_indices(args.sae_dir)
    input_dim = int(checkpoint["input_dim"])
    batch_size = int(config["sae_batch_size"])
    requested_units = parse_units(args.units)
    if requested_units is not None:
        available_units = set(range(int(kept_indices.numel())))
        missing_units = sorted(requested_units - available_units)
        if missing_units:
            raise ValueError(f"Requested units are outside the post-pruning range: {missing_units}")
        target_unit_ids = sorted(requested_units)
    else:
        target_unit_ids = list(range(int(kept_indices.numel())))
    target_kept_indices = kept_indices[target_unit_ids]

    save_run_info(output_dir / "run_info.json", config, args, checkpoint, activation_dir)

    print(f"Loaded {features.shape[0]} train activations with dimension {features.shape[1]}.")
    if requested_units is None:
        print(f"Naming {kept_indices.numel()} kept SAE units.")
    else:
        print(f"Naming {len(requested_units)} selected SAE units.")
    print("Selecting images for requested units.")
    top_indices, top_values, activations = select_top_indices_for_units(
        autoencoder,
        features,
        target_kept_indices,
        batch_size,
        10,
        device,
    )

    target_model, preprocess = get_target_model(config["backbone"], device)
    target_model.eval()
    model_fmaps = get_feature_map_module(target_model, config["feature_layer"]).to(device).eval()
    llm_image_transform = get_llm_image_transform(config["backbone"])
    class_names = get_class_names(config["dataset"])

    names = read_existing_names(names_path)
    for target_position, concept_id in enumerate(target_unit_ids):
        unit_tensor = kept_indices[concept_id]
        original_unit = int(unit_tensor.item())
        if concept_id in names:
            print(f"Skipping unit {concept_id} (already named).")
            continue

        contrastive_indices = select_contrastive_indices_for_unit(
            features,
            activations,
            target_position,
            concept_id,
            top_indices[target_position],
            config["seed"],
            device,
        )
        examples = make_unit_examples(
            config["dataset"],
            "train",
            image_names,
            top_indices[target_position],
            top_values[target_position],
            contrastive_indices,
            autoencoder,
            model_fmaps,
            preprocess,
            llm_image_transform,
            original_unit,
            input_dim,
            device,
        )

        if args.save_images:
            save_unit_images(output_dir, concept_id, examples)

        classes_found = sorted({class_names[int(labels[int(index)].item())] for index in top_indices[target_position].tolist()})
        names[concept_id] = call_namer_with_retry(args, config, examples, classes_found)
        save_names(names_path, names)
        print(f"Named unit {concept_id}: {names[concept_id]}")

    print(f"Saved concept names to {names_path}")


def main():
    args = parse_args()
    validate_llm_options(args)
    config = json.loads(Path(args.config).read_text())
    sae_run_info = load_run_info(args.sae_dir)
    activation_dir = require_run_info_path(sae_run_info, "activation_dir", args.sae_dir)
    device = torch.device(args.device or config.get("device", "cuda"))
    name_concepts(config, args, activation_dir, device)


if __name__ == "__main__":
    main()
