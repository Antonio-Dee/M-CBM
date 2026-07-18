import argparse
import csv
import json
from pathlib import Path

import numpy as np
from google import genai
from openai import OpenAI
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from mcbm.data.utils import get_dataset_domain


DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"


def parse_args():
    parser = argparse.ArgumentParser(description="Merge similar concept names with text embeddings.")
    parser.add_argument("--config", required=True, help="Path to dataset config JSON.")
    parser.add_argument("--concept-dir", required=True, help="Concept naming run directory.")
    parser.add_argument("--output-dir", help="Directory for merged concept outputs.")
    parser.add_argument("--embedding-provider", choices=["openai", "gemini"], default="openai")
    parser.add_argument("--embedding-model", default=DEFAULT_OPENAI_EMBEDDING_MODEL)
    return parser.parse_args()


def load_run_info(run_dir):
    path = Path(run_dir) / "run_info.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return json.loads(path.read_text())


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


def resolve_api_key(provider):
    key_path = Path("keys.env")
    if not key_path.exists():
        raise FileNotFoundError("Expected LLM API keys in keys.env.")
    keys = read_keys_file(key_path)
    if provider == "openai":
        key_name = "OPENAI_API_KEY"
    elif provider == "gemini":
        key_name = "GEMINI_API_KEY"
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

    api_key = keys.get(key_name)
    if api_key in (None, "", "put key here"):
        raise ValueError(f"keys.env must contain {key_name}=<real key>.")
    return api_key


def read_concepts(path):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if not {"unit", "concept_name"}.issubset(fieldnames):
            raise ValueError("Concept CSV must have unit and concept_name columns.")

        rows = []
        for row in reader:
            concept_name = row["concept_name"].strip()
            if concept_name:
                rows.append({"unit": int(row["unit"]), "concept_name": concept_name})

    if not rows:
        raise ValueError(f"{path} contains no concept names.")
    return rows


def write_merged_concepts(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["unit", "concept_name", "merged_from"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_merged_output(output_dir, concept_dir, run_info, merged):
    output_path = output_dir / "concept_names_merged.csv"
    write_merged_concepts(output_path, merged)
    run_info["merged_concepts"] = str(output_path)
    if output_dir != concept_dir:
        run_info["concept_dir"] = str(concept_dir)
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"Saved {len(merged)} merged concept names to {output_path}.")


def merge_identical_names(rows):
    concept_to_neurons = {}
    for row in rows:
        concept_to_neurons.setdefault(row["concept_name"], []).append(int(row["unit"]))

    merged_rows = []
    for new_id, (name, neurons) in enumerate(concept_to_neurons.items()):
        merged_rows.append(
            {
                "unit": new_id,
                "concept_name": name,
                "merged_from": ", ".join(map(str, sorted(neurons))),
            }
        )
    return merged_rows


def apply_context(concept, domain):
    return f"This is a visual concept in the context of {domain}: '{concept}'"


def get_openai_embeddings(concepts, model):
    client = OpenAI(api_key=resolve_api_key("openai"))
    embeddings = []
    for concept in concepts:
        response = client.embeddings.create(model=model, input=concept)
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings, dtype=np.float32)


def get_gemini_embeddings(concepts, model):
    client = genai.Client(api_key=resolve_api_key("gemini"))
    embeddings = []
    for concept in concepts:
        response = client.models.embed_content(model=model, contents=concept)
        embeddings.append(response.embeddings[0].values)
    return np.array(embeddings, dtype=np.float32)


def get_embeddings(concepts, provider, model):
    if provider == "openai":
        return get_openai_embeddings(concepts, model)
    if provider == "gemini":
        if model == DEFAULT_OPENAI_EMBEDDING_MODEL:
            raise ValueError("Specify --embedding-model when using --embedding-provider gemini.")
        return get_gemini_embeddings(concepts, model)
    raise ValueError(f"Unsupported embedding provider: {provider}")


def representative_by_average_similarity(intermediate, embeddings, cluster_indices):
    cluster_embeddings = embeddings[cluster_indices]
    similarities = cosine_similarity(cluster_embeddings)
    best_idx = cluster_indices[int(np.argmax(similarities.mean(axis=1)))]
    return intermediate[best_idx]["concept_name"]


def build_merged_output(intermediate, embeddings, labels):
    rows = []
    for new_id, cluster_id in enumerate(sorted(set(labels))):
        cluster_indices = [
            index
            for index, label in enumerate(labels)
            if label == cluster_id
        ]
        merged_neurons = []
        for index in cluster_indices:
            merged = intermediate[index]["merged_from"]
            merged_neurons.extend(int(value.strip()) for value in merged.split(","))
        merged_neurons = sorted(set(merged_neurons))

        representative = representative_by_average_similarity(intermediate, embeddings, cluster_indices)

        rows.append(
            {
                "unit": new_id,
                "concept_name": representative,
                "merged_from": ", ".join(map(str, merged_neurons)),
            }
        )
    return rows


def get_config_float(config, key):
    if key not in config:
        raise KeyError(f"Config is missing required key: {key}")
    return float(config[key])


def merge_concepts(config, concept_dir, output_dir, embedding_provider, embedding_model):
    dataset = config["dataset"]
    concept_dir = Path(concept_dir)
    concepts_path = concept_dir / "concept_names.csv"
    output_dir = Path(output_dir) if output_dir is not None else concept_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_info = load_run_info(concept_dir)

    concepts = read_concepts(concepts_path)
    intermediate = merge_identical_names(concepts)
    domain = get_dataset_domain(dataset)
    concepts_with_context = [
        apply_context(row["concept_name"], domain)
        for row in intermediate
    ]

    print(f"Loaded {len(concepts)} concept names.")
    print(f"After exact-name merging: {len(intermediate)} concept names.")
    print(f"Embedding provider: {embedding_provider}")
    print(f"Embedding model: {embedding_model}")

    if len(intermediate) == 1:
        raise ValueError("Only one concept name remains after exact-name merging.")

    embeddings = get_embeddings(concepts_with_context, embedding_provider, embedding_model)
    threshold = get_config_float(config, "merge_similarity_threshold")
    print(f"Similarity threshold: {threshold}")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="complete",
        distance_threshold=1 - threshold,
    )
    labels = clustering.fit_predict(embeddings)

    merged = build_merged_output(intermediate, embeddings, labels)
    run_info["embedding_provider"] = embedding_provider
    run_info["embedding_model"] = embedding_model
    save_merged_output(output_dir, concept_dir, run_info, merged)


def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    merge_concepts(
        config,
        args.concept_dir,
        args.output_dir,
        args.embedding_provider,
        args.embedding_model,
    )


if __name__ == "__main__":
    main()
