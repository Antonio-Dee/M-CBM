# M-CBM: Learning Concept Bottleneck Models from Mechanistic Explanations
Official repository for the paper *_Learning Concept Bottleneck Models from Mechanistic Explanations_*, Antonio De Santis, Schrasing Tong, Marco Brambilla, Lalana Kagal, ICLR 2026. [[Paper]](https://openreview.net/pdf?id=gdEWoxhb70).

## What is a Mechanistic Concept Bottleneck Model (M-CBM)?
A Mechanistic Concept Bottleneck Model (M-CBM) is an inherently interpretable model whose predictions can be traced back and explained in terms of human-understandable concepts that the model has automatically learned from the data it was trained on.

**How does this differ from prior CBMs?**
Prior CBM approaches typically require prescribing the model with concepts chosen a priori by human experts, via prompting an LLM, from open knowledge graphs, or from vision-language models such as CLIP. However, such predefined concepts may not have sufficient predictive power for the task, or may not even be observable in the data. M-CBM instead uses mechanistic interpretability to reverse-engineer human-understandable concepts already learned by a trained black-box model, and then uses these concepts to convert the black-box into an interpretable CBM while trying to preserve as much as possible the performance of the original model.

## How does it work?

<br>
<p align="center">
  <img src="assets/pipeline.svg" width="90%" alt="M-CBM training pipeline">
</p>
<br>

At a high level, learning an M-CBM consists of four main steps:

**1. Concept Extraction:** Train a Sparse Autoencoder (SAE) on backbone representations to decompose them into a sparse set of features (i.e., SAE neurons) that serve as candidate concepts.

**2. Concept Naming:** Use a Multimodal LLM (MLLM) to assign human-understandable names to these neurons by reverse-engineering what they respond to from activating and non-activating examples.

**3. Concept Annotation:** Use an MLLM to annotate concept presence or absence on a targeted subset of images selected using activations of SAE neurons, creating supervision for concept prediction.

**4. CBM Training:** Train a Concept Bottleneck Layer (CBL) that predicts concept presence and a Sparse Linear Layer that uses these concepts to generate the final prediction.

## Setup

Create the conda environment:

```bash
conda create -n mcbm python=3.10 -y
conda activate mcbm
```

Install the Python packages used by the current scripts:

```bash
pip install -r requirements.txt
```

The repo looks for datasets in the folder specified by the environment variable `DATASET_FOLDER`, which defaults to `datasets/`.

For the LLM steps, open `keys.env` and replace the placeholder with your API key. Use `OPENAI_API_KEY` for OpenAI models and `GEMINI_API_KEY` for Gemini models.

## Example: run the current pipeline on CUB

Run commands from the repository root.

**0. Extract activations**

```bash
python -m mcbm.scripts.extract_activations --config configs/cub.json
```

This creates an activation folder inside `activations/cub/` and saves `activations_train.pt`, `activations_val.pt`, and `activations_test.pt`.

**Optional: use precomputed annotations**

After extracting activations, you can optionally skip steps 1-3 by downloading the precomputed annotations from [here](https://dataverse.harvard.edu/previewurl.xhtml?token=c0f70db2-4bd8-4249-bee2-fc32a310e9fe) and placing them inside the `outputs/annotations/` folder. Then continue from step 4.1.

**1. Train the SAE**

Replace `<ACTIVATION_RUN_DIR>` with the activation folder created in step 0, for example `activations/cub/<ACTIVATION_RUN_DIR>`.

```bash
python -m mcbm.scripts.train_sae --config configs/cub.json --activation-dir activations/cub/<ACTIVATION_RUN_DIR>
```

This creates a SAE run folder inside `outputs/sae/cub/`.

**2.1. Name SAE units**

Replace `<SAE_RUN_DIR>` with the SAE folder created in step 1, for example `outputs/sae/cub/<SAE_RUN_DIR>`.

```bash
python -m mcbm.scripts.name_concepts --config configs/cub.json --sae-dir outputs/sae/cub/<SAE_RUN_DIR> --llm-model gpt-4.1
```

This creates a concept naming folder inside `outputs/concepts/cub/`.

For reasoning models also pass `--reasoning-effort`. For example, replace the last line with:

```bash
python -m mcbm.scripts.name_concepts --config configs/cub.json --sae-dir outputs/sae/cub/<SAE_RUN_DIR> --llm-model gpt-5.6-sol --reasoning-effort none
```

Gemini models are also supported, including models available through the free API tier:

```bash
python -m mcbm.scripts.name_concepts --config configs/cub.json --sae-dir outputs/sae/cub/<SAE_RUN_DIR> --llm-provider gemini --llm-model gemini-3.5-flash
```

**2.2. Merge similar concept names**

Replace `<CONCEPT_RUN_DIR>` with the concept naming folder created in step 2.1, for example `outputs/concepts/cub/<CONCEPT_RUN_DIR>`.

```bash
python -m mcbm.scripts.merge_concepts --config configs/cub.json --concept-dir outputs/concepts/cub/<CONCEPT_RUN_DIR>
```

To use Gemini embeddings instead:

```bash
python -m mcbm.scripts.merge_concepts --config configs/cub.json --concept-dir outputs/concepts/cub/<CONCEPT_RUN_DIR> --embedding-provider gemini --embedding-model gemini-embedding-001
```

This saves `concept_names_merged.csv` in the concept folder.

**3. Annotate concepts**

```bash
python -m mcbm.scripts.annotate_concepts --config configs/cub.json --concept-dir outputs/concepts/cub/<CONCEPT_RUN_DIR> --llm-model gpt-4.1
```

This creates a dated annotation folder inside `outputs/annotations/cub/` and saves `annotations_train.pt` and `annotations_test.pt`.

**4.1. Train the CBM**

Use the activation folder from step 0 and the annotation folder from step 3.

```bash
python -m mcbm.scripts.train_cbm --config configs/cub.json --activation-dir activations/cub/<ACTIVATION_RUN_DIR> --annotation-dir outputs/annotations/cub/<ANNOTATION_RUN_DIR>
```

This trains the concept bottleneck layer and the final sparse linear layer. It creates a CBM run folder inside `outputs/cbm/cub/`.

**4.2. Evaluate at target NCC values**

Replace `<CBM_RUN_DIR>` with the CBM run folder created in step 4.1, for example `outputs/cbm/cub/<CBM_RUN_DIR>`.

```bash
python -m mcbm.scripts.ncc_evaluation --cbm-dir outputs/cbm/cub/<CBM_RUN_DIR>
```

This saves NCC-selected final layers inside `outputs/cbm/cub/<CBM_RUN_DIR>/ncc/`, including `W_g@NCC_target=5.pt` and `b_g@NCC_target=5.pt`.

The notebook `notebooks/evaluate_trained_cbm.ipynb` can be used to inspect a trained CBM, including class-to-concept weights, top activating images, and example explanations.

## Sources

- CUB dataset: https://www.vision.caltech.edu/datasets/cub_200_2011/
- CUB pretrained model: https://github.com/osmr/pytorchcv
- ISIC 2018 dataset: https://challenge.isic-archive.com/data/#2018
- ImageNet dataset: https://www.image-net.org/download.php
- Sparse autoencoder training: https://github.com/ai-safety-foundation/sparse_autoencoder
- GLM-SAGA for sparse final layer training: https://github.com/MadryLab/glm_saga
- VLG-CBM for parts of CBM training and evaluation: https://github.com/Trustworthy-ML-Lab/VLG-CBM

## Citation
```bibtex
@inproceedings{
desantis2026learning,
title={Learning Concept Bottleneck Models from Mechanistic Explanations},
author={Antonio De Santis and Schrasing Tong and Marco Brambilla and Lalana Kagal},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=gdEWoxhb70}
}
```
