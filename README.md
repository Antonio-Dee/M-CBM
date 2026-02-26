# M-CBM: Learning Concept Bottleneck Models from Mechanistic Explanations
Official repository for the paper *_Learning Concept Bottleneck Models from Mechanistic Explanations_*, Antonio De Santis, Schrasing Tong, Marco Brambilla, Lalana Kagal, ICLR 2026. [[Paper]](https://openreview.net/pdf?id=gdEWoxhb70).

🚧 **Repository status:** Code will be added soon.

## What is a Mechanistic Concept Bottleneck Model (M-CBM)?
A Mechanistic Concept Bottleneck Model (M-CBM) is an inherently interpretable model whose predictions can be traced back and explained in terms of human-understandable concepts that the model has automatically learned from the data it was trained on.

**How does this differ from prior CBMs?**
Prior CBM approaches typically require prescribing the model with concepts chosen a priori by human experts, via prompting an LLM, from open knowledge graphs, or from vision-language models such as CLIP. However, such predefined concepts may not have sufficient predictive power for the task, or may not even be observable in the data. M-CBM instead uses mechanistic interpretability to reverse-engineer human-understandable concepts already learned by a trained black-box model, and then uses these concepts to convert the black-box into a CBM that better preserves the performance of the original model.

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
