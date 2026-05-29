# fMRI BOLD Signal Classification from Visual Stimuli

This project investigates the decoding of visual stimulus categories from fMRI BOLD signals recorded while subjects viewed natural images. The goal is to classify brain activity patterns into four semantic categories: **object**, **animal or human**, **plant or food**, and **scene**.

---

## Datasets

| Dataset | Description |
|---|---|
| **BOLD5000** | 5000 natural images from COCO, ImageNet, and SUN, used as visual stimuli |
| **ds001499** (OpenNeuro) | fMRI data from 4 subjects across multiple sessions (15 for subjects 1–3, 10 for subject 4), 37 image trials per run |

---

## Pipeline Overview

### 1. Label Mapping

The three source datasets required different strategies to be mapped to the four final labels:

- **ImageNet** — categories generalized via WordNet hypernym hierarchy (level 6), then assigned using SentenceTransformer (`all-MiniLM-L6-v2`) cosine similarity
- **SUN** — all categories assigned to the *scene* label due to their scene-only nature
- **COCO** — excluded due to its multi-label structure being incompatible with single-label classification

### 2. Voxel Response Extraction

fMRI runs were processed using a **General Linear Model (GLM)** with a **Least-Squares-Separate (LSS)** approach, estimating one beta map per trial to isolate the brain response to each individual stimulus.

### 3. Visual Encoding

Two visual encoding strategies were compared:

| Model | Pretraining | Feature Dim | Epochs | Test Accuracy | Test F1 |
|---|---|---|---|---|---|
| Custom CNN | None (from scratch) | 32-D | 100 | 0.954 | 0.948 |
| Fine-tuned ResNet-50 | ImageNet | 2048-D | 40 | 0.950 | 0.942 |

**ResNet-50 was selected** as the final feature extractor for its higher-dimensional, semantically richer representations and alignment with the reference paper methodology.

### 4. Voxel Selection

For each Region of Interest (ROI), voxels were selected via **Ridge Regression** encoding models, ranking them by Pearson correlation between predicted and true responses:

- **BRNN input**: top-100 voxels per ROI pair
- **Multibranch GNN input**: top-51 voxels per single ROI

**ROIs used**: EarlyVis, LOC, OPA, PPA, RSC (both hemispheres)

---

## Decoding Models

### MLP Baseline
Whole-brain LSS voxel features projected to 1024-D via PCA. Grid search over 288 configurations.

---

### BRNN (Bidirectional RNN with LSTM)
ROI pairs treated as time steps (sequence length = 5, 100 voxels per step), capturing both bottom-up and top-down interactions across the visual hierarchy. Grid search over 64 configurations.

---

### Multibranch GNN
Two parallel GAT-based branches, one per hemisphere, each processing 5 ROI nodes (51 voxels each). Hemisphere embeddings are concatenated and fed to a classifier.

---


## Challenges

- Limited dataset size causing generalization issues
- Class imbalance across the four semantic labels
- Complexity of the voxel extraction strategy
- Assumptions required on ROI ordering for the BRNN
- Use of paired ROIs introduces structural constraints

---

## Future Directions

- Vary top-k voxel selection and compare performance
- Explore alternative voxel extraction strategies
- Use lower-dimensional visual features instead of 2048-D ResNet embeddings

---

## References

- Chang et al., *BOLD5000, a public fMRI dataset while viewing 5000 visual images*, Scientific Data, 6(49), 2019
- Qiao et al., *Category Decoding of Visual Stimuli From Human Brain Activity Using a Bidirectional Recurrent Neural Network*, Frontiers in Neuroscience, 13:692, 2019. doi:10.3389/fnins.2019.00692
- Miyawaki et al., *Visual image reconstruction from human brain activity using a combination of multiscale local image decoders*, Neuron, 60(5), 915–929, 2008. doi:10.1016/j.neuron.2008.11.004
- Sentence-Transformers Team. [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

*MSc Artificial Intelligence for Science and Technology — Università degli Studi di Milano-Bicocca*  
*Camilla Tomasoni*
