# Self-Supervised Collider Foundation Model

A self-supervised Transformer-based foundation-model prototype for learning representations of particle-collision events from collider data.

This project investigates whether a Transformer can learn useful particle-level and event-level representations **without requiring explicit event-class labels during pre-training**. The model is trained using a masked-particle reconstruction objective, inspired by masked-language-modeling approaches in natural language processing.

The project is designed as a research prototype for applying modern self-supervised learning techniques to High Energy Physics (HEP).

---

## Overview

Particle-collision events contain variable numbers of reconstructed particles, with each particle carrying kinematic and particle-identification information.

Instead of training a model directly for a single supervised classification task, this project explores a more general approach:

> **Can a model learn a useful representation of collider events by reconstructing information that has been deliberately masked from the input?**

For each event, the model receives a sequence of particles. Each particle is represented using:

* transverse momentum (`pT`)
* pseudorapidity (`eta`)
* azimuthal angle represented by `sin(phi)` and `cos(phi)`
* energy
* mass
* electric charge
* particle identity through a PDG-ID token

During self-supervised training, a fraction of particles is masked. The Transformer must reconstruct:

1. the masked particle's continuous features
2. the masked particle's PDG identity

The resulting Transformer representation is then studied as an event-level latent representation.

---

# Scientific Motivation

Modern collider experiments produce extremely large and complex datasets. Traditional machine-learning approaches often train a model for a specific downstream task, such as:

* event classification
* particle identification
* anomaly detection
* signal/background discrimination

Such approaches can require large amounts of labelled training data and may produce representations optimized only for the task for which the model was trained.

Self-supervised learning provides an alternative.

The central idea of this project is to first learn a general representation of collider events from the structure of the data itself. A model can then potentially be adapted to different downstream physics tasks.

This project therefore explores a collider analogue of the general strategy used by foundation models:

**particle data → self-supervised pre-training → learned representation → downstream physics applications**

---

# Model Architecture

The model is a Transformer encoder operating on sequences of particles.

### Particle representation

Each particle contains seven continuous features:

| Feature      | Description                      |
| ------------ | -------------------------------- |
| `log_pt`     | Logarithm of transverse momentum |
| `eta`        | Pseudorapidity                   |
| `sin_phi`    | Sine of azimuthal angle          |
| `cos_phi`    | Cosine of azimuthal angle        |
| `log_energy` | Logarithm of energy              |
| `log_mass`   | Logarithm of mass                |
| `charge`     | Electric charge                  |

Particle identity is represented separately using a token corresponding to the particle's PDG ID.

### Particle embedding

The continuous features are passed through a small neural network:

```text
7 continuous features
        │
        ▼
Linear(7 → 128)
        │
       GELU
        │
        ▼
Linear(128 → 128)
```

The PDG token is mapped into a 128-dimensional embedding.

The continuous-feature embedding and PDG embedding are then added together to form the particle representation.

### Transformer

The particle representations are processed by a Transformer encoder with:

* embedding dimension: `128`
* attention heads: `8`
* Transformer layers: `4`
* feed-forward dimension: `512`
* dropout: `0.1`
* GELU activation

The model accepts variable-length particle sequences using padding masks.

---

# Self-Supervised Objective

Approximately 40% of the valid particles in each event are randomly masked.

The model receives the remaining information and attempts to reconstruct the masked particles.

Two prediction heads are used.

### 1. PDG reconstruction

The model predicts the PDG-token identity of each masked particle.

The PDG vocabulary contains 31 particle categories plus a padding token.

### 2. Continuous-feature reconstruction

The model predicts the seven continuous particle features:

```text
log_pt
eta
sin_phi
cos_phi
log_energy
log_mass
charge
```

This creates a multi-task self-supervised objective:

```text
                   Collider event
                         │
                         ▼
              Particle representations
                         │
                  Random masking
                         │
                         ▼
                Transformer Encoder
                    /           \
                   /             \
                  ▼               ▼
          PDG prediction    Feature prediction
```

---

# Dataset

The processed dataset contains:

```text
Number of events:       45,830
Training events:        36,664
Validation events:       4,583
Test events:             4,583
PDG vocabulary size:        32
```

The dataset is stored in:

```text
processed_collider_events.pt
```

The processed file contains:

```text
continuous_list
pdg_id_list
train_idx
val_idx
test_idx
feature_mean
feature_std
pdg_to_id
PAD_ID
VOCAB_SIZE
```

The particle features are standardized using the training-data statistics.

The PDG IDs are converted into integer token IDs for use by the neural network.

---

# Data Representation

Each event is represented as a variable-length sequence.

For example:

```text
Event
│
├── Particle 1
│   ├── log_pt
│   ├── eta
│   ├── sin_phi
│   ├── cos_phi
│   ├── log_energy
│   ├── log_mass
│   └── charge
│
├── Particle 2
│   ├── ...
│
├── Particle 3
│   ├── ...
│
└── ...
```

Since collider events contain different numbers of particles, sequences are padded within each batch.

A padding mask prevents the Transformer from treating padding positions as physical particles.

---

# Repository Structure

The repository is organized around a sequence of notebooks corresponding to the different stages of the project.

```text
self-supervised-collider-foundation-model/
│
├── README.md
│
├── Notebook1_*.ipynb
├── Notebook2_*.ipynb
├── Notebook3_*.ipynb
├── Notebook4_*.ipynb
├── Notebook5_*.ipynb
│
├── processed_collider_events.pt
│
├── checkpoints/
│   └── best_model.pt
│
└── ...
```

The exact notebook filenames may differ. Their conceptual workflow is:

```text
Raw collider data
       │
       ▼
Data preprocessing
       │
       ▼
Particle/event representation
       │
       ▼
Train/validation/test split
       │
       ▼
Self-supervised Transformer training
       │
       ▼
Best-model checkpoint
       │
       ▼
Model evaluation
       │
       ├── PDG reconstruction
       │
       ├── Continuous-feature reconstruction
       │
       ├── Event embeddings
       │
       └── Latent-space analysis
```

---

# Notebook Workflow

## Notebook 1 — Data Preparation

The first notebook prepares the collider-event data for machine learning.

Typical steps include:

* loading the collider-event data
* selecting the relevant particle information
* constructing particle-level features
* calculating transformed quantities such as logarithmic momentum/energy/mass
* representing the azimuthal angle using sine and cosine
* assigning PDG tokens
* constructing event sequences

The result is a structured representation suitable for Transformer-based learning.

---

## Notebook 2 — Dataset Processing

The second stage prepares the final machine-learning dataset.

It includes:

* feature normalization
* construction of the PDG vocabulary
* assignment of token IDs
* train/validation/test splitting
* storage of processed data

The resulting dataset is saved as:

```text
processed_collider_events.pt
```

The split used in the current experiment is:

```text
Train:      36,664 events
Validation:  4,583 events
Test:       4,583 events
```

---

## Notebook 3 — Model Training

The training notebook defines and trains the masked-particle Transformer.

The main architecture is:

```text
Particle features
       +
PDG embedding
       │
       ▼
Particle embedding
       │
       ▼
Transformer Encoder
       │
       ├───────────────┐
       ▼               ▼
PDG head          Feature head
       │               │
       ▼               ▼
PDG prediction    Feature reconstruction
```

The training objective is based on reconstructing masked particles.

The best checkpoint is saved as:

```text
checkpoints/best_model.pt
```

The currently selected best model was obtained at:

```text
Best epoch: 14
Validation loss: 2.51238
```

---

## Notebook 4 — Model Evaluation

The evaluation notebook loads the best trained model and evaluates its self-supervised reconstruction performance.

The evaluation includes:

* PDG reconstruction accuracy
* classification report
* confusion matrix
* continuous-feature reconstruction
* predicted-vs-true feature plots
* event-level embedding extraction
* PCA visualization
* correlation with particle multiplicity

The model is evaluated on the held-out validation and test datasets.

---

## Notebook 5 — Representation Analysis

The final notebook focuses on the learned event representations.

The Transformer produces a 128-dimensional representation for each event.

The particle-level Transformer outputs are aggregated over the valid particles to construct an event-level embedding.

For the current test set:

```text
Number of test events: 4,583
Embedding dimension:   128
```

Principal Component Analysis (PCA) is then used to visualize the learned representation in two dimensions.

The first two PCA components explain approximately:

```text
PC1: 20.8%
PC2: 18.2%

Combined: 39.0%
```

The latent representation also shows a measurable relationship with particle multiplicity.

---

# Results

## PDG Reconstruction

The model achieves approximately:

```text
Validation accuracy: 47.6%

Test accuracy:       47.9%
```

with approximately 40% of valid particles masked.

The test set contains approximately 144,000 masked-particle predictions.

The result demonstrates that the Transformer learns non-trivial information about particle identity from the surrounding event context.

However, the classification performance is strongly influenced by the highly imbalanced particle distribution.

For example, the photon token represents a very large fraction of the masked particles, and the current model predicts this dominant class very frequently.

Therefore, overall accuracy alone should **not** be interpreted as evidence of equally strong reconstruction performance for all particle species.

Future versions should include class-balanced metrics and/or a weighted reconstruction loss.

---

# Continuous-Feature Reconstruction

The model also reconstructs the continuous particle features.

Current test-set results are:

| Feature      |    MAE |   RMSE | Correlation |
| ------------ | -----: | -----: | ----------: |
| `log_pt`     | 0.7634 | 1.0019 |      0.1374 |
| `eta`        | 0.7398 | 0.9941 |      0.1521 |
| `sin_phi`    | 0.8845 | 0.9891 |      0.1565 |
| `cos_phi`    | 0.8813 | 0.9872 |      0.1611 |
| `log_energy` | 0.7821 | 1.0073 |      0.0971 |
| `log_mass`   | 0.6663 | 0.9954 |      0.0817 |
| `charge`     | 0.7271 | 0.9905 |      0.1401 |

The relatively modest correlations indicate that the current model is still at an early prototype stage and that the reconstruction task can be substantially improved.

These results are therefore treated as a baseline rather than a final physics result.

---

# Learned Event Representation

A 128-dimensional event embedding is extracted from the Transformer.

PCA is used to visualize the learned representation.

The first two principal components explain:

```text
PC1 = 0.208
PC2 = 0.182

Total = 0.390
```

or approximately 39% of the variance.

The first PCA component also shows a statistically significant correlation with particle multiplicity:

```text
Pearson r  ≈ -0.357
Spearman r ≈ -0.347
```

This indicates that the learned latent representation contains information related to the structure/complexity of collider events.

Importantly, this does not by itself demonstrate that the representation has learned a physically optimal representation. It is an initial diagnostic showing that the latent space is structured rather than completely random.

---

# Computational Environment

The current experiments were run using **CPU-only PyTorch**.

Example environment:

```text
Python
PyTorch 2.11.0
scikit-learn
SciPy
NumPy
Matplotlib
```

The model automatically selects the available device:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

For the current environment:

```text
CUDA available: False
Using device: cpu
```

A GPU is therefore **not required to reproduce the evaluation notebooks**.

GPU acceleration would be strongly recommended for larger-scale training experiments, larger models, or substantially larger datasets.

---

# Reproducibility

## 1. Clone the repository

```bash
git clone https://github.com/Shima-m83/self-supervised-collider-foundation-model.git

cd self-supervised-collider-foundation-model
```

## 2. Create a Python environment

For example:

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install torch numpy matplotlib scikit-learn scipy jupyter
```

If a `requirements.txt` file is provided in the repository, use:

```bash
pip install -r requirements.txt
```

instead.

## 4. Launch Jupyter

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

## 5. Run the notebooks in order

Run the notebooks sequentially:

```text
Notebook 1
    ↓
Notebook 2
    ↓
Notebook 3
    ↓
Notebook 4
    ↓
Notebook 5
```

The later notebooks depend on files and/or model checkpoints generated by earlier stages.

---

# Important Files

### Processed dataset

```text
processed_collider_events.pt
```

Contains the processed event sequences, train/validation/test indices, normalization parameters, and PDG vocabulary.

### Model checkpoint

```text
checkpoints/best_model.pt
```

Contains the trained Transformer model state corresponding to the best validation loss.

---

# Limitations

This repository represents a **research prototype**, not a production-ready collider foundation model.

Several aspects require further development.

### 1. Class imbalance

The PDG vocabulary is highly imbalanced.

The current overall accuracy is therefore dominated by common particle species.

Future evaluation should include:

* per-class accuracy
* macro F1
* weighted F1
* balanced accuracy
* confusion matrices normalized by class
* possibly class-weighted losses

### 2. Continuous-feature reconstruction

The current reconstruction correlations are relatively low.

Possible improvements include:

* improved feature parameterization
* alternative reconstruction losses
* feature-specific loss weighting
* larger Transformer models
* improved masking strategies
* incorporating additional particle information

### 3. Physics-aware representations

The current model treats the particle sequence using a standard Transformer architecture.

Future work could investigate:

* permutation-aware architectures
* Lorentz-equivariant networks
* particle-flow representations
* four-momentum-based representations
* graph neural networks
* physics-informed positional representations

### 4. Larger-scale pre-training

A true collider foundation model would require significantly larger datasets and potentially larger models.

The present work should therefore be considered a **proof-of-concept / baseline study**.

### 5. Downstream tasks

The most important next step is to test whether the learned representation transfers to physics tasks that were not used during self-supervised training.

Possible downstream tasks include:

* particle identification
* event classification
* signal/background discrimination
* anomaly detection
* jet classification
* new-physics searches
* event reconstruction

---

# Future Work

The project can be extended in several directions.

## Short-term

* Improve the masked-particle objective.
* Introduce class-balanced PDG losses.
* Report per-particle-species performance.
* Optimize continuous-feature reconstruction.
* Compare different masking ratios.
* Compare different Transformer sizes.
* Evaluate different event-pooling strategies.

## Medium-term

Use the learned representation for downstream supervised tasks.

For example:

```text
Self-supervised pre-training
            │
            ▼
       Event embedding
            │
     ┌──────┼────────┐
     ▼      ▼        ▼
Classification  Anomaly  Physics
                detection  task
```

This would provide a stronger test of whether the representation learned during pre-training is genuinely useful.

## Long-term

Investigate larger collider foundation models trained on diverse event samples and evaluate transfer between:

* collision energies
* processes
* detector configurations
* simulation samples
* physics tasks

---

# Why This Project Matters

The main purpose of this project is not to claim that the current model is already a complete collider foundation model.

Instead, it demonstrates an end-to-end research workflow:

```text
Collider data
     │
     ▼
Physics-motivated preprocessing
     │
     ▼
Variable-length particle sequences
     │
     ▼
Self-supervised learning
     │
     ▼
Transformer representation
     │
     ▼
Quantitative evaluation
     │
     ▼
Latent-space analysis
     │
     ▼
Potential downstream physics applications
```

The project combines concepts from:

* High Energy Physics
* collider event reconstruction
* particle physics phenomenology
* deep learning
* Transformer architectures
* self-supervised learning
* representation learning
* foundation models

---

# Technologies

The project uses:

* **Python**
* **PyTorch**
* **NumPy**
* **SciPy**
* **scikit-learn**
* **Matplotlib**
* **Jupyter**

---

# Author

**Shima**

This project was developed as an independent research project exploring the application of self-supervised learning and Transformer architectures to collider physics.

---

# Citation

If you use this repository or build upon this work, please cite the repository:

```bibtex
@software{self_supervised_collider_foundation_model,
  author = {Shima},
  title = {Self-Supervised Collider Foundation Model},
  year = {2026},
  url = {https://github.com/Shima-m83/self-supervised-collider-foundation-model}
}
```

---

# License

Add an appropriate open-source license before publishing the repository for reuse.

For example, an MIT license can be used if you want to allow broad reuse and modification of the code.

---

# Project Status

**Research prototype — actively developing**

The current version demonstrates:

* [x] Collider-event preprocessing
* [x] Particle-level feature representation
* [x] PDG tokenization
* [x] Train/validation/test splitting
* [x] Masked-particle self-supervised learning
* [x] Transformer encoder
* [x] PDG reconstruction
* [x] Continuous-feature reconstruction
* [x] Event-level embedding extraction
* [x] PCA latent-space analysis
* [x] CPU-based evaluation

Planned:

* [ ] Class-balanced reconstruction
* [ ] Improved continuous-feature reconstruction
* [ ] Larger-scale training
* [ ] Downstream classification
* [ ] Anomaly-detection study
* [ ] Comparison with alternative architectures
* [ ] Physics-specific transfer-learning experiments

---

## Summary

This repository presents a prototype self-supervised learning framework for collider events.

The model learns from masked particle information rather than requiring explicit event labels during pre-training. A Transformer encoder processes variable-length particle sequences and learns representations that can be evaluated through particle reconstruction and event-level latent-space analysis.

The current results demonstrate the feasibility of the approach while also identifying clear areas for improvement, particularly in class-balanced particle reconstruction and continuous-feature prediction.

The longer-term goal is to investigate whether self-supervised Transformer representations can serve as reusable building blocks for multiple collider-physics applications.

