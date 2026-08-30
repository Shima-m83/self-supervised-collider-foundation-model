# Self-Supervised Collider Foundation Model

A self-supervised Transformer-based foundation-model prototype for learning representations of particle-collision events from collider data.

This project investigates whether a Transformer can learn useful particle-level and event-level representations **without requiring explicit event-class labels during pre-training**. The model is trained using a masked-particle reconstruction objective, inspired by masked-language-modeling approaches in natural language processing.

The project is designed as a research prototype for applying modern self-supervised learning techniques to High Energy Physics (HEP).

---

# Overview

Particle-collision events contain variable numbers of particles, with each particle carrying kinematic and particle-identification information.

Instead of training a model directly for a single supervised classification task, this project explores a more general approach:

> **Can a model learn a useful representation of collider events by reconstructing information that has been deliberately masked from the input?**

For each event, the model receives a sequence of particles. Each particle is represented using:

- transverse momentum, through `log_pt`
- pseudorapidity, `eta`
- azimuthal angle represented by `sin_phi` and `cos_phi`
- energy, through `log_energy`
- mass, through `log_mass`
- electric charge
- particle identity through a PDG-ID token

During self-supervised training, approximately 40% of valid particles are masked. The Transformer must reconstruct:

1. the masked particle's continuous features
2. the masked particle's PDG identity

The resulting Transformer representation is then studied as an event-level latent representation.

---

# Scientific Motivation

Modern collider experiments produce extremely large and complex datasets. Traditional machine-learning approaches often train a model for a specific downstream task, such as:

- event classification
- particle identification
- anomaly detection
- signal/background discrimination

Such approaches can require labelled training data and may produce representations optimized primarily for the specific task for which the model was trained.

Self-supervised learning provides an alternative.

The central idea of this project is to first learn a general representation of collider events from the structure of the data itself. A model can then potentially be adapted to different downstream physics tasks.

This project therefore explores a collider analogue of the general strategy used by foundation models:

```text
Particle data
     ↓
Self-supervised pre-training
     ↓
Learned representation
     ↓
Downstream physics applications
```

The central question motivating this project is closely related to representation learning in scientific data:

> **What can a model learn about particle-collision events when no event-class labels are provided during pre-training?**

---

# Model Architecture

The model is a Transformer encoder operating on variable-length sequences of particles.

## Particle Representation

Each particle contains seven continuous features:

| Feature | Description |
|---|---|
| `log_pt` | Logarithmically transformed transverse momentum |
| `eta` | Pseudorapidity |
| `sin_phi` | Sine of azimuthal angle |
| `cos_phi` | Cosine of azimuthal angle |
| `log_energy` | Logarithmically transformed energy |
| `log_mass` | Logarithmically transformed mass |
| `charge` | Electric charge |

Particle identity is represented separately using a token corresponding to the particle's PDG ID.

## Particle Embedding

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
        │
        ▼
128-dimensional feature embedding
```

The PDG token is mapped into a learned 128-dimensional embedding.

The continuous-feature embedding and PDG embedding are added together to form the particle representation.

For masked particles, the complete particle representation is replaced with a learned `[MASK]` token.

```text
Continuous features ───┐
                       ├──► Particle representation
PDG embedding ─────────┘
                       │
                       ▼
              Masked if selected
                       │
                       ▼
             Transformer Encoder
```

## Transformer

The particle representations are processed by a Transformer encoder with:

- embedding dimension: `128`
- attention heads: `8`
- Transformer layers: `4`
- feed-forward dimension: `512`
- dropout: `0.1`
- GELU activation

The model accepts variable-length particle sequences using padding masks.

The trained model contains approximately:

```text
836,391 trainable parameters
```

---

# Self-Supervised Objective

Approximately 40% of the valid particles in each event are randomly masked.

The model receives the remaining information and attempts to reconstruct the masked particles from the surrounding event context.

Two prediction heads are used.

## 1. PDG Reconstruction

The model predicts the PDG-token identity of each masked particle.

The PDG vocabulary contains:

- 31 final-state particle categories
- 1 padding token

giving a total vocabulary size of:

```text
32
```

## 2. Continuous-Feature Reconstruction

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

The total reconstruction loss is:

```text
Total loss = Feature reconstruction loss + PDG reconstruction loss
```

Continuous features are reconstructed using mean-squared error, while PDG tokens are reconstructed using cross-entropy loss.

Only masked, non-padding particles contribute to the training loss.

---

# Dataset

The project uses particle-level collider-event data stored in an EDM4HEP ROOT file.

The initial data inspection and preprocessing select generator-level final-state particles using:

```text
generatorStatus == 1
```

Particles with nonzero three-momentum are retained for the baseline dataset.

Particles within each event are sorted by descending transverse momentum to provide a deterministic sequence ordering for the Transformer prototype.

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

The continuous particle features are standardized using training-data statistics.

The PDG IDs are converted into integer token IDs for use by the neural network.

---

# Data Representation

Each collider event is represented as a variable-length sequence of particles.

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

Each particle also has a categorical PDG token.

Since collider events contain different numbers of particles, sequences are padded within each batch.

A padding mask prevents the Transformer from treating padding positions as physical particles.

This allows batches to have the form:

```text
Features:     [batch_size, sequence_length, 7]
PDG tokens:   [batch_size, sequence_length]
Padding mask: [batch_size, sequence_length]
```

---

# Repository Structure

The repository is organized around the different stages of the project workflow.

```text
fcc-ee-masked-particle-modeling/
│
├── data/
│   └── clean_data_ecm240.root
│
├── notebooks/
│   ├── 01_inspect_edm4hep.ipynb
│   ├── 02_build_dataset.ipynb
│   ├── 03_masked_particle_transformer.ipynb
│   └── 04_representation_evaluation.ipynb
│
├── figures/
│   ├── 03_PDG_Reconstruction_Loss.png
│   ├── 04_Masked_Particle_Reconstruction_eta.png
│   
│   └── ...
│
├── checkpoints/
│   ├── processed_collider_events.pt
│   └── Readme.md
│
├── README.md
└── requirements.txt

The conceptual workflow is:

```text
Raw EDM4HEP collider data
       │
       ▼
Data inspection
       │
       ▼
Particle-level preprocessing
       │
       ▼
Variable-length event representation
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

## Notebook 1 — EDM4HEP Data Inspection

**File:**

```text
01_inspect_edm4hep.ipynb
```

The first notebook inspects the EDM4HEP collider-event data.

The notebook includes:

- opening the ROOT file
- inspecting the available ROOT trees
- reading the `events` tree
- examining the particle collection
- selecting generator-level final-state particles
- studying event and particle multiplicities
- examining particle kinematics
- inspecting PDG IDs and particle distributions

This stage provides an initial understanding of the collider data before constructing the machine-learning dataset.

---

## Notebook 2 — Dataset Processing

**File:**

```text
02_build_dataset.ipynb
```

The second notebook prepares the final machine-learning dataset.

It includes:

- particle selection
- particle-level feature construction
- logarithmic transformations of momentum, energy, and mass
- representing the azimuthal angle using sine and cosine
- PDG vocabulary construction
- PDG tokenization
- feature normalization
- variable-length event sequences
- train/validation/test splitting
- storage of the processed data

The resulting dataset is saved as:

```text
processed_collider_events.pt
```

The split used in the current experiment is:

```text
Train:      36,664 events
Validation:  4,583 events
Test:        4,583 events
```

---

## Notebook 3 — Model Training

**File:**

```text
03_masked_particle_transformer.ipynb
```

The training notebook defines and trains the masked-particle Transformer.

The main architecture is:

```text
7 continuous features
        │
        ▼
Feature embedding ──────┐
                        │
PDG token               │
        │               │
        ▼               │
PDG embedding ──────────┤
                        │
                        ▼
              Particle representation
                        │
                  Mask some particles
                        │
                        ▼
               Learned [MASK] token
                        │
                        ▼
               Transformer Encoder
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
       Feature decoder      PDG classifier
              │                   │
              ▼                   ▼
       7 reconstructed       32 classes
          features
```

The notebook includes:

- loading the processed dataset
- constructing PyTorch datasets and data loaders
- padding variable-length events
- creating padding masks
- implementing random particle masking
- building particle embeddings
- defining the Transformer encoder
- defining feature and PDG reconstruction heads
- implementing the masked reconstruction loss
- optimizer configuration
- a full training loop
- validation after every epoch
- gradient clipping
- checkpoint saving
- best-model selection
- training and validation loss curves

The best checkpoint is saved as:

```text
checkpoints/best_model.pt
```

The last checkpoint is saved as:

```text
checkpoints/last_model.pt
```

The currently selected best model was obtained at:

```text
Best epoch: 14
Best validation loss: 2.51238
```

The best validation loss was composed of approximately:

```text
Feature reconstruction loss: 0.98557
PDG reconstruction loss:     1.52681
```

---

## Notebook 4 — Representation Evaluation

**File:**

```text
04_representation_evaluation.ipynb
```

The evaluation notebook loads the best trained model and evaluates its self-supervised reconstruction performance and learned representations.

The evaluation includes:

- masked PDG reconstruction accuracy
- classification report
- confusion matrix
- continuous-feature reconstruction
- feature-level reconstruction diagnostics
- predicted-versus-true feature comparisons
- particle-level latent representation extraction
- event-level embedding extraction
- PCA visualization
- latent-space analysis
- analysis of the relationship between event embeddings and particle multiplicity

The model is evaluated on held-out validation and test datasets.

The event-level embedding is constructed by aggregating particle-level Transformer outputs over valid particles.

---

# Results

The current results should be interpreted as results from a **baseline research prototype**, rather than a state-of-the-art or fully optimized collider foundation model.

The purpose of the project is to demonstrate an end-to-end self-supervised learning workflow and investigate what information the model can learn from collider-event structure without using event labels during pre-training.

---

## Training

The model was trained for:

```text
30 epochs
```

using:

```text
AdamW optimizer
Learning rate: 1e-4
Weight decay:  1e-4
Mask ratio:    40%
```

The best validation checkpoint occurred at:

```text
Epoch 14
Validation loss: 2.51238
```

The training notebook also produces:

- total training and validation loss curves
- continuous-feature reconstruction loss curves
- PDG reconstruction loss curves

The training and validation losses remain relatively close, suggesting that this baseline configuration does not show a large train/validation gap during the 30-epoch training run.

---

## PDG Reconstruction

The model achieves approximately:

```text
Validation accuracy: approximately 47.6–47.7%

Test accuracy: approximately 47.8–47.9%
```

with approximately 40% of valid particles masked.

The result demonstrates that the Transformer learns non-trivial information about particle identity from the surrounding event context.

However, the classification performance is strongly influenced by the highly imbalanced particle distribution.

Some particle species occur much more frequently than others, and the model can favor dominant particle classes.

Therefore, overall accuracy alone should **not** be interpreted as evidence of equally strong reconstruction performance for all particle species.

The classification report and confusion matrix should be considered together with the overall accuracy.

Future versions should include:

- per-class accuracy
- macro F1
- weighted F1
- balanced accuracy
- normalized confusion matrices
- potentially class-weighted reconstruction losses

---

# Continuous-Feature Reconstruction

The model also reconstructs the continuous particle features.

The test-set reconstruction metrics obtained in the evaluation are:

| Feature | MAE | RMSE | Correlation |
|---|---:|---:|---:|
| `log_pt` | 0.7634 | 1.0019 | 0.1374 |
| `eta` | 0.7398 | 0.9941 | 0.1521 |
| `sin_phi` | 0.8845 | 0.9891 | 0.1565 |
| `cos_phi` | 0.8813 | 0.9872 | 0.1611 |
| `log_energy` | 0.7821 | 1.0073 | 0.0971 |
| `log_mass` | 0.6663 | 0.9954 | 0.0817 |
| `charge` | 0.7271 | 0.9905 | 0.1401 |

The relatively modest correlations indicate that the current model is still at an early prototype stage and that the reconstruction task can be substantially improved.

These results are therefore treated as a baseline rather than a final physics result.

---

# Learned Event Representation

A 128-dimensional event embedding is extracted from the Transformer.

Particle-level Transformer outputs are aggregated over valid particles to construct an event-level representation.

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

The first principal component has approximately:

```text
Pearson correlation:  r ≈ -0.357
Spearman correlation: r ≈ -0.347
```

This indicates that the learned latent representation contains information related to the structure and complexity of collider events.

Importantly, this does not by itself demonstrate that the representation has learned a physically optimal representation.

It is an initial diagnostic showing that the latent space is structured rather than completely random.

A stronger test would involve transferring the learned representation to a downstream physics task that was not used during self-supervised training.

---

# Computational Environment

The project is implemented using Python and PyTorch.

The model automatically selects the available device:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

The training experiments in the project were run using CUDA with a Tesla T4 GPU.

Example training environment:

```text
PyTorch: 2.11.0
CUDA available: True
GPU: Tesla T4
```

The evaluation notebooks can also run on CPU, although GPU acceleration is useful for faster training and larger-scale experiments.

A GPU is strongly recommended for:

- larger datasets
- larger Transformer models
- extensive hyperparameter studies
- repeated training experiments

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
pip install torch numpy awkward uproot matplotlib scikit-learn scipy jupyter
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
01_inspect_edm4hep.ipynb
        ↓
02_build_dataset.ipynb
        ↓
03_masked_particle_transformer.ipynb
        ↓
04_representation_evaluation.ipynb
```

The later notebooks depend on files and/or model checkpoints generated by earlier stages.

---

# Important Files

## Raw collider data

```text
clean_data_ecm240.root
```

Contains the input EDM4HEP collider-event data used for the project.

Depending on repository size and data-sharing restrictions, the raw ROOT file may not be included in the GitHub repository.

---

## Processed dataset

```text
processed_collider_events.pt
```

Contains:

- processed event sequences
- continuous particle features
- PDG token sequences
- train/validation/test indices
- normalization parameters
- PDG vocabulary
- padding and vocabulary information

---

## Best model checkpoint

```text
checkpoints/best_model.pt
```

Contains the trained Transformer model state corresponding to the best validation loss.

---

## Last model checkpoint

```text
checkpoints/last_model.pt
```

Contains the model and optimizer state from the final training epoch.

---

# Limitations

This repository represents a **research prototype**, not a production-ready collider foundation model.

Several aspects require further development.

## 1. Class Imbalance

The PDG vocabulary is highly imbalanced.

The current overall PDG reconstruction accuracy is therefore influenced strongly by common particle species.

Future evaluation should include:

- per-class accuracy
- macro F1
- weighted F1
- balanced accuracy
- normalized confusion matrices
- class-weighted losses

---

## 2. Continuous-Feature Reconstruction

The current reconstruction correlations are relatively low.

Possible improvements include:

- improved feature parameterization
- alternative reconstruction losses
- feature-specific loss weighting
- larger Transformer models
- improved masking strategies
- incorporating additional particle information

---

## 3. Physics-Aware Representations

The current model treats each event as an ordered sequence of particles sorted by transverse momentum.

This provides a simple deterministic baseline, but collider events are fundamentally more naturally described as sets of particles.

Future work could investigate:

- permutation-aware architectures
- set Transformers
- Lorentz-equivariant networks
- particle-flow representations
- four-momentum-based representations
- graph neural networks
- physics-informed relational information
- physics-informed positional representations

---

## 4. Larger-Scale Pre-Training

A true collider foundation model would require significantly larger and more diverse datasets, and potentially larger models.

The present work should therefore be considered a:

```text
Proof-of-concept / baseline study
```

rather than a full-scale foundation model.

---

## 5. Downstream Tasks

The most important next step is to test whether the learned representation transfers to physics tasks that were not used during self-supervised training.

Possible downstream tasks include:

- particle identification
- event classification
- signal/background discrimination
- anomaly detection
- jet classification
- new-physics searches
- event reconstruction

---

# Future Work

The project can be extended in several directions.

## Short-Term

- Improve the masked-particle objective.
- Introduce class-balanced PDG losses.
- Report per-particle-species performance.
- Optimize continuous-feature reconstruction.
- Compare different masking ratios.
- Compare different Transformer sizes.
- Evaluate different event-pooling strategies.
- Investigate separate masking strategies for PDG and continuous features.
- Test different reconstruction loss weightings.

---

## Medium-Term

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

Other possible experiments include:

- fine-tuning the pretrained model
- comparing pretrained and randomly initialized models
- linear-probe evaluations
- anomaly detection using latent representations
- supervised event classification using frozen embeddings

---

## Long-Term

Investigate larger collider foundation models trained on diverse event samples and evaluate transfer between:

- collision energies
- physical processes
- detector configurations
- simulation samples
- particle-level and reconstructed data
- physics tasks

A longer-term objective would be to investigate whether a self-supervised pretrained model can provide reusable representations across multiple HEP applications.

---

# Why This Project Matters

The main purpose of this project is not to claim that the current model is already a complete collider foundation model.

Instead, it demonstrates an end-to-end research workflow:

```text
EDM4HEP collider data
     │
     ▼
Physics-motivated preprocessing
     │
     ▼
Variable-length particle sequences
     │
     ▼
Self-supervised masked learning
     │
     ▼
Transformer representation
     │
     ▼
Quantitative reconstruction evaluation
     │
     ▼
Event-level latent-space analysis
     │
     ▼
Potential downstream physics applications
```

The project combines concepts from:

- High Energy Physics
- collider-event data
- EDM4HEP data formats
- particle physics
- deep learning
- Transformer architectures
- self-supervised learning
- representation learning
- masked modeling
- foundation models

The project is particularly intended as a hands-on exploration of the question:

> **Can useful physics representations emerge from the structure of collision data without providing explicit event labels during pre-training?**

---

# Technologies

The project uses:

- **Python**
- **PyTorch**
- **NumPy**
- **Awkward Array**
- **Uproot**
- **SciPy**
- **scikit-learn**
- **Matplotlib**
- **Jupyter**
- **EDM4HEP / ROOT-format collider data**

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

For example, an MIT License can be used if you want to allow broad reuse and modification of the code.

---

# Project Status

**Research prototype — baseline completed**

The current version demonstrates:

- EDM4HEP collider-data inspection
- particle-level feature engineering
- variable-length event representation
- PDG tokenization
- train/validation/test splitting
- masked-particle self-supervised learning
- Transformer encoder training
- feature reconstruction
- PDG reconstruction
- validation-based checkpoint selection
- test-set evaluation
- event-level embedding extraction
- PCA latent-space analysis
- analysis of the relationship between latent representations and particle multiplicity

Planned future work includes:

- class-balanced reconstruction
- improved continuous-feature reconstruction
- improved masking strategies
- larger-scale training
- downstream classification
- anomaly-detection studies
- comparison with alternative architectures
- set-based and physics-aware architectures
- physics-specific transfer-learning experiments

---

# Summary

This repository presents a prototype self-supervised learning framework for collider events.

The project starts from EDM4HEP collider data, constructs variable-length particle sequences, and trains a Transformer to reconstruct masked particle information without requiring explicit event-class labels during pre-training.

The model combines continuous particle features and PDG-token embeddings, randomly masks approximately 40% of valid particles, and uses a Transformer encoder to reconstruct:

- particle identity
- continuous particle properties

The learned particle representations are then aggregated into event-level embeddings and analyzed using PCA and correlations with event properties.

The current results demonstrate the feasibility of the end-to-end approach while also identifying clear areas for improvement, particularly in:

- class-balanced particle reconstruction
- continuous-feature prediction
- physics-aware architectures
- larger-scale pre-training
- downstream transfer learning

The longer-term goal is to investigate whether self-supervised Transformer representations can serve as reusable building blocks for multiple collider-physics applications.
