# Masked Particle Transformer for Collider Event Representation Learning

A Transformer-based self-supervised learning pipeline for learning particle and
event representations from high-energy collider data.

## Overview

This project explores masked modeling for particle-physics events using a
Transformer encoder.

Collider events are represented as variable-length sequences of final-state
particles. Each particle is described by continuous kinematic information and
a categorical PDG-ID token.

During training, 40% of valid particles are randomly masked. The Transformer
must reconstruct information about the masked particles from the surrounding
event context.

The model has two reconstruction objectives:

1. Continuous particle-feature reconstruction
2. PDG-ID reconstruction

No external physics labels are used to train the reconstruction task.

The project covers the complete workflow from ROOT/EDM4HEP data inspection to
dataset construction, Transformer training, checkpointing, and representation
evaluation.

---

## Project pipeline

```text
EDM4HEP / ROOT data
        │
        ▼
Data inspection
        │
        ▼
Final-state particle selection
        │
        ▼
Particle representation
        │
        ├── Continuous features
        │
        └── PDG-ID token
        │
        ▼
Variable-length event dataset
        │
        ▼
Masked Particle Transformer
        │
        ├── Feature reconstruction
        │
        └── PDG reconstruction
        │
        ▼
Learned particle/event representations
        │
        ▼
Evaluation and latent-space analysis
