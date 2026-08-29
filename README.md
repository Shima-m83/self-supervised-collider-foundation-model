# self-supervised-collider-foundation-model
Developed a Transformer-based foundation model prototype for collider event representation learning. The model performs self-supervised pretraining on simulated particle collision events by masking particle tokens and reconstructing continuous kinematic features and particle identities. Implemented in PyTorch and trained on CERN SWAN GPU Infrastructure.


Implemented a masked particle modeling approach inspired by MAE-style
foundation models.

Input:
- simulated collider events
- particle features
- PDG identifiers

Architecture:
- particle feature projection
- PDG embeddings
- Transformer encoder
- reconstruction heads

Pretraining objective:
- mask 40% of particles
- reconstruct particle kinematics
- reconstruct particle identity

Results:
- trained on CERN SWAN Tesla T4 GPU
- validation reconstruction loss monitored
- latent event representations visualized using PCA
