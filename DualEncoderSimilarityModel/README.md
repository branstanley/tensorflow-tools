# DualEncoderSimilarityModel

This folder contains code for dual encoder similarity models, including dynamic triplet mining and training with adaptive margin. These models are typically used for tasks involving similarity learning and metric learning.

The goal of this project was that I have two seperate signal streams, and I needed to match them to a single device.

Training process:
1) An encoder is created for both signal streams, and trained for Multi-class classification.
2) Encoders are frozen, and a projection head is applied to both encoders
3) Encoders are trained using a combined loss methodology of triplet loss and contrastive loss.  The training curiculum uses a progressive difficulty with a stage-wise margin, allowing the model to become aclimated to each stage of the curriculum.
4) The final encoders become a Dual Network with a Cosine Similarity output.