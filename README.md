# A physics-guided neural network-based inversion framework for monitoring of CO2 storage via angle-dependent seismic images

This repository implements a physics-guided neural network framework for monitoring geologic carbon storage. The network predicts CO2 saturation fields from time-lapse, angle-dependent seismic reflection data, embedding both rock-physics and seismic forward modeling in a differentiable training objective.  

The architecture combines a U-Net CNN with a transformer bottleneck, enabling fast, interpretable inference of plume shape and migration over time. The method is tested on synthetic datasets and recovers plume extent reliably, with smooth fronts as expected from band-limited imaging.

---

## Repository Structure

```
src/
├── Model training, evaluation, and inference code

data_preparation/
├── Scripts and notebooks for generating input data, including
│   base survey models, wavelets, and angle-dependent seismic images

documentation/
├── Written report and slides

dataset/
├── base-survey models and access to full dataset

```
