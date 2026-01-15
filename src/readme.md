## Code Overview

- **`run.py`**  
  Entry point for running the training and evaluation workflows.

- **`impl.py`**  
  Contains helper utilities used during training
  
- **`datio.py`**  
  Implements data input and loading routines for reading and preprocessing the input datasets.

- **`unet.py`**  
  Defines U-Net–based neural network architectures, including:
  - `unet`
  - `unet_res`
  - `unet_res_tf`

- **`resnet.py`**  
  Provides ResNet-style building blocks (residual blocks) used by the network architectures.

- **`physics.py`**  
  Implements the complete physics-guided forward modeling workflow used in the study.
