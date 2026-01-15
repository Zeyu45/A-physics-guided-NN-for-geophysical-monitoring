# Dataset Description and Access

## Overview
This dataset contains 4,800 NumPy (.npy) files generated for the experiments presented in the associated EAGE conference paper. The data are organized into 2 subsets, saturation_maps contains the groundtruth and prediction results, and input_data contains angle-dependent seismic images for training.

The dataset supports all preprocessing, training, and evaluation workflows provided in the accompanying GitHub repository. Due to size constraints, only a small representative subset is hosted on GitHub, while the complete dataset is archived here for reproducibility and transparency.
Each data sample corresponds to a single simulation/observation and is stored as a standalone `.npy` file. The dataset is organized consistently with the preprocessing and training pipelines provided in this repository.

## Dataset Structure
dataset/
├─ saturation_maps/
│ ├─ frames_true/
│ │ ├─ ...
│ │ └─ ...
│ ├─ frames_pref/
│ │ ├─ ...
│ │ └─ ...
│
├─ angle-dependent seismic images/
│ ├─ ...
│ └─ ..
│
├─ base-survey models/
│ ├─ ...
│ └─ ..
│
├─ wavelets/
│ ├─ ...
│ └─ ..

## Data Size
Due to the large number of files and total size, the complete dataset is **not hosted directly on GitHub**.

## Access to Full Dataset
The full dataset is archived on **Zenodo** and can be accessed via the following DOI:

> https://doi.org/10.5281/zenodo.18258167


