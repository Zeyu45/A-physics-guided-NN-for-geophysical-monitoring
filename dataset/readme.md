# Dataset Description and Access

## Overview
This project uses a dataset consisting of 3,200 NumPy (`.npy`) files, generated and used for the experiments presented in the accompanying conference paper.

Each data sample corresponds to a single simulation/observation and is stored as a standalone `.npy` file. The dataset is organized consistently with the preprocessing and training pipelines provided in this repository.

## Dataset Structure
The full dataset contains:
- 800 samples
- 4 `.npy` files per sample
- Total files: 3,200

Example structure:



## Data Size
Due to the large number of files and total size, the complete dataset is **not hosted directly on GitHub**.

## Access to Full Dataset
The full dataset is archived on **Zenodo** and can be accessed via the following DOI:

> **DOI:** _to be added upon publication_

The Zenodo archive includes:
- The complete dataset
- A checksum file for data integrity verification
- A README describing the archive structure

## Download script:
#!/bin/bash

echo "Downloading dataset from Zenodo..."

ZENODO_DOI="10.5281/zenodo.XXXXXXX"
ZENODO_URL="https://zenodo.org/record/XXXXXXX/files/dataset.zip"

wget $ZENODO_URL -O dataset.zip

echo "Extracting dataset..."
unzip dataset.zip

echo "Done. Dataset available in ./dataset/"

## Sample Dataset
To allow immediate testing and reproducibility, a **small representative subset** of the dataset is provided in this repository under:


