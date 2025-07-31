# QMCL to ANI-DS Converter

This folder contains a script to convert data from the [QMCL dataset](https://www.nature.com/articles/s41597-025-04720-7) (Google DeepMind, 2024), released in TensorFlow's TFRecord format, into a **TorchANI-compatible HDF5** format for use in training machine learning models on molecular systems.

## Motivation

The release of the **QMCL dataset** marked a significant step forward in the scale and resolution of quantum mechanical data available for machine learning. Compared to earlier datasets such the original ANI-1x/2x, with more elements coverage and charged species.

However, QMCL is stored in **sharded TFRecord** files optimized for TensorFlow. In contrast, frameworks like **TorchANI** and many PyTorch-based ML pipelines expect **HDF5 datasets**.

## About the Script

The script `convert_qmcl_to_anids.py` converts a specified QMCL subshard into a valid `.h5` file using TorchANI's `ANIDataset` interface. It performs:
- Reading and parsing of TFRecords
- Conversion of atomic units (e.g., Bohr to Angström)
- Grouping and filtering by atom count
- Optional batching (`CHUNKSIZE`) and hash checking
- Output to `.h5` format ready for training or inference

## What's next?

We aim to use this dataset to train more general-purpose machine learning interatomic potentials. Currently, I'm focusing on analyzing the dataset to better understand the diversity of structures it contains. This includes using RDKit, OpenBabel, and dimensionality reduction techniques to characterize the chemical space covered by QMCL, and to compare it with that of ANI-2x.
