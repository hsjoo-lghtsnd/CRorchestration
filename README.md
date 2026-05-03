
# Heterogeneity-Aware Compression-Ratio Orchestration for Multi-User CSI Feedback in AI-RAN (MATLAB Implementation)

## Overview

Code for "Heterogeneity-Aware Compression-Ratio Orchestration for Multi-User CSI Feedback in AI-RAN," submitted.

This repository provides the MATLAB implementation for the CR-aware multi-user CSI feedback framework, including dataset generation, CR heterogeneity analysis (Fig. 2), and the main allocation experiments.

The code is organized into three main components:

* Dataset generation
* Figure reproduction (CR heterogeneity)
* Main experiments (CR allocation policies)

---

## Environment

* MATLAB: 2024b
* Required Toolboxes:

  * 5G Toolbox *(required for data generation only)*
  * Wireless Communications Toolbox *(required for data generation only)*

> The main experiments and figure reproduction do **not** require these toolboxes if pre-generated datasets are used.

---

## Project Structure

```
.
├── Data_Generation/
│   └── generate_data.m
│
├── CR_Heterogeneity_Fig2/
│   ├── main_pipeline.m
│   └── data/
│
├── Main_Experiment/
│   ├── main_pipeline.m
│   └── data/
│
└── combined/   (generated)
    ├── train.mat
    ├── valid.mat
    └── test.mat
```

---

## Quick Start

### 1. Dataset Generation

Run the following script:

```matlab
Data_Generation/generate_data.m
```

This generates:

```
dataset/combined/train.mat
dataset/combined/valid.mat
dataset/combined/test.mat
```

---

### 2. Reproduce Fig. 2 (CR Heterogeneity)

1. Move dataset files:

```
dataset/combined/*.mat → CR_Heterogeneity_Fig2/data/
```

2. Run:

```matlab
CR_Heterogeneity_Fig2/main_pipeline.m
```

This script generates the CR heterogeneity analysis figure used in Fig. 2.

---

### 3. Run Main Experiments

1. Move dataset files:

```
dataset/combined/*.mat → Main_Experiment/data/
```

2. Run:

```matlab
Main_Experiment/main_pipeline.m
```

This executes:

* CR allocation policies
* Performance evaluation (e.g., sum-rate comparison)
* Summary-driven and heuristic policy evaluation

---

## Dataset Description

Each `.mat` file contains pre-generated channel samples used for:

* Training (train.mat)
* Validation (valid.mat)
* Testing (test.mat)

> The dataset is shared across Fig. 2 and all main experiments.

---

## Reproducibility

* All experiments are designed to be reproducible with fixed dataset splits.
* For full reproducibility, ensure:

  * Same MATLAB version (2024b recommended)
  * Same dataset files are used

---

## Notes

* Run all scripts from the **project root directory**:

```matlab
addpath(genpath(pwd));
```

* Data generation requires MATLAB toolboxes, but:

  * Fig. 2 reproduction
  * Main experiments
    can be executed using only the generated `.mat` files.

* Brute-force oracle methods may become computationally infeasible for large user counts.

---

## Contact

For questions regarding the implementation or paper, please contact the repository owner.
