# Colosseum-ORAN Federated Slicing: Offline FL for 5G/6G RAN Resource Allocation
> **Colosseum-ORAN Federated Slicing** is an end-to-end Google Colab notebook that trains a federated deep learning model to optimize PRB re-allocation, scheduler switching, and load balancing across RAN slices.  
> It uses the **ColO-RAN dataset** released by WINES Lab and provides GPU-ready diagnostics plus optional Differential Privacy.

## Why this project?
* **Real RAN traces** – built on the Colosseum/ORAN “ColO-RAN” dataset 【[Dataset link](https://github.com/wineslab/colosseum-oran-coloran-dataset)】  
* **One-click Colab** – no local CUDA setup needed  
* **Full reproducibility** – random seeds fixed, artifacts serialized  
* **Differential Privacy ready** – clipping norm, noise multiplier, and TFF DP aggregators exposed as flags  

The notebook splits **`coloran_processed_features.parquet`** into seven base-station clients and, using TensorFlow Federated’s FedAvg, trains a four-layer fully connected network to regress **`allocation_efficiency`**. The resulting model and its scalers are saved as a `.keras` file alongside a `.pkl`, so they can be embedded directly into a gNB or xApp for sub-20 ms-cycle inference—driving dynamic slice PRB reallocation, scheduler switching, and cross-cell load balancing.

## Quick start
1. Open the notebook in Colab  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1l_sfn29npZRbG6vuYu2amyAkt1vie4Jk)
2. [Download](https://github.com/thc1006/coloran-dynamic-slice-optimizer/blob/main/coloran_processed_features.parquet) `coloran_processed_features.parquet` (≈ 400 MB) and upload it to Colab **files**(temp) or mount GDrive.  
3. Press **▶ Run all**. Training logs and plots appear inline; a Keras model and pickle artifacts are saved.

## Repository layout
| Path | Purpose |
|------|---------|
| `notebook.ipynb` | Main workflow – data prep, GPU diagnostics, TensorFlow Federated setup, training loop, plotting. |
| `coloran_processed_features.parquet` | Pre-processed features derived from the ColO-RAN traces. |
| `README.md` | You are here. |

## Dataset
ColO-RAN traces are provided by WINES Lab.：<https://github.com/wineslab/colosseum-oran-coloran-dataset>
* If you use this code or original dataset, u need to use license: GPL-3.0 license
* Below is thier research:
> M. Polese *et al.*, “ColO-RAN: Developing Machine Learning-based xApps for Open RAN Closed-loop Control,” **IEEE TMC**, 2022.

## Training details
| Hyper-parameter | Default |
|-----------------|---------|
| Total clients | 7 |
| Rounds | 30 |
| Clients per round | 7 |
| Local epochs | 2 |
| Client LR | 5e-4 |
| Server LR (Adam) | 0.01 |
| Clipping norm (DP) | 1.0 (configurable) |
| Noise multiplier (DP) | 0.0 (disabled by default) |

## License
Released under the AGPL-3.0 license – see `LICENSE`.

## Acknowledgements
* This notebook builds on TensorFlow Federated tutorials and the Colosseum/ORAN open dataset.
* NTSC and my liver.
