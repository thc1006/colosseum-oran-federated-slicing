# Version 1.0.0: 基礎聯邦學習流程建構

## 版本摘要

此版本是 `colosseum-oran-federated-slicing` 專案的初始版本。目標是建立一個功能性的聯邦學習 (Federated Learning) 基礎框架，用於在 O-RAN 網路切片資料上訓練一個 DNN (深度神經網路) 模型。

## 主要功能與變更

* **基礎聯邦學習框架**: 使用 `tensorflow_federated.learning.build_federated_averaging_process` API 建立標準的 FedAvg (聯邦平均) 演算法。
* **模型架構**: 建立了一個包含 `Dropout` 和 `L2` 正規化的序貫式 (Sequential) DNN 模型，以防止過擬合。
* **資料處理**:
    * 從 Parquet 檔案讀取預處理過的特徵資料。
    * 使用**全域 `StandardScaler`** 對整個資料集進行標準化。
    * 將資料集按照 `enb_id` (模擬基地台 ID) 分割給 7 個客戶端。
* **訓練流程**:
    * 實現了基本的聯邦學習訓練迴圈。
    * 每一輪 (Round) 訓練後，記錄並保存伺服器模型的損失 (Loss) 與準確率 (Accuracy)。
* **結果分析**: 繪製基本的學習曲線 (Loss vs. Round, Accuracy vs. Round) 來評估模型收斂情況。

## 狀態

* **穩定性**: 可運行。
* **限制**:
    * 資料標準化採用全域方式，未能充分考慮客戶端之間的資料異質性 (Data Heterogeneity)。
    * 超參數 (`CLIENT_LEARNING_RATE`, `NUM_ROUNDS` 等) 尚處於初步設定階段，未經優化。
    * 未包含差分隱私 (Differential Privacy) 保護機制。

---
### English Version

# Version 1.0.0: Basic Federated Learning Process Construction

## Version Summary

This is the initial version of the `colosseum-oran-federated-slicing` project. The goal is to establish a functional Federated Learning (FL) framework for training a Deep Neural Network (DNN) model on O-RAN network slicing data.

## Key Features & Changes

* **Basic Federated Learning Framework**: Uses the `tensorflow_federated.learning.build_federated_averaging_process` API to create a standard FedAvg (Federated Averaging) algorithm.
* **Model Architecture**: A sequential DNN model including `Dropout` and `L2` regularization was created to prevent overfitting.
* **Data Processing**:
    * Reads preprocessed feature data from a Parquet file.
    * Uses a **global `StandardScaler`** to standardize the entire dataset.
    * Splits the dataset among 7 clients based on `enb_id` (simulated eNodeB ID).
* **Training Process**:
    * Implements a basic federated learning training loop.
    * Records and saves the server model's loss and accuracy after each round.
* **Results Analysis**: Plots basic learning curves (Loss vs. Round, Accuracy vs. Round) to evaluate model convergence.

## Status

* **Stability**: Runnable.
* **Limitations**:
    * Data standardization is performed globally, failing to adequately consider data heterogeneity among clients.
    * Hyperparameters (e.g., `CLIENT_LEARNING_RATE`, `NUM_ROUNDS`) are in a preliminary stage and have not been optimized.
    * Does not include a Differential Privacy (DP) protection mechanism.