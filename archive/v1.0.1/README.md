# Version 1.0.1: 超參數優化與本地訓練

## 版本摘要

此版本在前一版的基礎上，進行了重要的優化與修正。核心變更是調整了聯邦學習的超參數，引入了本地訓練週期 (Local Epochs)，並改進了資料處理流程，以更真實地模擬聯邦場景。

## 主要功能與變更

* **超參數調整**:
    * 增加 `NUM_ROUNDS` 至 30，期望模型能更充分地收斂。
    * 調整 `CLIENT_LEARNING_RATE` 和 `SERVER_LEARNING_RATE`，尋求更佳的學習性能。
* **引入本地訓練週期**:
    * 新增 `LOCAL_EPOCHS = 3` 參數。
    * 在每一輪聯邦學習中，客戶端會在本地資料上訓練 3 個 Epoch，這有助於加速模型收斂並減少通訊開銷。
* **關鍵修正：客戶端獨立資料標準化**:
    * 放棄了全域 `StandardScaler`。
    * 改為**為每一個客戶端獨立計算並應用 `StandardScaler`**。這是一個關鍵改進，能更好地處理不同客戶端之間資料分佈不均 (Non-IID) 的問題。

## 狀態

* **穩定性**: 可運行且性能優於 v1.0.0。
* **改進**: 模型的訓練效率和對資料異質性的處理能力得到了顯著提升。
* **限制**: 仍然未包含差分隱私 (Differential Privacy) 保護機制。

---
### English Version

# Version 1.0.1: Hyperparameter Tuning & Local Training

## Version Summary

This version builds upon the previous one with significant optimizations and corrections. The core changes involve tuning federated learning hyperparameters, introducing local epochs, and refining the data processing workflow to better simulate a real-world federated scenario.

## Key Features & Changes

* **Hyperparameter Tuning**:
    * Increased `NUM_ROUNDS` to 30 for more thorough model convergence.
    * Adjusted `CLIENT_LEARNING_RATE` and `SERVER_LEARNING_RATE` to seek better learning performance.
* **Introduction of Local Epochs**:
    * Added a new `LOCAL_EPOCHS = 3` parameter.
    * In each federated round, clients now train on their local data for 3 epochs, which helps accelerate convergence and reduce communication overhead.
* **Key Fix: Per-Client Data Standardization**:
    * Abandoned the global `StandardScaler`.
    * Switched to **calculating and applying `StandardScaler` independently for each client**. This is a crucial improvement for better handling of Non-IID data distributions among clients.

## Status

* **Stability**: Runnable, with performance superior to v1.0.0.
* **Improvements**: The model's training efficiency and its ability to handle data heterogeneity have been significantly enhanced.
* **Limitations**: Still does not include a Differential Privacy (DP) protection mechanism.
