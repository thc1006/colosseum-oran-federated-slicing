# Version 1.0.2: 首次整合差分隱私 (DP)

## 版本摘要

此版本是整合差分隱私 (Differential Privacy, DP) 的首次嘗試。目標是使用 `tensorflow_privacy` 套件為聯邦學習流程增加隱私保護，特別是針對客戶端在訓練過程中分享的模型更新。

## 主要功能與變更

* **引入差分隱私**:
    * 在組態設定中新增 DP 相關參數，如 `DP_L2_NORM_CLIP` (梯度裁剪範數) 和 `DP_NOISE_MULTIPLIER` (雜訊乘數)。
    * 嘗試使用 `tensorflow_privacy.DPKerasAdamOptimizer` 作為客戶端的優化器，以在本地訓練時自動添加雜訊。
* **API 使用變更**:
    * 為了整合自定義的 DP 優化器，將聯邦流程的建構 API 從 `build_federated_averaging_process` 切換為 `build_federated_averaging_process_from_model`。
* **隱私預算追蹤**:
    * 引入 `dp_accounting` 套件來計算並追蹤隱私預算 (Epsilon, δ)。
    * 在訓練迴圈中增加了基於隱私預算的停止條件，防止隱私洩漏超出預設範圍。
* **結果分析**: 新增了隱私-效用權衡 (Privacy-Utility Tradeoff) 的分析圖表。

## 狀態

* **穩定性**: **不穩定/已棄用**。
* **問題**: 此版本遭遇了嚴重的 **API 版本相容性問題**。`build_federated_averaging_process_from_model` 在較新的 TensorFlow Federated (TFF) 版本中已被棄用，導致程式無法順利執行。
* **結論**: 雖然整合 DP 的方向是正確的，但技術實現路徑需要修正。此版本是一個重要的「試錯」記錄。

---
### English Version

# Version 1.0.2: First Attempt at Differential Privacy (DP) Integration

## Version Summary

This version represents the first attempt to integrate Differential Privacy (DP). The objective was to use the `tensorflow_privacy` library to add privacy protection to the federated learning process, specifically for the model updates shared by clients during training.

## Key Features & Changes

* **Introduction of Differential Privacy**:
    * Added DP-related parameters to the configuration, such as `DP_L2_NORM_CLIP` (gradient clipping norm) and `DP_NOISE_MULTIPLIER`.
    * Attempted to use `tensorflow_privacy.DPKerasAdamOptimizer` as the client optimizer to automatically add noise during local training.
* **API Usage Change**:
    * To integrate the custom DP optimizer, the federation process builder API was switched from `build_federated_averaging_process` to `build_federated_averaging_process_from_model`.
* **Privacy Budget Tracking**:
    * Introduced the `dp_accounting` library to compute and track the privacy budget (Epsilon, δ).
    * Added a stopping condition to the training loop based on the privacy budget to prevent privacy leakage beyond a predefined threshold.
* **Results Analysis**: Added visualizations for the privacy-utility tradeoff.

## Status

* **Stability**: **Unstable / Deprecated**.
* **Issue**: This version encountered a critical **API compatibility issue**. The `build_federated_averaging_process_from_model` API is deprecated in newer versions of TensorFlow Federated (TFF), causing the program to fail.
* **Conclusion**: While the direction of integrating DP was correct, the technical implementation path required a revision. This version serves as an important record of a "trial and error" phase.