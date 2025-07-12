# Version 1.0.3: 修正 TFF API 相容性問題

## 版本摘要

此版本的主要目標是解決 v1.0.2 中遇到的 TensorFlow Federated (TFF) API 相容性問題。它標誌著專案從一個不可用的狀態恢復到一個可運行的基礎，為後續正確整合差分隱私 (DP) 鋪平了道路。

## 主要功能與變更

* **環境穩定**:
    * 透過大量的環境測試與除錯，鎖定了與差分隱私套件相容的 TFF 版本。
    * **將 `tensorflow-federated` 的版本固定在 `0.86.0`**，並對齊了 `tensorflow`, `tensorflow-privacy` 等相關依賴的版本。
* **核心 API 修正**:
    * 放棄了已棄用的 `build_federated_averaging_process_from_model` API。
    * **切換回使用 `tff.learning.algorithms.build_weighted_fed_avg`**，這是 TFF 0.86.0 版本中推薦的標準聯邦平均演算法建構器。
* **DP 整合暫緩**: 由於 `build_weighted_fed_avg` 不直接接受 `DPKerasAdamOptimizer` 作為參數，此版本暫時移除了 v1.0.2 中直接整合 DP 優化器的程式碼，但保留了 DP 相關的組態參數和隱私預算計算邏輯的註解，以待後續開發。

## 狀態

* **穩定性**: **可運行**。
* **改進**: 成功解決了前一版本的致命性 API 錯誤，專案環境和基礎框架變得穩健。
* **限制**: 雖然環境已準備好，但差分隱私的功能尚未在此版本中被**正確地重新整合**。

---
### English Version

# Version 1.0.3: Fixing TFF API Compatibility Issues

## Version Summary

The primary goal of this version was to resolve the TensorFlow Federated (TFF) API compatibility issues encountered in v1.0.2. It marks the project's recovery from a non-functional state to a runnable baseline, paving the way for the proper integration of Differential Privacy (DP).

## Key Features & Changes

* **Environment Stabilization**:
    * After extensive testing and debugging, a compatible TFF version for the DP libraries was identified.
    * **Pinned the `tensorflow-federated` version to `0.86.0`** and aligned the versions of related dependencies like `tensorflow` and `tensorflow-privacy`.
* **Core API Fix**:
    * Abandoned the deprecated `build_federated_averaging_process_from_model` API.
    * **Switched back to using `tff.learning.algorithms.build_weighted_fed_avg`**, the recommended standard federated averaging builder in TFF v0.86.0.
* **DP Integration Postponed**: Since `build_weighted_fed_avg` does not directly accept `DPKerasAdamOptimizer` as a parameter, the code for direct DP optimizer integration from v1.0.2 was temporarily removed. However, DP-related configuration parameters and commented-out privacy budget calculation logic were retained for future development.

## Status

* **Stability**: **Runnable**.
* **Improvements**: Successfully resolved the critical API errors from the previous version, resulting in a stable project environment and framework.
* **Limitations**: Although the environment is ready, the Differential Privacy functionality has **not yet been correctly reintegrated** in this version.
