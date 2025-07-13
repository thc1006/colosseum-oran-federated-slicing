# Version 1.0.5: 強化差分隱私 
### 1. 新增功能

* **環境疑難排解區塊**

  * `data_quality_report()`：輸出缺失值、異常值與分佈統計。
  * `visualize_feature_corr()`：快速生成特徵相關性熱圖。
* **差分隱私優化器封裝**

  * `make_dp_keras_optimizer()`：整合梯度裁剪、雜訊注入與參數動態覆寫。

### 2. 功能強化

* **資料前處理**

  * 引入 `RobustScaler`，提升對離群值的容忍度。
  * 批次大小預設改為 64。
* **隱私保護參數**

  * `DP_NOISE_MULTIPLIER` 預設值由 1.0 提升至 1.5。
* **訓練流程穩健性**

  * 增加 `try/except` 與 `gc.collect()`，減少 OOM 風險。
  * 支援模型與 scaler 以 `.keras` 與 `.pkl` 雙格式輸出。

### 3. 相容性修正

* 抑制 `FutureWarning`，確保與 scikit-learn 與 pandas 新版兼容。
* 調整匯入順序，避免 `ImportError: name 'python' is not defined`。

---

## Release Notes

**File**: `colosseum_oran_federated_slicing_v1_0_4_ipynb_的副本.ipynb`
**Compared to**: `colosseum_oran_federated_slicing_v1_0_4.ipynb`

### 1. New Features

  * `data_quality_report()` for missing-value, outlier, and distribution checks.
  * `visualize_feature_corr()` for quick feature-correlation heatmaps.
* **DP Optimizer Wrapper**

  * `make_dp_keras_optimizer()` encapsulates gradient clipping, noise injection, and dynamic parameter overrides.

### 2. Enhancements

* **Pre-processing**

  * Integrated `RobustScaler` to improve resilience to outliers.
  * Default batch size increased to 64.
* **Privacy Parameters**

  * Default `DP_NOISE_MULTIPLIER` raised from 1.0 to 1.5.
* **Training Stability**

  * Added `try/except` guards and `gc.collect()` to mitigate OOM issues.
  * Models and scalers are now saved in both `.keras` and `.pkl` formats.

### 3. Compatibility Fixes

* Suppressed `FutureWarning` messages for smoother upgrades to recent scikit-learn and pandas versions.
* Re-ordered imports to avoid `ImportError: name 'python' is not defined`.
