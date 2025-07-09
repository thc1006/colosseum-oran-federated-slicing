Below is a structured, code-aligned walkthrough of the notebook **《colosseum-oran-federated-slicing.ipynb》**.
For clarity I reference the notebook’s cell order (<#>) and key line numbers, then summarise the preprocessing logic and the exact content of the final artefacts.

> U should run with Goolge colab T4(large RAM), L4 or A100 GPU, Because only they have enough RAM.

## 1. Notebook Layout & Purpose

| Cell  | Type     | Purpose (high-level)                                                                                                                   |
| ----- | -------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | Markdown | Title / heading                                                                                                                        |
| **1** | Code     | Clone ColO-RAN dataset from GitHub and set `DATASET_PATH`                                                                              |
| 2     | Markdown | Section break                                                                                                                          |
| **3** | Code     | **`ColoRANDataProcessorPro`**: load **all 588 experiment combinations**, clean & merge raw CSVs, save *raw* Parquet files              |
| 4     | Markdown | Section break                                                                                                                          |
| **5** | Code     | **Memory-Optimised Feature Engineering Pipeline**: transform raw data → `coloran_processed_features.parquet` + `feature_metadata.json` |


## 2. Cell 1 – Dataset Download *(lines 1-\~80)*

1. Defines repository URL `https://github.com/wineslab/colosseum-oran-coloran-dataset.git`.
2. Deletes any pre-existing folder to avoid merge conflicts.
3. Executes **`git clone`** via `subprocess.run`.
4. Performs a sanity check: if `rome_static_medium/` directory is missing, prints full tree for debugging.
5. Sets global variable `DATASET_PATH` used by later cells.

> **Outcome:** Fresh local copy of the dataset ready for parsing.

## 3. Cell 3 – `ColoRANDataProcessorPro` *(\~300 lines)*

### 3.1 Constructor

```python
self.base_stations        = [1, 8, 15, 22, 29, 36, 43]
self.scheduling_policies  = ['sched0','sched1','sched2']   # RR, WF, PF
```

*Seven gNBs × three schedulers × 28 configs → 588 experiment folders.*

### 3.2 `auto_detect_structure()`

* Scans several possible path layouts to locate the top-level `rome_static_medium`.
* Logs discovered scheduler directories for transparency.

### 3.3 `load_metrics()`

Iterates through each **experiment-folder / base-station / scheduler** trio:

1. Builds glob patterns for per-second BS, UE and slice CSVs.
2. Reads each CSV with **`pandas.read_csv`**; missing files are reported not fatal.
3. Extracts metadata (experiment id, scheduler, training config, BS id) from the file path.

### 3.4 `merge_datasets()`

Merges BS-level, UE-level and slice-level frames on timestamp & identifiers, aligning disparate sampling rates.

### 3.5 Feature Building (raw-level)

* Saves three raw Parquet files:

  * `raw_bs_data.parquet`
  * `raw_ue_data.parquet`
  * `raw_slice_data.parquet`

All compressed with **Snappy** to minimise storage.

---

## 4. Cell 5 – Memory-Optimised Feature Engineering

### 4.1 Safety wrapper

`load_raw_data_if_exists()` ensures the three raw Parquets are present; if not, prints an error and aborts.

### 4.2 `SliceFeatureEngineer`

* **Batch size = 200 000 rows** to keep RAM usage stable on Colab.
* For each batch:

  1. **Timestamp decomposition** → `hour`, `minute`, `day_of_week`.
  2. **Scheduler mapping**: `'sched0'→0, sched1→1, sched2→2` ⇒ `sched_policy_num`.
  3. **Vectorised RBG allocation** by mapping `(training_config, slice_id)` to a lookup table built once at init.
  4. **Resource utilisation metrics**

     ```python
     prb_utilization = sum_granted_prbs / sum_requested_prbs
     throughput_efficiency = (tx_brate_dl_Mbps) / sum_granted_prbs
     ```
  5. **QoS score** (weighted error rates and latency) with a fully vectorised formula.
  6. **Network load** = `num_ues / 42`.
  7. **Composite target**

     ```python
     allocation_efficiency = 0.5*throughput_efficiency
                            +0.3*qos_score
                            +0.2*prb_utilization
     ```

     *Clipped to $0, 1$.*
  8. **Down-casting** integers (`int64`→`uint8/16/32`) and floats (`float64`→`float32`) to cut memory \~3-4×.

### 4.3 `save_processed_data_to_parquet()`

* Runs `optimize_datatypes()` one more time.
* Writes **`coloran_processed_features.parquet`** (Snappy, no index, PyArrow engine).
* Persists accompanying JSON:

  ```json
  {
    "feature_names": [...],
    "total_records": N,
    "processing_date": "...",
    "file_size_mb": ...,
    "compression_ratio": ...
  }
  ```

---

## 5. Final Dataset Schema

| Column                  | Type(post-opt.) | Derivation                                 |   |                            |
| ----------------------- | --------------- | ------------------------------------------ | - | -------------------------- |
| `num_ues`               | `uint8/16`      | Simultaneous UEs on slice-BS               |   |                            |
| `slice_id`              | `uint8`         | Original categorical id                    |   |                            |
| `sched_policy_num`      | `uint8`         | Encoded scheduler (0 = RR, 1 = WF, 2 = PF) |   |                            |
| `allocated_rbgs`        | `uint8/16`      | RBGs granted via lookup table              |   |                            |
| `bs_id`                 | `uint8`         | gNB identifier                             |   |                            |
| `exp_id`                | `uint16`        | Experiment folder id                       |   |                            |
| `sum_requested_prbs`    | `uint16/32`     | Total PRBs requested (batch)               |   |                            |
| `sum_granted_prbs`      | `uint16/32`     | Total PRBs granted                         |   |                            |
| `prb_utilization`       | `float32`       | Granted / requested (0-1)                  |   |                            |
| `throughput_efficiency` | `float32`       | Mbps per granted PRB                       |   |                            |
| `qos_score`             | `float32`       | Composite DL/UL error & latency metric     |   |                            |
| `network_load`          | `float32`       | `num_ues/42` normalised load               |   |                            |
| `hour`                  | `uint8`         | Hour of day                                |   |                            |
| `minute`                | `uint8`         | Minute of hour                             |   |                            |
| `day_of_week`           | `uint8`         | 0 = Mon … 6 = Sun                          |   |                            |
| `allocation_efficiency` | `float32`       | **Target variable** (0-1)                  |   |                            |
| `sched_policy`          | `category`      | Original string (\`sched0                  | 1 | 2\`)—kept for traceability |
| `training_config`       | `category`      | One of 28 baseline RBG configurations      |   |                            |

> **Row granularity:** “one time-stamp × one slice × one gNB”.
> **Typical size (Colab run):** \~3-4 M rows, 70–80 MB on disk (Snappy).

---

## 6. Key Pre-processing “Techniques at a Glance”

| Technique                      | Where (line)                    | Purpose                                                       |
| ------------------------------ | ------------------------------- | ------------------------------------------------------------- |
| **Path auto-detection**        | `auto_detect_structure()`       | Robust to repo layout changes.                                |
| **Vectorised feature mapping** | `_vectorized_rbg_allocation`    | O(N)→O(1) per row, avoids Python loops.                       |
| **Chunked processing**         | `process_data_in_batches()`     | Keeps peak RAM < 2 GB even for 3 M rows.                      |
| **Type down-casting**          | `optimize_datatypes()`          | Reduces memory & Parquet size **3-4×**.                       |
| **Composite efficiency KPI**   | `allocation_efficiency` formula | Single scalar target combining throughput, QoS & utilisation. |
| **Metadata sidecar**           | `feature_metadata.json`         | Records schema & processing provenance for reproducibility.   |

---

### Deliverables Generated by the Notebook

1. **`coloran_processed_features.parquet`** – ready-to-train feature table (18 columns above).
2. **`feature_metadata.json`** – schema & statistics to document the file.
3. (*Intermediate*) `raw_bs_data.parquet`, `raw_ue_data.parquet`, `raw_slice_data.parquet`.

---

This line-by-line breakdown should give reviewers a precise understanding of **every preprocessing step performed** and the exact structure of the **final dataset** to be cited in your project documentation.
