# Stage 1: High-Performance Ingestion Pipeline (Polars + PyArrow)

## Overview
Processed high-dimensional banking dataset (`data/DataSet.csv`, 116.5 MB, 3,925 columns, 9,082 rows) using a zero-copy Polars LazyFrame pipeline and PyArrow backend. The engine performs low-variance feature filtering, robust median imputation, `RobustScaler` transformation to mitigate wild banking outliers, and saves the cleaned dataset as an optimized Parquet file.

---

## Key Components Implemented

### 1. Preprocessing Engine
[polars_pipeline.py](file:///c:/Golden_Hour_Hold/src/preprocessing/polars_pipeline.py) & [__init__.py](file:///c:/Golden_Hour_Hold/src/preprocessing/__init__.py)
- **`HighDimensionalPreprocessor`**:
  - **Lazy Scanning**: Uses `pl.scan_csv(csv_path, infer_schema_length=10000, null_values=["NA", "null", "NULL", ""])`.
  - **Flexible Target Resolution**: Auto-detects whether the target vector is formatted as `"3924"` or `"F3924"` and isolates it as `pl.Int32` (`0=Legitimate, 1=Mule`).
  - **Numeric Column Evaluation**: Evaluates 3,852 numeric attributes, ignoring empty index columns (`""` / `"Unnamed: 0"`) and string/date attributes to prevent compute errors during variance calculation.
  - **Variance Filtering**: Computes attribute variances lazily across candidate columns. Retains attributes with variance > `variance_threshold` (1e-4) and supports optional `max_features` ranking (e.g. 150–300 top informative attributes).
  - **Robust Median Imputation**: Calculates column medians lazily and applies `pl.col(c).fill_null(self.medians[c])` without memory duplication.
  - **Outlier Mitigation**: Applies `RobustScaler` to numerical behavioral features.
  - **Parquet Export**: Writes output to `data/processed_data.parquet` via Polars/PyArrow.
  - **Persistence**: Implements `.save(filepath)` and `.load(filepath)` via `pickle`, plus `.transform(df)` for inference on new transaction batches.

### 2. Standalone Verification & Execution Script
[run_pipeline.py](file:///c:/Golden_Hour_Hold/run_pipeline.py)
- Command-line runner to execute the pipeline, verify dataset integrity, check for zero nulls, display class distributions, and test serialization.

---

## Verification & Execution Results

The pipeline ran on `data/DataSet.csv` with the following verification metrics:

| Metric | Raw Dataset (`DataSet.csv`) | Transformed Dataset (`processed_data.parquet`) |
| :--- | :--- | :--- |
| **Format** | CSV (text) | Apache Parquet (compressed tabular) |
| **File Size** | **116.5 MB** | **30.9 MB** (~73.5% disk reduction) |
| **Rows** | 9,082 | 9,082 |
| **Evaluated Numeric Attributes** | 3,852 | **3,235** (at `var > 1e-4`) / Configurable to top 150–300 |
| **Missing / Null Values** | Thousands of `NA`s across dense columns | **0** (All imputed via robust medians) |
| **Target Column** | `"F3924"` | `3924` (Cast to `Int32`, `0=Legitimate, 1=Mule`) |
| **Serialization** | N/A | State saved to `data/preprocessor.pkl` (131 KB) |

### Class Distribution
- **Class 0 (Legitimate)**: 8,875 transactions (97.72%)
- **Class 1 (Mule)**: 207 transactions (2.28%)

---

## How to Run

### Default (Variance Threshold = 1e-4)
```powershell
.venv\Scripts\python.exe run_pipeline.py
```

### With Top Informative Attribute Filtering (~150–300 Features)
```powershell
.venv\Scripts\python.exe run_pipeline.py 250
```
