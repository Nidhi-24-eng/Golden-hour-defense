# Implementation Plan: High-Performance Ingestion Pipeline (Polars + PyArrow)

Build a high-performance, low-memory data ingestion and preprocessing pipeline (`HighDimensionalPreprocessor`) in [polars_pipeline.py](file:///c:/Golden_Hour_Hold/src/preprocessing/polars_pipeline.py) using Polars LazyFrames and PyArrow to process 3,900+ dense banking features, filter low-variance columns down to ~150–300 informative features, perform robust median imputation against banking outliers, scale features with `RobustScaler`, and save clean tabular data as Parquet.

## User Review Required

> [!IMPORTANT]
> - **Target Column Name Resolution**: The CSV header contains column names in the format `"","F1","F2",...,"F3924"`. The target column in the CSV is `"F3924"`, while the user specification refers to `"3924"`. We will make `HighDimensionalPreprocessor` dynamically resolve both `"3924"` and `"F3924"` (and strip the leading empty string/index column `""`) so it seamlessly matches the dataset.
> - **Non-Numeric Column Handling**: The dataset contains some string/categorical columns (e.g. account types, dates). Computing `.var()` on string columns would trigger Polars compute errors. We will restrict variance filtering and numerical scaling to numeric feature columns, ensuring zero crashes on non-numeric columns.

## Proposed Changes

### Preprocessing Engine

#### [NEW] [__init__.py](file:///c:/Golden_Hour_Hold/src/__init__.py)
- Package root initializer.

#### [NEW] [__init__.py](file:///c:/Golden_Hour_Hold/src/preprocessing/__init__.py)
- Preprocessing module initializer exporting `HighDimensionalPreprocessor`.

#### [NEW] [polars_pipeline.py](file:///c:/Golden_Hour_Hold/src/preprocessing/polars_pipeline.py)
Implement `HighDimensionalPreprocessor`:
- `__init__(target_col="3924", variance_threshold=1e-4)`: Initialize preprocessor state, scaler (`RobustScaler`), feature list, and medians dict.
- `_resolve_target_col(schema)`: Auto-detect whether target is named `"3924"` or `"F3924"`.
- `fit_transform_lazy(csv_path, output_parquet, batch_size=None)`:
  1. Lazily scan CSV using `pl.scan_csv(csv_path, infer_schema_length=10000)`.
  2. Filter candidate columns: exclude empty index (`""`), target column, and keep only numeric types (`schema[c].is_numeric()`).
  3. Efficient variance computation via lazy expressions: `var_exprs = [pl.col(c).var().alias(c) for c in numeric_cols]`.
  4. Select `active_features` with variance > `variance_threshold` (reducing 3,900+ columns down to informative attributes).
  5. Compute medians for active features lazily and impute missing values (`fill_null(self.medians[c])`).
  6. Materialize cleaned active features and cast target vector (`0=Legitimate, 1=Mule`).
  7. Apply `RobustScaler().fit_transform(X)` to withstand banking transaction outliers.
  8. Save preprocessed matrix directly to Parquet format via Polars/PyArrow (`output_parquet`).
- `save(path)` and `load(path)`: Methods to persist and reload fitted preprocessor state (`active_features`, `medians`, `scaler`) using pickle for production inference.
- Execution entrypoint / `__main__` block to run on `data/DataSet.csv` -> `data/processed_data.parquet` and print summary statistics.

### Run & Verification Script

#### [NEW] [run_pipeline.py](file:///c:/Golden_Hour_Hold/run_pipeline.py)
- Standalone runner script to execute the preprocessing engine on `data/DataSet.csv`, save `data/processed_data.parquet` and `data/preprocessor.pkl`, and output:
  - Total columns scanned vs. active features retained.
  - Imputation stats and null check verification.
  - Class distribution of target vector (`F3924`: 0=Legitimate, 1=Mule).
  - Shape and memory usage of the resulting Parquet file.

## Verification Plan

### Automated Tests
- Run `run_pipeline.py` using `.venv\Scripts\python.exe`:
  ```powershell
  .venv\Scripts\python.exe run_pipeline.py
  ```
- Validate that:
  - Input: `data/DataSet.csv` (116 MB, 3,925 columns, 9,083 rows).
  - Output: `data/processed_data.parquet` created successfully.
  - Number of selected active features is reduced from 3,900+ down to ~150–300.
  - Zero null values remaining in the transformed matrix.
  - Target vector is preserved with binary values 0 and 1.
  - Preprocessor state can be pickled and unpickled cleanly.
