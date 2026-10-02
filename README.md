# Golden Hour Hold: High-Performance Money Mule Detection System

An enterprise-grade, multi-stage machine learning system designed to detect and neutralize illicit money mule accounts in real-time banking pipelines during the critical "Golden Hour" of fraudulent transactions.

---

## Architecture Overview

```
                      +---------------------------------------+
                      | Raw High-Dimensional Banking Data     |
                      | (3,925 features, 9,082 transactions)  |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Stage 1: Ingestion & Preprocessing    |
                      | - Polars LazyFrame Zero-Copy Scanning |
                      | - Lazy Variance Threshold Filtering   |
                      | - Robust Median Imputation            |
                      | - RobustScaler Outlier Mitigation     |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Stage 2: Multi-Relational Graph Engine|
                      | - Shared Device Graph (228k edges)    |
                      | - Shared Proxy IP Graph (318k edges)  |
                      | - PyTorch Geometric Topology (555k)   |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Stage 3: XGBoost Behavioral Ensemble  |
                      | - Dynamic `scale_pos_weight` (111.12) |
                      | - Stratified 5-Fold Cross-Validation  |
                      | - Real-time Precision@K Optimization  |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Alert Triage & Production Checkpoints |
                      | - Precision@50: 74.0%                 |
                      | - ROC-AUC: 0.9286 | AUPRC: 0.6092     |
                      +---------------------------------------+
```

---

## Performance & Benchmark Results

Trained on an extremely imbalanced production banking dataset (**81 Mules vs. 9,001 Legitimate accounts**, 0.89% positive rate).

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Stratified 5-Fold Mean AUPRC** | **0.6324** (+/- 0.2224) | Area Under Precision-Recall Curve |
| **Overall Out-of-Fold AUPRC** | **0.6092** | Robust out-of-fold generalization |
| **Overall Out-of-Fold ROC-AUC** | **0.9286** | Excellent class separability |
| **Precision@50 (Top 50 Alerts)** | **74.0%** | 37 confirmed mules in top 50 flagged accounts |
| **Precision@100 (Top 100 Alerts)** | **48.0%** | Rapid fraud analyst queue prioritization |
| **Optimal Threshold** | **0.74** | Selected via F1 maximization |
| **F1-Score at Optimal Threshold** | **0.6154** | Balanced precision & recall |
| **Precision / Recall at Threshold** | **70.97% / 54.32%** | High alert confidence with low false positive rate |

### Confusion Matrix at Production Threshold (0.74)
- **True Negatives** (Legitimate accounts preserved): `8,983`
- **False Positives** (False alarms): `18`
- **False Negatives** (Missed mules): `37`
- **True Positives** (Mules successfully neutralized): `44`

---

## Big Data & Storage Policy

GitHub enforces a **100 MB hard limit** per file and warns on files exceeding **50 MB**.
To maintain a clean and lightweight repository:
- Raw transaction dumps (`data/DataSet.csv`, ~116.5 MB) are **git-ignored** and omitted from source control.
- Intermediate binaries (`.parquet`, `.npz`, `.pt`) are generated dynamically via reproduction scripts in seconds.
- Production model checkpoints and validation metrics (`models/`) are saved in lightweight JSON format and version-controlled.

---

## Repository Structure

```
Golden_Hour_Hold/
├── data/
│   ├── graphs/
│   │   ├── .gitkeep
│   │   └── node_mapping.json           # Node identifier mappings
│   └── .gitkeep
├── models/
│   ├── xgboost_metrics.json            # Cross-validation benchmark report
│   └── xgboost_mule_detector.json      # Production XGBoost checkpoint
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── xgboost_model.py            # XGBoost behavioral ensemble engine
│   └── preprocessing/
│       ├── __init__.py
│       ├── graph_builder.py            # PyTorch Geometric relational builder
│       └── polars_pipeline.py          # High-performance Polars LazyFrame preprocessor
├── .gitattributes                      # Git line ending configuration
├── .gitignore                          # Big data exclusion rules
├── requirements.txt                    # Project dependencies
├── run_pipeline.py                     # Stage 1 data pipeline runner
├── train_xgboost.py                    # Stage 3 model training runner
└── README.md
```

---

## Quickstart & Reproduction

### 1. Environment Setup

Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/Nidhi-24-eng/Golden-hour-hold.git
cd Golden-hour-hold

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Add Dataset
Place your raw dataset file `DataSet.csv` into the `data/` directory:
```
data/DataSet.csv
```

### 3. Run Preprocessing Pipeline (Stage 1)
```bash
python run_pipeline.py 250
```
*Processes the 3,900+ raw features using Polars LazyFrames, applies variance reduction down to 250 features, median imputation, and robust scaling in seconds.*

### 4. Train Behavioral Ensemble (Stage 3)
```bash
python train_xgboost.py
```
*Constructs the multi-relational graph topology, executes Stratified 5-Fold Cross Validation with `scale_pos_weight=111.12`, logs out-of-fold performance, and serializes the production model checkpoint to `models/`.*

---

## License
MIT License
