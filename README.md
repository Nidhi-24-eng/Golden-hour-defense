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
        +---------------------------------+---------------------------------+
        |                                 |                                 |
        v                                 v                                 v
+---------------+                 +---------------+                 +---------------+
|    Brain 1    |                 |    Brain 2    |                 |    Brain 3    |
| Tabular XGB   |                 | PyG Relational|                 |  Zero-Day EIF |
| (Behavioral)  |                 |      GCN      |                 | (Unsupervised)|
+-------+-------+                 +-------+-------+                 +-------+-------+
        | (w=0.5882)                      | (w=0.1765)                      | (w=0.2353)
        +---------------------------------+---------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Stage 6: Hybrid Late-Fusion Engine    |
                      | - Calibrated Dynamic Risk Weighted Sum|
                      | - False Positive Suppression Rules    |
                      | - Tiered Action Matrix                |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Real-Time Tiered Enforcement:         |
                      | - EMERGENCY_FREEZE_LIEN (>= 0.75)     |
                      | - STEP_UP_CHALLENGE (Suppressed)      |
                      | - ENHANCED_SURVEILLANCE (0.45 - 0.75) |
                      | - ALLOW_NORMAL (< 0.45)               |
                      +---------------------------------------+
```

---

## Hybrid Immune System: Model Comparison Matrix

Trained on an extremely imbalanced production banking dataset (**81 Mules vs. 9,001 Legitimate accounts**, 0.89% positive rate across 9,082 entities).

| Model Component | AUPRC | ROC-AUC | Primary Operational Role |
| :--- | :--- | :--- | :--- |
| **Brain 1: Tabular XGBoost** | **0.6092** | **0.9286** | Behavioral velocity & transactional bursts |
| **Brain 2: PyG Relational GCN** | **0.0088** | **0.5010** | Syndicate co-location & topological graph clustering |
| **Brain 3: Extended IsoForest (EIF)** | **0.0355** | **0.8310** | Zero-day novel fraud patterns & unlabelled outliers |
| **HYBRID FUSION ENGINE** | **0.6099** | **0.9324** | **Unified immune system with false positive suppression** |

### Benchmark Highlights:
- **Optimal Composite Threshold**: `0.75`
- **Precision@50 (Top 50 Alerts)**: `74.0%` (37 confirmed mules in top 50 flagged accounts)
- **Precision@100 (Top 100 Alerts)**: `48.0%`
- **Precision@200 (Top 200 Alerts)**: `30.0%`
- **Post-Suppression Precision / Recall**: `69.8% / 54.3%`
- **Post-Suppression F1-Score**: `0.6111`
- **Operational Confusion Matrix**:
  - **True Negatives** (Normal banking undisturbed): `8,982`
  - **False Positives** (Residual alerts): `19`
  - **False Negatives** (Missed mules): `37`
  - **True Positives** (Mules neutralized): `44`

---

## Frictionless Tiered Action Policy

The late-fusion decision engine categorizes accounts into granular operational risk tiers during the Golden Hour:

1. **EMERGENCY_FREEZE_LIEN** (`Composite >= 0.75`): Immediate hold applied to account disbursements.
2. **STEP_UP_CHALLENGE**: Triggered when high network connectivity is mitigated by pristine behavioral telemetry and KYC history.
3. **ENHANCED_SURVEILLANCE** (`0.45 <= Composite < 0.75`): Enqueued for sliding 60-second velocity tracking.
4. **ALLOW_NORMAL** (`Composite < 0.45`): Unimpeded frictionless customer transaction.

---

## Big Data & Storage Policy

GitHub enforces a **100 MB hard limit** per file and warns on files exceeding **50 MB**.
To maintain a clean and lightweight repository:
- Raw transaction dumps (`data/DataSet.csv`, ~116.5 MB) and generated tabular matrices are **git-ignored**.
- Heavy binary model checkpoints (e.g. `eif_mule_detector.pkl`, ~1.17 GB) are generated locally via reproduction scripts and excluded from version control.
- Lightweight model weights, graph topologies, evaluation metrics, and hybrid fusion configs (`models/*.json`) are tracked in source control.

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
│   ├── .gitkeep
│   ├── eif_metrics.json                # Brain 3 EIF evaluation report
│   ├── hybrid_fusion_config.json       # Calibrated fusion weights & policy
│   ├── rgcn_metrics.json               # Brain 2 Relational GCN report
│   ├── xgboost_metrics.json            # Brain 1 XGBoost benchmark report
│   └── xgboost_mule_detector.json      # Production XGBoost checkpoint
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── eif_detector.py             # Brain 3: Extended Isolation Forest
│   │   ├── hybrid_fusion.py            # Stage 6: Late-Fusion & Suppression Engine
│   │   ├── rgcn_model.py               # Brain 2: Relational Graph Convolutional Network
│   │   └── xgboost_model.py            # Brain 1: XGBoost Behavioral Ensemble
│   └── preprocessing/
│       ├── __init__.py
│       ├── graph_builder.py            # PyTorch Geometric relational builder
│       └── polars_pipeline.py          # High-performance Polars preprocessor
├── .gitattributes                      # Git line ending configuration
├── .gitignore                          # Big data & model binary exclusion rules
├── evaluate_hybrid_fusion.py           # Stage 6 multi-brain benchmark runner
├── requirements.txt                    # Project dependencies
├── run_pipeline.py                     # Stage 1 data pipeline runner
├── train_eif.py                        # Brain 3 training script
├── train_rgcn.py                       # Brain 2 training script
├── train_xgboost.py                    # Brain 1 training script
└── README.md
```

---

## Quickstart & Reproduction

### 1. Environment Setup

```bash
git clone https://github.com/Nidhi-24-eng/Golden-hour-defense.git
cd Golden-hour-defense

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Ingestion Pipeline (Stage 1)
```bash
python run_pipeline.py 250
```

### 3. Train the 3 Defense Brains
```bash
# Brain 1: Behavioral Velocity (XGBoost)
python train_xgboost.py

# Brain 2: Relational Topology (PyG R-GCN)
python train_rgcn.py

# Brain 3: Zero-Day Anomaly Detection (Extended Isolation Forest)
python train_eif.py
```

### 4. Evaluate Stage 6 Hybrid Late-Fusion Engine
```bash
python evaluate_hybrid_fusion.py
```

### 5. Benchmark Stateful Velocity Engine (Stage 7)
```bash
python run_streaming_benchmark.py
```

### 6. Test I4C 1930 Helpline Emergency Brake (Stage 8)
```bash
python test_i4c_webhook.py
```

### 7. Launch Interactive SOC Dashboard (Stage 9)
```bash
python run_dashboard.py
# Open http://127.0.0.1:8000 in your browser
```

---

## License
MIT License

