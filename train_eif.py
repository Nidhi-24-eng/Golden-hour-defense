"""
Execution and verification runner for Phase 6: Extended Isolation Forest (Zero-Day Safety Net).
"""
import json
import os
import sys
import time
import numpy as np
import polars as pl
from src.models.eif_detector import ZeroDayAnomalyDetector


def main():
    parquet_path = os.path.join("data", "processed_data.parquet")
    model_output_path = os.path.join("models", "eif_mule_detector.pkl")
    metrics_output_path = os.path.join("models", "eif_metrics.json")
    preds_output_path = os.path.join("data", "oof_eif_preds.npy")

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Processed parquet dataset not found at: {parquet_path}. "
            "Please run run_pipeline.py first."
        )

    print("=" * 70)
    print("STAGE 5: EXTENDED ISOLATION FOREST (ZERO-DAY SAFETY NET) TRAINING")
    print("=" * 70)

    start_time = time.time()

    # 1. Ingest preprocessed tabular data
    print(f"[*] Ingesting preprocessed data from: {parquet_path}")
    df = pl.read_parquet(parquet_path)

    target_col = "3924" if "3924" in df.columns else ("F3924" if "F3924" in df.columns else df.columns[-1])
    feature_cols = [c for c in df.columns if c != target_col]

    X = df.select(feature_cols).to_numpy()
    y = df[target_col].to_numpy().astype(np.int32)

    # 2. Extract baseline legitimate accounts (unsupervised training on normal behavior)
    legit_mask = (y == 0)
    X_baseline = X[legit_mask]
    print(f"[*] Training on {X_baseline.shape[0]:,} legitimate baseline customer profiles (zero supervision)")

    # 3. Instantiate and train Anomaly Detector
    detector = ZeroDayAnomalyDetector(
        ntrees=250,
        sample_size=256,
        extension_level=2,
        random_state=42,
    )
    detector.fit(X_baseline)

    # 4. Evaluate anomaly isolation power across the full population
    print("[*] Evaluating anomaly separation across the full banking population...")
    metrics = detector.evaluate_against_labels(X, y)

    # 5. Save model checkpoint and metrics
    detector.save(model_path=model_output_path, metrics_path=metrics_output_path)

    # 6. Save out-of-fold/population anomaly predictions for Phase 7 (Late Fusion)
    if detector.predictions is not None:
        os.makedirs(os.path.dirname(preds_output_path), exist_ok=True)
        np.save(preds_output_path, detector.predictions)
        print(f"[+] Zero-day anomaly prediction vector saved -> {preds_output_path}")

    elapsed_time = time.time() - start_time

    # 7. Display benchmark report
    print("\n" + "=" * 70)
    print("ZERO-DAY ANOMALY DETECTOR BENCHMARK REPORT")
    print("=" * 70)
    print(f"[*] Total Training & Scoring Time:     {elapsed_time:.2f} seconds")
    print(f"[*] Engine Deployed:                   {metrics['engine_used']}")
    print(f"[*] Unsupervised AUPRC:                {metrics['unsupervised_auprc']:.4f}")
    print(f"[*] Unsupervised ROC-AUC:              {metrics['unsupervised_roc_auc']:.4f}")
    print(f"[*] Precision@50  (Top 50 anomalies):  {metrics['precision_at_50'] * 100:.1f}%")
    print(f"[*] Precision@100 (Top 100 anomalies): {metrics['precision_at_100'] * 100:.1f}%")
    print(f"[*] Precision@200 (Top 200 anomalies): {metrics['precision_at_200'] * 100:.1f}%")
    print(f"[*] Mule Accounts Mean Anomaly Score:  {metrics['mule_mean_anomaly_score']:.4f}")
    print(f"[*] Legit Accounts Mean Anomaly Score: {metrics['legit_mean_anomaly_score']:.4f}")
    print(f"[*] Statistical Separation Delta:      {metrics['separation_delta']:+.4f}")
    print("=" * 70)
    print("PHASE 6 ZERO-DAY ANOMALY DETECTOR COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
