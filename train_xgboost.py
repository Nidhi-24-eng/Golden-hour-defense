"""
Execution and verification runner for Phase 4: Tabular XGBoost Behavioral Ensemble.
"""
import json
import os
import sys
import time
import numpy as np
import polars as pl
from src.models.xgboost_model import TabularMuleClassifier


def main():
    parquet_path = os.path.join("data", "processed_data.parquet")
    model_output_path = os.path.join("models", "xgboost_mule_detector.json")
    metrics_output_path = os.path.join("models", "xgboost_metrics.json")
    oof_output_path = os.path.join("data", "oof_xgboost_preds.npy")

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Processed parquet dataset not found at: {parquet_path}. "
            "Please run run_pipeline.py first."
        )

    print("=" * 70)
    print("STAGE 3: TABULAR XGBOOST BEHAVIORAL ENSEMBLE TRAINING & EVALUATION")
    print("=" * 70)

    start_time = time.time()

    # 1. Read cleaned parquet data
    print(f"[*] Ingesting preprocessed data from: {parquet_path}")
    df = pl.read_parquet(parquet_path)
    
    # Resolve target column
    target_col = "3924" if "3924" in df.columns else ("F3924" if "F3924" in df.columns else df.columns[-1])
    feature_cols = [c for c in df.columns if c != target_col]

    X = df.select(feature_cols).to_numpy()
    y = df[target_col].to_numpy().astype(np.int32)

    # 2. Instantiate and train model
    classifier = TabularMuleClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
    )

    metrics = classifier.train_cross_validated(
        X=X,
        y=y,
        feature_names=feature_cols,
        n_splits=5,
        early_stopping_rounds=35,
    )

    # 3. Save model checkpoint and evaluation metrics
    classifier.save(model_path=model_output_path, metrics_path=metrics_output_path)

    # 4. Save out-of-fold predictions for Phase 7 (Late Fusion)
    if classifier.oof_predictions is not None:
        os.makedirs(os.path.dirname(oof_output_path), exist_ok=True)
        np.save(oof_output_path, classifier.oof_predictions)
        print(f"[+] Out-of-fold probability vector saved -> {oof_output_path}")

    elapsed_time = time.time() - start_time

    # 5. Display comprehensive evaluation report
    print("\n" + "=" * 70)
    print("XGBOOST VALIDATION PERFORMANCE & BENCHMARK REPORT")
    print("=" * 70)
    print(f"[*] Total Training & Evaluation Time: {elapsed_time:.2f} seconds")
    print(f"[*] Stratified 5-Fold Mean AUPRC:     {metrics['mean_cv_auprc']:.4f} (+/- {metrics['std_cv_auprc']:.4f})")
    print(f"[*] Overall Out-of-Fold AUPRC:        {metrics['overall_oof_auprc']:.4f}")
    print(f"[*] Overall Out-of-Fold ROC-AUC:      {metrics['overall_oof_roc_auc']:.4f}")
    print(f"[*] Precision@50  (Top 50 alerts):    {metrics['precision_at_50'] * 100:.1f}%")
    print(f"[*] Precision@100 (Top 100 alerts):   {metrics['precision_at_100'] * 100:.1f}%")
    print(f"[*] Precision@200 (Top 200 alerts):   {metrics['precision_at_200'] * 100:.1f}%")
    print(f"[*] Optimal Decision Threshold:       {metrics['optimal_threshold']}")
    print(f"[*] F1-Score at Optimal Threshold:    {metrics['best_f1_score']:.4f}")
    print(f"[*] Precision / Recall at Threshold:  {metrics['precision_at_optimal']:.4f} / {metrics['recall_at_optimal']:.4f}")

    cm = metrics["confusion_matrix"]
    print("\n[*] Confusion Matrix at Optimal Threshold:")
    print(f"    - True Negatives  (Legitimate preserved): {cm['true_negative']:,}")
    print(f"    - False Positives (False alarms):        {cm['false_positive']:,}")
    print(f"    - False Negatives (Missed mules):        {cm['false_negative']:,}")
    print(f"    - True Positives  (Mules neutralized):   {cm['true_positive']:,}")

    print("\n[*] Top 10 Most Predictive Behavioral Features:")
    for item in metrics["top_15_features"][:10]:
        print(f"    {item['rank']:02d}. Feature {item['feature']:<8} (Importance Weight: {item['importance']:.4f})")

    print("\n" + "=" * 70)
    print("PHASE 4 TABULAR XGBOOST ENSEMBLE COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
