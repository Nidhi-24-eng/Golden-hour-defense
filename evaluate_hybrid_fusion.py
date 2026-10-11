"""
Execution and verification runner for Phase 7: Hybrid Late-Fusion & False Positive Suppression Engine.
"""
import json
import os
import sys
import time
import numpy as np
import polars as pl
from src.models.hybrid_fusion import HybridDecisionEngine


def main():
    parquet_path = os.path.join("data", "processed_data.parquet")
    xgb_preds_path = os.path.join("data", "oof_xgboost_preds.npy")
    rgcn_preds_path = os.path.join("data", "oof_rgcn_preds.npy")
    eif_preds_path = os.path.join("data", "oof_eif_preds.npy")
    config_output_path = os.path.join("models", "hybrid_fusion_config.json")

    # Verification checks
    for p, name in [
        (parquet_path, "Processed Parquet"),
        (xgb_preds_path, "XGBoost OOF Predictions"),
        (rgcn_preds_path, "R-GCN Predictions"),
        (eif_preds_path, "EIF Anomaly Predictions"),
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required artifact: {name} at {p}")

    print("=" * 70)
    print("STAGE 6: HYBRID LATE-FUSION & FALSE POSITIVE SUPPRESSION BENCHMARK")
    print("=" * 70)

    start_time = time.time()

    # 1. Load ground truth labels
    df = pl.read_parquet(parquet_path)
    target_col = "3924" if "3924" in df.columns else ("F3924" if "F3924" in df.columns else df.columns[-1])
    y = df[target_col].to_numpy().astype(np.int32)

    # 2. Load prediction arrays from all three brains
    p_xgb = np.load(xgb_preds_path)
    p_rgcn = np.load(rgcn_preds_path)
    s_eif = np.load(eif_preds_path)

    # Ensure length alignment
    n = min(len(y), len(p_xgb), len(p_rgcn), len(s_eif))
    y = y[:n]
    p_xgb = p_xgb[:n]
    p_rgcn = p_rgcn[:n]
    s_eif = s_eif[:n]

    print(f"[*] Ingested predictions across {n:,} accounts:")
    print(f"    - Brain 1 (Tabular XGBoost):            mean={np.mean(p_xgb):.4f}, max={np.max(p_xgb):.4f}")
    print(f"    - Brain 2 (Relational PyG R-GCN):       mean={np.mean(p_rgcn):.4f}, max={np.max(p_rgcn):.4f}")
    print(f"    - Brain 3 (Zero-Day Isolation Forest):  mean={np.mean(s_eif):.4f}, max={np.max(s_eif):.4f}")

    # 3. Instantiate Engine and Run Optimization & Benchmarking
    engine = HybridDecisionEngine()
    metrics = engine.optimize_and_benchmark(
        p_xgb=p_xgb,
        p_rgcn=p_rgcn,
        s_eif=s_eif,
        y_true=y,
    )

    # 4. Save configuration and metrics
    engine.save(config_output_path)

    elapsed_time = time.time() - start_time

    # 5. Display Comprehensive Comparison Matrix
    print("\n" + "=" * 70)
    print("HYBRID IMMUNE SYSTEM: MODEL COMPARISON MATRIX")
    print("=" * 70)
    print(f"{'Model Component':<30} | {'AUPRC':<10} | {'ROC-AUC':<10} | {'Role':<20}")
    print("-" * 75)
    
    comp = metrics["standalone_comparison"]
    print(f"{'Brain 1: Tabular XGBoost':<30} | {comp['xgboost']['auprc']:<10.4f} | {comp['xgboost']['roc_auc']:<10.4f} | {'Behavioral Velocity'}")
    print(f"{'Brain 2: PyG Relational GCN':<30} | {comp['rgcn']['auprc']:<10.4f} | {comp['rgcn']['roc_auc']:<10.4f} | {'Network Topology'}")
    print(f"{'Brain 3: Extended IsoForest':<30} | {comp['eif']['auprc']:<10.4f} | {comp['eif']['roc_auc']:<10.4f} | {'Zero-Day Safety Net'}")
    print("-" * 75)
    print(f"{'HYBRID FUSION ENGINE':<30} | {comp['hybrid_fusion']['auprc']:<10.4f} | {comp['hybrid_fusion']['roc_auc']:<10.4f} | {'Unified Immune System'}")
    print("=" * 70)

    opt_w = metrics["optimal_weights"]
    print(f"\n[*] Calibrated Fusion Weights: w_XGB={opt_w['w_xgb']} | w_RGCN={opt_w['w_rgcn']} | w_EIF={opt_w['w_eif']}")
    print(f"[*] Optimal Composite Threshold:       {metrics['decision_threshold']}")
    print(f"[*] Precision@50  (Top 50 accounts):   {metrics['precision_at_k']['p_at_50'] * 100:.1f}%")
    print(f"[*] Precision@100 (Top 100 accounts):  {metrics['precision_at_k']['p_at_100'] * 100:.1f}%")
    print(f"[*] Precision@200 (Top 200 accounts):  {metrics['precision_at_k']['p_at_200'] * 100:.1f}%")

    print("\n" + "=" * 70)
    print("FRICTIONLESS FALSE POSITIVE SUPPRESSION REPORT")
    print("=" * 70)
    print(f"[*] Clean Accounts Saved From Hard Lockouts: {metrics['false_alarms_suppressed']:,} accounts")
    print(f"[*] Post-Suppression Precision / Recall:     {metrics['precision_with_suppression'] * 100:.1f}% / {metrics['recall_with_suppression'] * 100:.1f}%")
    print(f"[*] Post-Suppression F1-Score:               {metrics['best_f1']:.4f}")

    cm = metrics["confusion_matrix"]
    print("\n[*] Final Operational Confusion Matrix:")
    print(f"    - True Negatives  (Normal banking undisturbed): {cm['true_negative']:,}")
    print(f"    - False Positives (Residual false alerts):       {cm['false_positive']:,}")
    print(f"    - False Negatives (Missed mules):               {cm['false_negative']:,}")
    print(f"    - True Positives  (Mules neutralized):          {cm['true_positive']:,}")

    # 6. Test Single Transaction Evaluation Simulation
    print("\n" + "=" * 70)
    print("REAL-TIME TRANSACTION INFERENCE TIERING TEST")
    print("=" * 70)
    
    # Test Scenario A: Organized Mule Syndicate Attack
    res_a = engine.evaluate_decision(p_xgb=0.91, p_rgcn=0.88, s_eif=0.75, kyc_verified=True)
    print("[1] Test Case: High Velocity + Syndicate Colocation")
    print(f"    Composite: {res_a['composite_score']} -> Action: {res_a['action']}")
    print(f"    Reason: {res_a['reason']}")

    # Test Scenario B: Legitimate User on Shared Office Wi-Fi
    res_b = engine.evaluate_decision(p_xgb=0.08, p_rgcn=0.72, s_eif=0.31, kyc_verified=True)
    print("\n[2] Test Case: Clean History + Shared Network Link (Suppression Triggered)")
    print(f"    Composite: {res_b['composite_score']} -> Action: {res_b['action']}")
    print(f"    Reason: {res_b['reason']}")

    # Test Scenario C: Zero-Day Attack Pattern
    res_c = engine.evaluate_decision(p_xgb=0.45, p_rgcn=0.20, s_eif=0.89, kyc_verified=False)
    print("\n[3] Test Case: Novel Unseen Cadence (EIF Zero-Day Triggered)")
    print(f"    Composite: {res_c['composite_score']} -> Action: {res_c['action']}")
    print(f"    Reason: {res_c['reason']}")

    print("\n" + "=" * 70)
    print(f"PHASE 7 HYBRID LATE-FUSION COMPLETE in {elapsed_time:.2f}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
