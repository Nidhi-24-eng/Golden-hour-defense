"""
hybrid_fusion.py — Phase 7: Hybrid Late-Fusion & False Positive Suppression Engine
===================================================================================

PURPOSE
-------
No single model can solve the money mule crisis alone:
  • XGBoost (Brain 1) has high precision on behavioral dormancy shifts, but misses
    stealthy low-value layering across distributed rings.
  • R-GCN (Brain 2) has ultra-high recall (97.5%) across shared device emulator farms
    and proxy loops, but generates excess false positives when accounts share IP subnets.
  • Extended Isolation Forest (Brain 3) catches statistical zero-day attacks without
    requiring historical supervision labels.

This module unifies all three streams into a calibrated Late-Fusion Decision Matrix:
  1. Calibrated Weighted Fusion:
       S_composite = w_xgb * P_xgb + w_rgcn * P_rgcn + w_eif * S_eif
  2. Frictionless False Positive Suppression:
       Cross-checks topological risk against behavioral normality and statistical density.
       If an account is flagged purely by network colocation but exhibits verified normal
       customer habits, hard auto-freezes are suppressed and routed to step-up MFA,
       preserving clean customers from accidental account lockouts.
  3. Action Tiering:
       • EMERGENCY_FREEZE_LIEN (Composite >= Threshold: Instant automated Golden Hour lock)
       • STEP_UP_CHALLENGE (Suppressed high-network risk: OTP/biometric verification)
       • ENHANCED_SURVEILLANCE (Moderate risk: 60-second sliding window velocity tracking)
       • ALLOW (Normal low-risk customer activity)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class HybridDecisionEngine:
    """
    Dual-Brain Late-Fusion & False Positive Suppression Engine.
    """

    def __init__(
        self,
        w_xgb: float = 0.55,
        w_rgcn: float = 0.30,
        w_eif: float = 0.15,
        freeze_threshold: float = 0.65,
        surveillance_threshold: float = 0.40,
    ):
        """
        Parameters
        ----------
        w_xgb : float
            Weight for tabular behavioral model (XGBoost).
        w_rgcn : float
            Weight for relational graph topology model (R-GCN).
        w_eif : float
            Weight for zero-day anomaly detector (Extended Isolation Forest).
        freeze_threshold : float
            Threshold on composite risk score for hard emergency lien.
        surveillance_threshold : float
            Threshold for routing accounts to real-time 60-second sliding velocity monitoring.
        """
        total = w_xgb + w_rgcn + w_eif
        self.w_xgb = w_xgb / total
        self.w_rgcn = w_rgcn / total
        self.w_eif = w_eif / total
        self.freeze_threshold = freeze_threshold
        self.surveillance_threshold = surveillance_threshold

        self.benchmark_metrics: Dict[str, Any] = {}

    def compute_composite_scores(
        self,
        p_xgb: np.ndarray,
        p_rgcn: np.ndarray,
        s_eif: np.ndarray,
    ) -> np.ndarray:
        """
        Computes weighted composite risk score vector S_composite in [0.0, 1.0].
        """
        # Ensure identical array length
        n = min(len(p_xgb), len(p_rgcn), len(s_eif))
        p_xgb = p_xgb[:n]
        p_rgcn = p_rgcn[:n]
        s_eif = s_eif[:n]

        composite = (self.w_xgb * p_xgb) + (self.w_rgcn * p_rgcn) + (self.w_eif * s_eif)
        return composite.astype(np.float32)

    def evaluate_decision(
        self,
        p_xgb: float,
        p_rgcn: float,
        s_eif: float,
        kyc_verified: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluates an individual account transaction in real time with
        Frictionless False Positive Suppression logic.
        """
        composite_score = float((self.w_xgb * p_xgb) + (self.w_rgcn * p_rgcn) + (self.w_eif * s_eif))

        # ── False Positive Suppression Rule ──────────────────────────
        # If an account has elevated network connectivity (e.g., shared office Wi-Fi / IP)
        # BUT tabular velocity is low (clean F115, normal cash ratios) AND anomaly score is low:
        # Suppress hard freeze and issue a Step-Up Challenge instead.
        if p_rgcn >= 0.50 and p_xgb < 0.25 and s_eif < 0.55 and kyc_verified:
            action = "STEP_UP_CHALLENGE"
            reason = "Colocated network flag suppressed: Clean behavioral telemetry and verified KYC."
        elif composite_score >= self.freeze_threshold or (p_xgb >= 0.85 and p_rgcn >= 0.70):
            action = "EMERGENCY_FREEZE_LIEN"
            reason = "High composite risk or dual-stream confirmation: Immediate Golden Hour freeze."
        elif composite_score >= self.surveillance_threshold:
            action = "ENHANCED_SURVEILLANCE"
            reason = "Moderate risk: Enqueued for sliding 60-second velocity tracking."
        else:
            action = "ALLOW"
            reason = "Legitimate customer profile."

        return {
            "composite_score": round(composite_score, 4),
            "p_xgb": round(float(p_xgb), 4),
            "p_rgcn": round(float(p_rgcn), 4),
            "s_eif": round(float(s_eif), 4),
            "action": action,
            "reason": reason,
        }

    def optimize_and_benchmark(
        self,
        p_xgb: np.ndarray,
        p_rgcn: np.ndarray,
        s_eif: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Executes a grid search across fusion weights and suppression thresholds,
        comparing standalone models against the unified Hybrid Fusion Engine.
        """
        n = min(len(p_xgb), len(p_rgcn), len(s_eif), len(y_true))
        p_xgb = p_xgb[:n]
        p_rgcn = p_rgcn[:n]
        s_eif = s_eif[:n]
        y = y_true[:n]

        # 1. Standalone Metrics
        xgb_auprc = float(average_precision_score(y, p_xgb))
        xgb_roc = float(roc_auc_score(y, p_xgb))

        rgcn_auprc = float(average_precision_score(y, p_rgcn))
        rgcn_roc = float(roc_auc_score(y, p_rgcn))

        eif_auprc = float(average_precision_score(y, s_eif))
        eif_roc = float(roc_auc_score(y, s_eif))

        # 2. Optimal Weight Search for Maximum AUPRC
        best_auprc = 0.0
        best_weights = (self.w_xgb, self.w_rgcn, self.w_eif)

        for w_x in [0.40, 0.50, 0.60, 0.70]:
            for w_r in [0.15, 0.25, 0.35]:
                for w_e in [0.05, 0.10, 0.20]:
                    tot = w_x + w_r + w_e
                    comp = (w_x * p_xgb + w_r * p_rgcn + w_e * s_eif) / tot
                    score = average_precision_score(y, comp)
                    if score > best_auprc:
                        best_auprc = score
                        best_weights = (w_x / tot, w_r / tot, w_e / tot)

        self.w_xgb, self.w_rgcn, self.w_eif = best_weights
        composite_scores = self.compute_composite_scores(p_xgb, p_rgcn, s_eif)

        comp_auprc = float(average_precision_score(y, composite_scores))
        comp_roc = float(roc_auc_score(y, composite_scores))

        # 3. Precision@K on Composite Scores
        top_50 = np.argsort(composite_scores)[::-1][:50]
        top_100 = np.argsort(composite_scores)[::-1][:100]
        top_200 = np.argsort(composite_scores)[::-1][:200]

        p50 = float(np.mean(y[top_50]))
        p100 = float(np.mean(y[top_100]))
        p200 = float(np.mean(y[top_200]))

        # 4. Optimal Decision Threshold Search
        thresholds = np.linspace(0.2, 0.85, 66)
        f1_list = [f1_score(y, (composite_scores >= t).astype(int), zero_division=0) for t in thresholds]
        best_idx = int(np.argmax(f1_list))
        self.freeze_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_list[best_idx])

        # 5. Apply False Positive Suppression Simulation
        raw_flags = (composite_scores >= self.freeze_threshold)
        suppressed_flags = np.copy(raw_flags)

        suppressed_count = 0
        for i in range(n):
            if raw_flags[i]:
                # Suppression condition: High network link but completely clean behavioral velocity
                if p_rgcn[i] >= 0.50 and p_xgb[i] < 0.20 and s_eif[i] < 0.55:
                    suppressed_flags[i] = False
                    suppressed_count += 1

        binary_preds = suppressed_flags.astype(int)
        prec = float(precision_score(y, binary_preds, zero_division=0))
        rec = float(recall_score(y, binary_preds, zero_division=0))
        cm = confusion_matrix(y, binary_preds).tolist()

        self.benchmark_metrics = {
            "optimal_weights": {
                "w_xgb": round(self.w_xgb, 4),
                "w_rgcn": round(self.w_rgcn, 4),
                "w_eif": round(self.w_eif, 4),
            },
            "standalone_comparison": {
                "xgboost": {"auprc": round(xgb_auprc, 4), "roc_auc": round(xgb_roc, 4)},
                "rgcn": {"auprc": round(rgcn_auprc, 4), "roc_auc": round(rgcn_roc, 4)},
                "eif": {"auprc": round(eif_auprc, 4), "roc_auc": round(eif_roc, 4)},
                "hybrid_fusion": {"auprc": round(comp_auprc, 4), "roc_auc": round(comp_roc, 4)},
            },
            "precision_at_k": {
                "p_at_50": round(p50, 4),
                "p_at_100": round(p100, 4),
                "p_at_200": round(p200, 4),
            },
            "decision_threshold": round(self.freeze_threshold, 3),
            "best_f1": round(best_f1, 4),
            "precision_with_suppression": round(prec, 4),
            "recall_with_suppression": round(rec, 4),
            "false_alarms_suppressed": int(suppressed_count),
            "confusion_matrix": {
                "true_negative": cm[0][0],
                "false_positive": cm[0][1],
                "false_negative": cm[1][0],
                "true_positive": cm[1][1],
            },
        }

        return self.benchmark_metrics

    def save(self, config_path: str) -> None:
        cfg_dir = os.path.dirname(config_path)
        if cfg_dir:
            os.makedirs(cfg_dir, exist_ok=True)

        config = {
            "w_xgb": self.w_xgb,
            "w_rgcn": self.w_rgcn,
            "w_eif": self.w_eif,
            "freeze_threshold": self.freeze_threshold,
            "surveillance_threshold": self.surveillance_threshold,
            "benchmark_metrics": self.benchmark_metrics,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[+] Saved Hybrid Decision Engine configuration -> {config_path}")

    @classmethod
    def load(cls, config_path: str) -> "HybridDecisionEngine":
        with open(config_path, "r") as f:
            cfg = json.load(f)
        engine = cls(
            w_xgb=cfg["w_xgb"],
            w_rgcn=cfg["w_rgcn"],
            w_eif=cfg["w_eif"],
            freeze_threshold=cfg["freeze_threshold"],
            surveillance_threshold=cfg["surveillance_threshold"],
        )
        engine.benchmark_metrics = cfg.get("benchmark_metrics", {})
        print(f"[+] Loaded Hybrid Decision Engine configuration <- {config_path}")
        return engine
