"""
eif_detector.py — Phase 6: Extended Isolation Forest (Zero-Day Safety Net)
===========================================================================

PURPOSE
-------
Supervised models (XGBoost and R-GCN) excel at detecting fraud patterns they have
already seen in historical training data. However, sophisticated criminal syndicates
continually engineer brand-new "zero-day" money mule strategies (e.g., novel transaction
cadences, abnormal token velocities, synthetic identity smurfing) that do not conform
to known labels.

This module implements Brain 3 (The Multi-Dimensional Safety Net):
  • Algorithm: Extended Isolation Forest (EIF) with Scikit-Learn IsolationForest fallback
  • Key Mathematical Innovation:
      - Standard Isolation Forest cuts space strictly parallel to coordinate axes (x_i <= θ).
        In high-dimensional space, this creates geometric blind spots and artifact bounding boxes.
      - Extended Isolation Forest branches along random hyperplanes at arbitrary angles:
            n_vec · (x_vec - p_vec) <= 0
        where n_vec is a random slope vector and p_vec is a random intercept.
  • Unsupervised Training: Trained on legitimate banking transactions to learn the true
    density distribution of normal customer behavior.
  • Anomaly Scoring: Samples isolated in few splits (short tree paths) receive high anomaly
    scores s in [0.0, 1.0], exposing zero-day fraud before funds leave the ecosystem.
  • Output: Out-of-fold anomaly vector saved to `data/oof_eif_preds.npy` for Phase 7 Late-Fusion.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

# Graceful import: prefer eif, fallback to sklearn.ensemble.IsolationForest
try:
    import eif as iso
    HAS_EIF_LIB = True
except ImportError:
    HAS_EIF_LIB = False
from sklearn.ensemble import IsolationForest


class ZeroDayAnomalyDetector:
    """
    Unsupervised Anomaly Detector using Extended Isolation Forests
    to identify zero-day, previously unseen fraud patterns.
    """

    def __init__(
        self,
        ntrees: int = 250,
        sample_size: int = 256,
        extension_level: int = 2,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        ntrees : int
            Number of isolation trees in the ensemble.
        sample_size : int
            Subsample size per isolation tree.
        extension_level : int
            Number of features combined per random hyperplane cut.
            extension_level=2 selects random slope hyperplanes combining 2 features,
            removing coordinate-axis bias without excessive compute overhead.
        random_state : int
            Random seed for deterministic reproducibility.
        """
        self.ntrees = ntrees
        self.sample_size = sample_size
        self.extension_level = extension_level
        self.random_state = random_state

        self.forest: Any = None
        self.use_eif: bool = HAS_EIF_LIB
        self.metrics: Dict[str, Any] = {}
        self.predictions: Optional[np.ndarray] = None

    def fit(self, X_baseline: np.ndarray) -> None:
        """
        Fits the isolation forest on normal/baseline banking transactions.
        """
        num_samples, num_features = X_baseline.shape
        print(f"[*] Training Anomaly Forest on {num_samples:,} baseline records x {num_features} features...")

        if self.use_eif:
            print(f"[*] Using native Extended Isolation Forest (ExtensionLevel={self.extension_level})...")
            # eif expects float64 array
            np.random.seed(self.random_state)
            X_clean = np.ascontiguousarray(X_baseline, dtype=np.float64)
            self.forest = iso.iForest(
                X_clean,
                ntrees=self.ntrees,
                sample_size=min(self.sample_size, num_samples),
                ExtensionLevel=min(self.extension_level, num_features - 1),
            )

        else:
            print("[*] Using Scikit-Learn IsolationForest engine...")
            self.forest = IsolationForest(
                n_estimators=self.ntrees,
                max_samples=min(self.sample_size, num_samples),
                contamination=0.01,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.forest.fit(X_baseline)

        print("[+] Anomaly forest fitting completed successfully.")

    def compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Computes normalized anomaly scores in range [0.0, 1.0].
        Score -> 1.0: Statistically anomalous / extreme outlier.
        Score -> 0.0: Centered deep within normal legitimate clusters.
        """
        if self.forest is None:
            raise RuntimeError("Anomaly detector has not been fitted yet.")

        if self.use_eif:
            X_clean = np.ascontiguousarray(X, dtype=np.float64)
            # Path lengths: shorter path means isolated faster (more anomalous)
            raw_paths = self.forest.compute_paths(X_in=X_clean)
            
            # Theoretical average path length c(n) for sample size n
            n = float(min(self.sample_size, X.shape[0]))
            if n > 2:
                c_n = 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)
            else:
                c_n = 1.0

            # s = 2^(- E(h) / c(n))
            scores = 2.0 ** (- (raw_paths / max(1e-5, c_n)))
            # Min-max scale to ensure full [0.0, 1.0] coverage
            s_min, s_max = np.min(scores), np.max(scores)
            if s_max > s_min:
                scores = (scores - s_min) / (s_max - s_min)
            return scores.astype(np.float32)
        else:
            # sklearn: score_samples returns opposite of anomaly score (lower is more anomalous)
            # range typically around [-0.8, -0.3]
            raw_scores = -self.forest.score_samples(X)
            s_min, s_max = np.min(raw_scores), np.max(raw_scores)
            if s_max > s_min:
                norm_scores = (raw_scores - s_min) / (s_max - s_min)
            else:
                norm_scores = raw_scores
            return norm_scores.astype(np.float32)

    def evaluate_against_labels(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates unsupervised anomaly scores against ground-truth labels.
        Demonstrates how effectively pure mathematical isolation surfaces fraudulent accounts
        without ever seeing their supervision labels during training!
        """
        scores = self.compute_anomaly_scores(X)
        self.predictions = scores

        auprc = average_precision_score(y, scores)
        roc_auc = roc_auc_score(y, scores)

        # Precision@K on top isolated accounts
        top_50 = np.argsort(scores)[::-1][:50]
        top_100 = np.argsort(scores)[::-1][:100]
        top_200 = np.argsort(scores)[::-1][:200]

        p_50 = float(np.mean(y[top_50]))
        p_100 = float(np.mean(y[top_100]))
        p_200 = float(np.mean(y[top_200]))

        # Anomaly score distribution separation
        mule_scores = scores[y == 1]
        legit_scores = scores[y == 0]

        self.metrics = {
            "unsupervised_auprc": float(auprc),
            "unsupervised_roc_auc": float(roc_auc),
            "precision_at_50": float(p_50),
            "precision_at_100": float(p_100),
            "precision_at_200": float(p_200),
            "mule_mean_anomaly_score": float(np.mean(mule_scores)),
            "legit_mean_anomaly_score": float(np.mean(legit_scores)),
            "separation_delta": float(np.mean(mule_scores) - np.mean(legit_scores)),
            "engine_used": "Extended_Isolation_Forest" if self.use_eif else "Sklearn_Isolation_Forest",
        }

        return self.metrics

    def save(self, model_path: str, metrics_path: Optional[str] = None) -> None:
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(self, f)
        print(f"[+] Saved Anomaly Detector checkpoint -> {model_path}")

        if metrics_path:
            metrics_dir = os.path.dirname(metrics_path)
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(self.metrics, f, indent=2)
            print(f"[+] Saved Anomaly Detector metrics -> {metrics_path}")

    @classmethod
    def load(cls, model_path: str) -> "ZeroDayAnomalyDetector":
        with open(model_path, "rb") as f:
            instance = pickle.load(f)
        print(f"[+] Loaded Anomaly Detector checkpoint <- {model_path}")
        return instance
