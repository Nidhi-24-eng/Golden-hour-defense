"""
xgboost_model.py — Phase 4: Tabular XGBoost Behavioral Ensemble
================================================================

PURPOSE
-------
In money mule fraud detection, tabular behavioral signals capture how an account's
transactional habits evolve over time. Mule herders often "wake up" dormant accounts
and route sudden bursts of high-velocity funds before abandoning them.

This module implements Brain 1 of our Dual-Brain architecture:
  • Algorithm: Gradient Boosted Decision Trees (XGBoost)
  • Objective: `binary:logistic`
  • Metric: `aucpr` (Area Under the Precision-Recall Curve)
  • Imbalance Mitigation: Dynamic `scale_pos_weight` (penalizes false negatives
    proportionally to the legitimate-to-mule ratio: ~42.8x)
  • Validation: Stratified 5-Fold Cross-Validation (prevents data leakage and ensures
    identical fraud distribution across every validation fold)
  • Persistence: Native JSON model checkpoint (`models/xgboost_mule_detector.json`)
    and out-of-fold probability vectors for downstream Late-Fusion (Phase 7).
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
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb


class TabularMuleClassifier:
    """
    Behavioral Tabular Classifier using an imbalanced-tuned XGBoost ensemble.
    """

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 6,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.7,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        n_estimators : int
            Maximum number of gradient boosted trees.
        max_depth : int
            Maximum tree depth (depth 6 balances complex interactions with regularization).
        learning_rate : float
            Shrinkage step size (0.03 prevents early overfitting).
        subsample : float
            Row subsampling ratio per tree.
        colsample_bytree : float
            Feature subsampling ratio per tree.
        scale_pos_weight : float, optional
            Negative / Positive ratio. If None, dynamically computed from training labels.
        random_state : int
            Seed for deterministic reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state

        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []
        self.cv_metrics: Dict[str, Any] = {}
        self.oof_predictions: Optional[np.ndarray] = None

    def _compute_precision_at_k(self, y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
        """
        Computes Precision@K:
        Of the top K highest-risk accounts flagged by the model,
        what fraction are verified mule accounts?
        """
        top_k_indices = np.argsort(y_prob)[::-1][:k]
        return float(np.mean(y_true[top_k_indices]))

    def train_cross_validated(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_splits: int = 5,
        early_stopping_rounds: int = 40,
    ) -> Dict[str, Any]:
        """
        Executes Stratified 5-Fold Cross Validation. Computes out-of-fold
        predictions, validation AUPRC, ROC-AUC, and Precision@K.
        """
        self.feature_names = feature_names
        num_samples, num_features = X.shape
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())

        # Dynamically compute scale_pos_weight if not specified
        calc_weight = (n_neg / max(1, n_pos)) if self.scale_pos_weight is None else self.scale_pos_weight
        self.scale_pos_weight = calc_weight

        print(f"[*] Training Data: {num_samples:,} samples x {num_features} features")
        print(f"[*] Class Distribution: Legitimate={n_neg:,} | Mule={n_pos:,} ({n_pos / num_samples * 100:.2f}%)")
        print(f"[*] Dynamic scale_pos_weight: {self.scale_pos_weight:.2f}")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        self.oof_predictions = np.zeros(num_samples, dtype=np.float32)

        fold_auprcs: List[float] = []
        fold_rocs: List[float] = []
        fold_p50: List[float] = []
        fold_p100: List[float] = []

        print(f"\n[*] Executing Stratified {n_splits}-Fold Cross Validation...")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            fold_model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                scale_pos_weight=self.scale_pos_weight,
                objective="binary:logistic",
                eval_metric="aucpr",
                tree_method="hist",
                random_state=self.random_state + fold,
                early_stopping_rounds=early_stopping_rounds,
            )

            fold_model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            val_preds = fold_model.predict_proba(X_val)[:, 1]
            self.oof_predictions[val_idx] = val_preds

            val_auprc = average_precision_score(y_val, val_preds)
            val_roc = roc_auc_score(y_val, val_preds)
            p_50 = self._compute_precision_at_k(y_val, val_preds, k=min(50, len(val_preds)))
            p_100 = self._compute_precision_at_k(y_val, val_preds, k=min(100, len(val_preds)))

            fold_auprcs.append(val_auprc)
            fold_rocs.append(val_roc)
            fold_p50.append(p_50)
            fold_p100.append(p_100)

            best_iter = getattr(fold_model, "best_iteration", self.n_estimators)
            print(f"    Fold {fold}/{n_splits} | Best Iter: {best_iter:03d} | AUPRC: {val_auprc:.4f} | ROC-AUC: {val_roc:.4f} | P@50: {p_50:.2f}")

        # Overall Out-Of-Fold Evaluation
        overall_auprc = average_precision_score(y, self.oof_predictions)
        overall_roc = roc_auc_score(y, self.oof_predictions)
        overall_p50 = self._compute_precision_at_k(y, self.oof_predictions, k=50)
        overall_p100 = self._compute_precision_at_k(y, self.oof_predictions, k=100)
        overall_p200 = self._compute_precision_at_k(y, self.oof_predictions, k=200)

        # Optimal F1 threshold search
        thresholds = np.linspace(0.1, 0.9, 81)
        f1_scores = [f1_score(y, (self.oof_predictions >= t).astype(int), zero_division=0) for t in thresholds]
        best_idx = int(np.argmax(f1_scores))
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])

        binary_preds = (self.oof_predictions >= best_threshold).astype(int)
        prec = float(precision_score(y, binary_preds, zero_division=0))
        rec = float(recall_score(y, binary_preds, zero_division=0))
        cm = confusion_matrix(y, binary_preds).tolist()

        # Train final production model on full dataset
        print("\n[*] Training final full-dataset production model checkpoint...")
        self.model = xgb.XGBClassifier(
            n_estimators=int(np.mean([m for m in fold_auprcs])) if fold_auprcs else self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=self.random_state,
        )
        self.model.fit(X, y, verbose=False)

        # Feature Importance Analysis
        feature_importances = self.model.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]
        top_features = [
            {"rank": i + 1, "feature": self.feature_names[idx], "importance": float(feature_importances[idx])}
            for i, idx in enumerate(sorted_indices[:15])
        ]

        self.cv_metrics = {
            "mean_cv_auprc": float(np.mean(fold_auprcs)),
            "std_cv_auprc": float(np.std(fold_auprcs)),
            "mean_cv_roc_auc": float(np.mean(fold_rocs)),
            "overall_oof_auprc": float(overall_auprc),
            "overall_oof_roc_auc": float(overall_roc),
            "precision_at_50": float(overall_p50),
            "precision_at_100": float(overall_p100),
            "precision_at_200": float(overall_p200),
            "optimal_threshold": round(best_threshold, 3),
            "best_f1_score": round(best_f1, 4),
            "precision_at_optimal": round(prec, 4),
            "recall_at_optimal": round(rec, 4),
            "confusion_matrix": {
                "true_negative": cm[0][0],
                "false_positive": cm[0][1],
                "false_negative": cm[1][0],
                "true_positive": cm[1][1],
            },
            "top_15_features": top_features,
        }

        return self.cv_metrics

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """
        Infers continuous mule risk probability P(Mule | Features) in range [0.0, 1.0].
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, model_path: str, metrics_path: Optional[str] = None) -> None:
        """
        Saves native XGBoost JSON checkpoint and evaluation metrics.
        """
        if self.model is None:
            raise RuntimeError("No trained model to save.")

        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        self.model.save_model(model_path)
        print(f"[+] Saved native XGBoost model checkpoint -> {model_path}")

        if metrics_path:
            metrics_dir = os.path.dirname(metrics_path)
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(self.cv_metrics, f, indent=2)
            print(f"[+] Saved model validation metrics -> {metrics_path}")

    @classmethod
    def load(cls, model_path: str) -> "TabularMuleClassifier":
        """
        Loads a trained XGBoost JSON checkpoint for real-time inference.
        """
        instance = cls()
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(model_path)
        print(f"[+] Loaded XGBoost model checkpoint <- {model_path}")
        return instance
