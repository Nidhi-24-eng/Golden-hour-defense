"""
rgcn_model.py — Phase 5: Relational Graph Convolutional Network (R-GCN)
========================================================================

PURPOSE
-------
While XGBoost (Brain 1) evaluates what an individual account is doing in isolation,
it cannot detect distributed mule rings where fraudsters circulate stolen cash in small
amounts across multiple accounts that share devices, Wi-Fi networks, and proxy loops.

This module implements Brain 2: The Network Detective (R-GCN):
  • Framework: PyTorch Geometric (PyG)
  • Architecture: Multi-layer Relational Graph Convolution (`RGCNConv`)
  • Relations (r in {0, 1, 2}):
      - Relation 0: TRANSFERRED_TO   (Directed monetary flows)
      - Relation 1: SHARED_DEVICE    (Hardware colocation links)
      - Relation 2: SHARED_PROXY_IP  (Proxy/VPN cluster loops)
  • Basis Decomposition: `num_bases=4` compresses relation-specific transformation
    matrices W_r, preventing overfitting on dense relations.
  • Mathematical Formulation:
      h_i^{(l+1)} = σ( W_0^{(l)} h_i^{(l)} + Σ_{r} Σ_{j \in N_i^r} (1 / c_{i,r}) W_r^{(l)} h_j^{(l)} )
  • Loss Function: BCEWithLogitsLoss with `pos_weight` tuned to the ~0.89% class imbalance.
  • Output: Relational risk score P_RGCN for every account, saved to `data/oof_rgcn_preds.npy`
    for downstream Late-Fusion (Phase 7).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv


class MuleRGCN(nn.Module):
    """
    Relational Graph Convolutional Network for Money Mule Account Detection.
    """

    def __init__(
        self,
        in_channels: int = 250,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_relations: int = 3,
        num_bases: int = 4,
        dropout: float = 0.25,
    ):
        super(MuleRGCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.dropout = dropout

        # Layer 1: Multi-relational neighborhood aggregation
        self.conv1 = RGCNConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            num_relations=num_relations,
            num_bases=num_bases,
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        # Layer 2: Higher-order 2-hop topological structure
        self.conv2 = RGCNConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            num_bases=num_bases,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Classification Head: Project structural representations + raw tabular features (Residual Skip)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels + out_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Computes forward pass over multi-relational graph with residual skip connection.
        Returns raw unscaled logits [N].
        """
        x_in = x  # Retain raw tabular behavioral features

        # Block 1
        h1 = self.conv1(x, edge_index, edge_type)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Block 2
        h2 = self.conv2(h1, edge_index, edge_type)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        # Residual Skip: Concatenate raw tabular features with graph topology embeddings
        combined = torch.cat([x_in, h2], dim=-1)

        # Classifier logits
        logits = self.classifier(combined).squeeze(-1)
        return logits

    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Returns continuous risk probability P(Mule | Topology, Features) in [0.0, 1.0].
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index, edge_type)
            return torch.sigmoid(logits)



class RGCNDetectorTrainer:
    """
    Trainer and evaluation harness for MuleRGCN.
    """

    def __init__(
        self,
        in_channels: int = 250,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_relations: int = 3,
        num_bases: int = 4,
        dropout: float = 0.25,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        pos_weight: Optional[float] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight

        self.model = MuleRGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            num_bases=num_bases,
            dropout=dropout,
        ).to(self.device)

        self.metrics: Dict[str, Any] = {}
        self.predictions: Optional[np.ndarray] = None

    def _compute_precision_at_k(self, y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
        top_k_indices = np.argsort(y_prob)[::-1][:k]
        return float(np.mean(y_true[top_k_indices]))

    def train_and_evaluate(
        self,
        data: Data,
        epochs: int = 120,
        val_ratio: float = 0.2,
        patience: int = 25,
    ) -> Dict[str, Any]:
        """
        Trains the R-GCN on graph data with stratified validation split.
        """
        num_nodes = data.x.size(0)
        y_np = data.y.cpu().numpy()
        n_pos = int((y_np == 1).sum())
        n_neg = int((y_np == 0).sum())

        # Dynamically set pos_weight if not specified (balanced at 40.0 to prevent false-alarm explosion)
        pw = min(40.0, float(n_neg / max(1, n_pos))) if self.pos_weight is None else self.pos_weight
        self.pos_weight = pw
        pos_weight_tensor = torch.tensor([self.pos_weight], dtype=torch.float32, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

        # Stratified node split (train/validation masks)
        rng = np.random.default_rng(42)
        pos_indices = np.where(y_np == 1)[0]
        neg_indices = np.where(y_np == 0)[0]

        rng.shuffle(pos_indices)
        rng.shuffle(neg_indices)

        n_val_pos = max(1, int(len(pos_indices) * val_ratio))
        n_val_neg = max(1, int(len(neg_indices) * val_ratio))

        val_mask = np.zeros(num_nodes, dtype=bool)
        val_mask[pos_indices[:n_val_pos]] = True
        val_mask[neg_indices[:n_val_neg]] = True

        train_mask = ~val_mask

        train_mask_t = torch.from_numpy(train_mask).to(self.device)
        val_mask_t = torch.from_numpy(val_mask).to(self.device)

        data = data.to(self.device)
        labels = data.y.float()

        print(f"[*] R-GCN Device: {self.device}")
        print(f"[*] Training Nodes: {train_mask.sum():,} | Validation Nodes: {val_mask.sum():,}")
        print(f"[*] Imbalance pos_weight: {self.pos_weight:.2f}")

        best_val_auprc = 0.0
        best_state = None
        no_improve_epochs = 0

        print("\n[*] Starting R-GCN Multi-Relational Training Loop...")
        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            logits = self.model(data.x, data.edge_index, data.edge_type)
            loss = criterion(logits[train_mask_t], labels[train_mask_t])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            optimizer.step()

            # Validation evaluation
            self.model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(logits[val_mask_t]).cpu().numpy()
                val_labels = labels[val_mask_t].cpu().numpy()

                val_auprc = average_precision_score(val_labels, val_probs)
                val_roc = roc_auc_score(val_labels, val_probs)

            scheduler.step(val_auprc)

            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"    Epoch {epoch:03d}/{epochs} | Train Loss: {loss.item():.4f} | Val AUPRC: {val_auprc:.4f} | Val ROC-AUC: {val_roc:.4f}")

            if no_improve_epochs >= patience:
                print(f"[*] Early stopping triggered at epoch {epoch} (best Val AUPRC: {best_val_auprc:.4f})")
                break

        # Load best checkpoint
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        # Final full graph evaluation
        self.model.eval()
        with torch.no_grad():
            all_probs = torch.sigmoid(self.model(data.x, data.edge_index, data.edge_type)).cpu().numpy()
            self.predictions = all_probs

        overall_auprc = average_precision_score(y_np, all_probs)
        overall_roc = roc_auc_score(y_np, all_probs)
        p50 = self._compute_precision_at_k(y_np, all_probs, k=50)
        p100 = self._compute_precision_at_k(y_np, all_probs, k=100)
        p200 = self._compute_precision_at_k(y_np, all_probs, k=200)

        thresholds = np.linspace(0.1, 0.9, 81)
        f1_scores = [f1_score(y_np, (all_probs >= t).astype(int), zero_division=0) for t in thresholds]
        best_idx = int(np.argmax(f1_scores))
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])

        binary_preds = (all_probs >= best_threshold).astype(int)
        prec = float(precision_score(y_np, binary_preds, zero_division=0))
        rec = float(recall_score(y_np, binary_preds, zero_division=0))
        cm = confusion_matrix(y_np, binary_preds).tolist()

        self.metrics = {
            "best_val_auprc": float(best_val_auprc),
            "overall_auprc": float(overall_auprc),
            "overall_roc_auc": float(overall_roc),
            "precision_at_50": float(p50),
            "precision_at_100": float(p100),
            "precision_at_200": float(p200),
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
        }

        return self.metrics

    def save(self, model_path: str, metrics_path: Optional[str] = None) -> None:
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "out_channels": self.out_channels,
            "num_relations": self.num_relations,
            "num_bases": self.num_bases,
            "dropout": self.dropout,
        }, model_path)
        print(f"[+] Saved R-GCN PyTorch model checkpoint -> {model_path}")

        if metrics_path:
            metrics_dir = os.path.dirname(metrics_path)
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(self.metrics, f, indent=2)
            print(f"[+] Saved R-GCN validation metrics -> {metrics_path}")
