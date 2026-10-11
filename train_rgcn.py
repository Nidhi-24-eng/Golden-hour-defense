"""
Execution and verification runner for Phase 5: PyTorch Geometric Relational GCN (R-GCN).
"""
import json
import os
import sys
import time
import numpy as np
import torch
from torch_geometric.data import Data
from src.models.rgcn_model import RGCNDetectorTrainer


def load_graph_data(graph_pt_path: str, npz_path: str) -> Data:
    """
    Loads PyG Data object from .pt if present, else constructs Data from .npz.
    """
    if os.path.exists(graph_pt_path):
        print(f"[*] Ingesting PyTorch Geometric graph from: {graph_pt_path}")
        data = torch.load(graph_pt_path, weights_only=False)
        return data

    if os.path.exists(npz_path):
        print(f"[*] Constructing PyTorch Geometric graph from: {npz_path}")
        npz = np.load(npz_path)
        data = Data(
            x=torch.from_numpy(npz["x"]).float(),
            y=torch.from_numpy(npz["y"]).long(),
            edge_index=torch.from_numpy(npz["edge_index"]).long(),
            edge_type=torch.from_numpy(npz["edge_type"]).long(),
        )
        return data

    raise FileNotFoundError(
        f"Graph data not found at {graph_pt_path} or {npz_path}. "
        "Please run 'python -m src.preprocessing.graph_builder' first."
    )


def main():
    graph_pt_path = os.path.join("data", "graphs", "relational_graph.pt")
    npz_path = os.path.join("data", "graphs", "graph_arrays.npz")
    model_output_path = os.path.join("models", "rgcn_mule_detector.pt")
    metrics_output_path = os.path.join("models", "rgcn_metrics.json")
    preds_output_path = os.path.join("data", "oof_rgcn_preds.npy")

    print("=" * 70)
    print("STAGE 4: RELATIONAL GRAPH CONVOLUTIONAL NETWORK (R-GCN) TRAINING")
    print("=" * 70)

    start_time = time.time()

    # 1. Load compiled multi-relational graph
    data = load_graph_data(graph_pt_path, npz_path)
    num_nodes = data.x.size(0)
    in_channels = data.x.size(1)
    num_edges = data.edge_index.size(1)

    print(f"[*] Graph Summary: {num_nodes:,} nodes | {num_edges:,} multi-relational edges | {in_channels} features")

    # 2. Instantiate and train R-GCN
    trainer = RGCNDetectorTrainer(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=32,
        num_relations=3,
        num_bases=4,
        dropout=0.25,
        lr=0.005,
        weight_decay=1e-4,
    )

    metrics = trainer.train_and_evaluate(
        data=data,
        epochs=120,
        val_ratio=0.20,
        patience=25,
    )

    # 3. Save model checkpoint and evaluation metrics
    trainer.save(model_path=model_output_path, metrics_path=metrics_output_path)

    # 4. Save out-of-fold/graph predictions for Phase 7 (Late Fusion)
    if trainer.predictions is not None:
        os.makedirs(os.path.dirname(preds_output_path), exist_ok=True)
        np.save(preds_output_path, trainer.predictions)
        print(f"[+] Relational risk probability vector saved -> {preds_output_path}")

    elapsed_time = time.time() - start_time

    # 5. Display evaluation report
    print("\n" + "=" * 70)
    print("R-GCN VALIDATION PERFORMANCE & BENCHMARK REPORT")
    print("=" * 70)
    print(f"[*] Total Training & Evaluation Time: {elapsed_time:.2f} seconds")
    print(f"[*] Best Validation AUPRC:            {metrics['best_val_auprc']:.4f}")
    print(f"[*] Overall Graph AUPRC:              {metrics['overall_auprc']:.4f}")
    print(f"[*] Overall Graph ROC-AUC:            {metrics['overall_roc_auc']:.4f}")
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

    print("\n" + "=" * 70)
    print("PHASE 5 R-GCN RELATIONAL DETECTOR COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
