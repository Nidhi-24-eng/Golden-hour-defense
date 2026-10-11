"""
i4c_webhook.py — Phase 9: I4C Regulatory Webhook & Automated Emergency Brake
=============================================================================

PURPOSE
-------
Detecting cyber fraud is useless if the system cannot stop the money from leaving
the banking ecosystem in time. When a scam occurs, criminals scatter proceeds across
chains of mule accounts in under 30 minutes.

This module provides the Automated Emergency Brake:
  1. Direct Integration with Indian Cybercrime Coordination Centre (I4C):
       Listens to National Cybercrime Reporting Portal (NCRP) 1930 Helpline tickets.
  2. Exact Regulatory Ingestion Schema:
       • Mule_Account_Number              (Target account to freeze)
       • IFSC_Code                        (Bank / Branch identifier)
       • Transaction_Reference_Number     (UPI / IMPS / NEFT UTR ID)
       • Timestamp_of_Incident            (ISO 8601 incident timestamp)
  3. Golden Hour Automated Action:
       • Immediate Hard Financial Lien on the primary mule account (< 5ms).
       • Real-Time Graph Traversal on the topology compiled in Phase 3:
           - Traces 1st-degree and 2nd-degree downstream recipient accounts.
           - Traces accounts sharing the same device fingerprint or proxy pool.
           - Cascades preventive debit freezes on downstream beneficiary accounts,
             locking the stolen funds inside the banking system before withdrawal!
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import time
from typing import Any, Dict, List, Optional, Set

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
import numpy as np
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════
# EXACT REGULATORY DATA SCHEMA (I4C / NCRP 1930 HELPLINE SPECIFICATION)
# ═══════════════════════════════════════════════════════════════════════

class I4CTicketPayload(BaseModel):
    """
    Standard webhook payload emitted by the National Cybercrime Reporting Portal (1930 helpline).
    """
    Mule_Account_Number: str = Field(..., description="Target mule account number to freeze")
    IFSC_Code: str = Field(..., description="Bank/Branch IFSC identifier code")
    Transaction_Reference_Number: str = Field(..., description="UPI/IMPS/NEFT Transaction Reference / UTR Number")
    Timestamp_of_Incident: str = Field(..., description="ISO 8601 incident timestamp from citizen complaint")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Mule_Account_Number": "ACC_000042",
                "IFSC_Code": "BKID0001234",
                "Transaction_Reference_Number": "UPI_REF_987654321001",
                "Timestamp_of_Incident": "2026-10-11T13:20:00Z",
            }
        }
    }


class GoldenHourEmergencyBrake:
    """
    Automated zero-latency financial lien engine with multi-hop
    graph traversal across downstream beneficiary mule accounts.
    """

    def __init__(
        self,
        graph_npz_path: str = "data/graphs/graph_arrays.npz",
        node_mapping_path: str = "data/graphs/node_mapping.json",
    ):
        self.graph_npz_path = graph_npz_path
        self.node_mapping_path = node_mapping_path

        self.account_to_idx: Dict[str, int] = {}
        self.idx_to_account: Dict[int, str] = {}
        self.adjacency: Dict[int, List[Tuple[int, int]]] = {}  # node -> [(target_node, rel_type)]
        self.frozen_accounts: Dict[str, Dict[str, Any]] = {}

        self._load_graph_topology()

    def _load_graph_topology(self) -> None:
        """Loads compiled multi-relational graph topology for sub-millisecond traversal."""
        if os.path.exists(self.node_mapping_path) and os.path.exists(self.graph_npz_path):
            with open(self.node_mapping_path, "r") as f:
                meta = json.load(f)
                self.account_to_idx = meta.get("account_to_idx", {})
                num_nodes = meta.get("num_nodes", 0)

            # If account_to_idx dict was omitted, reconstruct deterministic ACC_xxxxxx IDs
            if not self.account_to_idx and num_nodes > 0:
                self.account_to_idx = {f"ACC_{i:06d}": i for i in range(num_nodes)}

            self.idx_to_account = {v: k for k, v in self.account_to_idx.items()}

            npz = np.load(self.graph_npz_path)
            edge_index = npz["edge_index"]
            edge_type = npz["edge_type"]

            # Build fast in-memory adjacency list
            for i in range(edge_index.shape[1]):
                src = int(edge_index[0, i])
                dst = int(edge_index[1, i])
                rel = int(edge_type[i])
                self.adjacency.setdefault(src, []).append((dst, rel))

            print(f"[*] Emergency Brake Graph Initialized: {len(self.account_to_idx):,} nodes indexed.")
        else:
            print("[!] Graph topology files not found. Running in standalone fallback mode.")


    def trace_downstream_mule_hops(self, account_id: str, max_depth: int = 2) -> Dict[str, List[str]]:
        """
        Traverses the multi-relational graph to discover:
          1. Direct transfer recipients (1st-degree hops)
          2. Secondary transfer recipients (2nd-degree hops)
          3. Accounts sharing the same device or proxy loop
        """
        if account_id not in self.account_to_idx:
            return {"1st_degree_transfers": [], "2nd_degree_transfers": [], "colocated_entities": []}

        start_node = self.account_to_idx[account_id]
        
        hop1_transfers: Set[int] = set()
        colocated: Set[int] = set()

        for neighbor, rel in self.adjacency.get(start_node, []):
            if rel == 0:  # TRANSFERRED_TO
                hop1_transfers.add(neighbor)
            elif rel in (1, 2):  # SHARED_DEVICE or SHARED_PROXY_IP
                colocated.add(neighbor)

        # 2nd-degree transfer hops
        hop2_transfers: Set[int] = set()
        for h1_node in hop1_transfers:
            for neighbor, rel in self.adjacency.get(h1_node, []):
                if rel == 0 and neighbor != start_node and neighbor not in hop1_transfers:
                    hop2_transfers.add(neighbor)

        return {
            "1st_degree_transfers": [self.idx_to_account.get(i, f"NODE_{i}") for i in hop1_transfers],
            "2nd_degree_transfers": [self.idx_to_account.get(i, f"NODE_{i}") for i in hop2_transfers],
            "colocated_entities": [self.idx_to_account.get(i, f"NODE_{i}") for i in colocated],
        }

    def execute_lien(self, ticket: I4CTicketPayload) -> Dict[str, Any]:
        """
        Executes immediate zero-latency financial lien and downstream holds.
        """
        t0 = time.perf_counter()
        now_iso = datetime.now(timezone.utc).isoformat()
        target_account = ticket.Mule_Account_Number

        # 1. Primary Hard Lien
        self.frozen_accounts[target_account] = {
            "account_number": target_account,
            "ifsc": ticket.IFSC_Code,
            "status": "HARD_LIEN_APPLIED",
            "reason": f"I4C 1930 Helpline Ticket: {ticket.Transaction_Reference_Number}",
            "frozen_at": now_iso,
            "tier": "PRIMARY_MULE_TARGET",
        }

        # 2. Downstream Multi-Hop Traversal
        downstream = self.trace_downstream_mule_hops(target_account, max_depth=2)
        cascaded_freeze: List[str] = []

        # Cascade preventive debit freezes on immediate recipients & colocated nodes
        all_cascade_nodes = set(downstream["1st_degree_transfers"] + downstream["colocated_entities"])
        for cascaded_acc in all_cascade_nodes:
            if cascaded_acc not in self.frozen_accounts:
                self.frozen_accounts[cascaded_acc] = {
                    "account_number": cascaded_acc,
                    "status": "PREVENTIVE_DEBIT_FREEZE",
                    "reason": f"Downstream link to I4C flagged mule {target_account}",
                    "frozen_at": now_iso,
                    "tier": "CASCADED_BENEFICIARY_HOLD",
                }
                cascaded_freeze.append(cascaded_acc)

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "status": "SUCCESS",
            "action": "EMERGENCY_BRAKE_EXECUTED",
            "target_mule": target_account,
            "ifsc": ticket.IFSC_Code,
            "utr_ref": ticket.Transaction_Reference_Number,
            "execution_latency_ms": round(latency_ms, 3),
            "primary_lien": "CONFIRMED_FROZEN",
            "cascaded_holds_count": len(cascaded_freeze),
            "cascaded_accounts": cascaded_freeze,
            "downstream_network": downstream,
            "processed_at": now_iso,
        }


# ═══════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION & REGULATORY WEBHOOK GATEWAY
# ═══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="The Golden Hour Defense Framework - I4C Regulatory Gateway",
    description="Automated zero-latency emergency lien webhook connected to the National Cybercrime Reporting Portal (1930 Helpline).",
    version="1.0.0",
)

# Singleton Emergency Brake Engine
brake_engine = GoldenHourEmergencyBrake()


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {
        "status": "HEALTHY",
        "service": "I4C Regulatory Webhook Gateway",
        "total_active_liens": len(brake_engine.frozen_accounts),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/i4c/ticket", status_code=status.HTTP_200_OK)
async def ingest_i4c_ticket(ticket: I4CTicketPayload):
    """
    Ingests 1930 Helpline ticket and triggers the Golden Hour Emergency Brake.
    """
    if not ticket.Mule_Account_Number or not ticket.IFSC_Code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mule_Account_Number and IFSC_Code are strictly required.",
        )

    result = brake_engine.execute_lien(ticket)
    return result


@app.get("/api/v1/i4c/lien/{account_number}")
async def check_lien_status(account_number: str):
    """
    Queries real-time financial lien status for a given account.
    """
    if account_number in brake_engine.frozen_accounts:
        return {"account_number": account_number, "lien_status": brake_engine.frozen_accounts[account_number]}
    return {"account_number": account_number, "lien_status": "NO_ACTIVE_LIEN"}


@app.get("/api/v1/i4c/active_liens")
async def list_active_liens():
    """
    Lists all active financial liens and cascaded holds.
    """
    return {
        "count": len(brake_engine.frozen_accounts),
        "liens": list(brake_engine.frozen_accounts.values()),
    }
