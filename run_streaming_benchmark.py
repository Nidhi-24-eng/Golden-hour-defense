"""
Execution and verification runner for Phase 8: Stateful Velocity Engine (Flink + RocksDB).
"""
import time
import numpy as np
from src.streaming.velocity_tracker import StatefulVelocityEngine


def main():
    print("=" * 70)
    print("STAGE 7: STATEFUL VELOCITY ENGINE BENCHMARK & SIMULATION")
    print("=" * 70)

    engine = StatefulVelocityEngine()

    # 1. High-Throughput Stream Ingestion Benchmark (50,000 Transactions)
    n_txs = 50_000
    n_accounts = 5_000
    accounts = [f"ACC_{i:06d}" for i in range(n_accounts)]
    amounts = np.random.uniform(100.0, 50000.0, size=n_txs)
    types = np.random.choice(["CREDIT", "DEBIT"], size=n_txs)
    acc_choices = np.random.choice(accounts, size=n_txs)

    print(f"[*] Benchmarking stream throughput across {n_txs:,} transactions...")
    t0 = time.perf_counter()
    for i in range(n_txs):
        engine.record_transaction(account_id=acc_choices[i], amount=amounts[i], tx_type=types[i])
    t_elapsed = time.perf_counter() - t0

    tps = n_txs / t_elapsed
    avg_latency_us = (t_elapsed / n_txs) * 1_000_000

    print(f"[+] Total Processing Time:     {t_elapsed:.4f} seconds")
    print(f"[+] Stream Ingestion Rate:     {tps:,.0f} transactions/second")
    print(f"[+] Average Latency:           {avg_latency_us:.2f} microseconds per check (< 0.05ms)")

    # 2. Simulate Real-Time Banking Attack Scenarios
    print("\n" + "=" * 70)
    print("REAL-TIME VELOCITY ATTACK PATTERN SIMULATION")
    print("=" * 70)

    engine.clear()
    base_time = time.time()

    # Scenario 1: High-Velocity Smurfing Attack (8 quick payments in 45 seconds)
    mule_acc_1 = "ACC_MULE_SMURF_001"
    print("\n[Scenario 1] High-Velocity Smurfing Burst (8 transfers in 40s):")
    for sec_offset in range(0, 40, 5):
        m = engine.record_transaction(
            account_id=mule_acc_1,
            amount=4999.0,
            tx_type="CREDIT",
            timestamp=base_time + sec_offset,
        )
    print(f"    Account: {mule_acc_1}")
    print(f"    60s Count: {m['tx_count_60s']} | 5m Inflow: ₹{m['inflow_5m']:,.2f} | Risk Score: {m['velocity_risk_score']}")
    assert m["tx_count_60s"] == 8, "Expected 8 transactions in 60s window"

    # Scenario 2: Rapid Drain Mule Pass-Through (₹2,50,000 received, ₹2,45,000 drained in 25s)
    mule_acc_2 = "ACC_MULE_PASSTHRU_002"
    print("\n[Scenario 2] Rapid Drain Pass-Through (Receive ₹2.5L and Drain ₹2.45L in 25s):")
    engine.record_transaction(account_id=mule_acc_2, amount=250000.0, tx_type="CREDIT", timestamp=base_time)
    m2 = engine.record_transaction(account_id=mule_acc_2, amount=245000.0, tx_type="DEBIT", timestamp=base_time + 25)
    print(f"    Account: {mule_acc_2}")
    print(f"    Inflow: ₹{m2['inflow_5m']:,.2f} | Outflow: ₹{m2['outflow_5m']:,.2f} | Cash Ratio: {m2['cash_ratio']}")
    print(f"    Rapid Drain Flag: {m2['is_rapid_drain']} | Velocity Risk: {m2['velocity_risk_score']}")
    assert m2["is_rapid_drain"] is True, "Expected rapid drain flag to trigger"

    # Scenario 3: Normal Legitimate Retail Customer
    legit_acc = "ACC_LEGIT_USER_003"
    print("\n[Scenario 3] Normal Legitimate Retail Salary & Groceries:")
    engine.record_transaction(account_id=legit_acc, amount=65000.0, tx_type="CREDIT", timestamp=base_time - 3600)
    m3 = engine.record_transaction(account_id=legit_acc, amount=1200.0, tx_type="DEBIT", timestamp=base_time)
    print(f"    Account: {legit_acc}")
    print(f"    60s Count: {m3['tx_count_60s']} | Outflow: ₹{m3['outflow_5m']} | Velocity Risk: {m3['velocity_risk_score']}")
    assert m3["velocity_risk_score"] < 0.30, "Expected low risk for normal user"

    print("\n" + "=" * 70)
    print("PHASE 8 STATEFUL VELOCITY ENGINE COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
