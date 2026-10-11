"""
Execution and verification runner for Phase 9: I4C Regulatory Gateway & Emergency Brake.
"""
from datetime import datetime, timezone
import json
import time
from src.ingestion.i4c_webhook import GoldenHourEmergencyBrake, I4CTicketPayload


def main():
    print("=" * 70)
    print("STAGE 8: I4C / NCRP 1930 HELPLINE REGULATORY GATEWAY & EMERGENCY BRAKE")
    print("=" * 70)

    # 1. Initialize Emergency Brake Engine with Phase 3 Graph Topology
    print("[*] Initializing Golden Hour Emergency Brake on compiled graph...")
    brake = GoldenHourEmergencyBrake()

    # 2. Simulate Citizen Cyber Fraud Complaint on Helpline 1930
    # Citizen was scammed; stolen money was sent via UPI to account ACC_000042
    ticket = I4CTicketPayload(
        Mule_Account_Number="ACC_000042",
        IFSC_Code="BKID0001001",
        Transaction_Reference_Number="UPI/CR/20261011/987654321042",
        Timestamp_of_Incident=datetime.now(timezone.utc).isoformat(),
    )

    print("\n[*] Ingesting Simulated 1930 Helpline Ticket:")
    print(f"    - Target Mule Account:  {ticket.Mule_Account_Number}")
    print(f"    - Bank Identifier/IFSC: {ticket.IFSC_Code}")
    print(f"    - UTR / Reference ID:   {ticket.Transaction_Reference_Number}")
    print(f"    - Incident Timestamp:   {ticket.Timestamp_of_Incident}")

    # 3. Trigger Emergency Brake Execution
    print("\n[🚨 EMERGENCY BRAKE ENGAGED - ZERO LATENCY GOLDEN HOUR INTERVENTION]")
    t0 = time.perf_counter()
    result = brake.execute_lien(ticket)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"[+] Status:                     {result['status']}")
    print(f"[+] Action:                     {result['action']}")
    print(f"[+] Execution Latency:          {result['execution_latency_ms']:.3f} ms (Target < 25ms)")
    print(f"[+] Primary Account Status:     {result['primary_lien']}")
    print(f"[+] Cascaded Beneficiary Holds: {result['cascaded_holds_count']} connected accounts locked")

    # 4. Inspect Downstream Network Traversal
    net = result["downstream_network"]
    print("\n[*] Graph Traversal Multi-Hop Discovery:")
    print(f"    - 1st-Degree Recipients:    {len(net['1st_degree_transfers'])} accounts -> {net['1st_degree_transfers'][:3]}")
    print(f"    - 2nd-Degree Recipients:    {len(net['2nd_degree_transfers'])} accounts -> {net['2nd_degree_transfers'][:3]}")
    print(f"    - Colocated Emulator Nodes: {len(net['colocated_entities'])} accounts -> {net['colocated_entities'][:3]}")

    # 5. Verify Lien Status Query
    target_acc = ticket.Mule_Account_Number
    assert target_acc in brake.frozen_accounts, "Target mule must have an active lien"
    assert brake.frozen_accounts[target_acc]["status"] == "HARD_LIEN_APPLIED"

    print("\n[+] Verification Check: Target account verified in active lien register.")
    print(f"    Details: {json.dumps(brake.frozen_accounts[target_acc], indent=2)}")

    print("\n" + "=" * 70)
    print("PHASE 9 I4C REGULATORY EMERGENCY BRAKE COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
