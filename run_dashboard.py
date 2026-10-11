"""
Runner to launch The Golden Hour Defense Framework Interactive Dashboard & API.
Supports automatic port fallback if port 8000 is already in use by a background process.
"""
import os
import socket
import sys
import uvicorn


def find_available_port(preferred_port: int = 8000) -> int:
    for p in range(preferred_port, preferred_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return preferred_port


def main():
    target_port = int(sys.argv[1]) if len(sys.argv) > 1 else find_available_port(8000)

    print("=" * 70)
    print("THE GOLDEN HOUR DEFENSE FRAMEWORK: LAUNCHING INTERACTIVE SOC DASHBOARD")
    print("=" * 70)
    print(f"\n[+] Initializing FastAPI Server & Embedded Engines...")
    print(f"[+] Dashboard URL: http://127.0.0.1:{target_port}")
    print(f"[+] API Docs:      http://127.0.0.1:{target_port}/docs")
    print("\n[*] Press CTRL+C to stop the server at any time.\n")

    uvicorn.run("src.server:app", host="127.0.0.1", port=target_port, log_level="info")


if __name__ == "__main__":
    main()
