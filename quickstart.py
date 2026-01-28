"""
Quick Start Script for Real-Time NIDS
Simple demonstration of real-time capabilities
"""

import time
import sys
from datetime import datetime
import numpy as np
from sklearn.datasets import make_classification

from realtime_nids import RealTimeNIDS


def quick_demo():
    """Quick demonstration of real-time NIDS"""

    print("\n" + "="*70)
    print("🛡️  REAL-TIME NETWORK INTRUSION DETECTION SYSTEM")
    print("="*70)
    print("Quick Start Demo - 60 Second Test")
    print("="*70)

    # Initialize system
    print("\n[1/4] Initializing Real-Time NIDS...")
    nids = RealTimeNIDS()

    # Generate and train ML model
    print("\n[2/4] Training ML Model...")
    print("      (In production, use real training data)")

    X_train, y_train = make_classification(
        n_samples=1000,
        n_features=8,
        n_classes=2,
        weights=[0.7, 0.3],  # 70% normal, 30% attacks
        random_state=42
    )

    nids.train_ml_model(X_train, y_train)
    print("      ✅ ML model trained successfully")

    # Start real-time detection
    print("\n[3/4] Starting Real-Time Detection...")
    print("      Mode: SIMULATION (safe for testing)")
    print("      Duration: 60 seconds")
    print("      Press Ctrl+C to stop early\n")
    print("-" * 70)

    try:
        nids.start(use_real_capture=False, duration=60)
    except KeyboardInterrupt:
        print("\n\n[!] Stopped by user")
        nids.stop()

    # Summary
    print("\n[4/4] Demo Complete!")
    print("="*70)
    print("\n📊 What you just saw:")
    print("  • Real-time packet processing")
    print("  • Hybrid detection (Signature + ML)")
    print("  • Automatic alert generation")
    print("  • Database logging")
    print("  • Multi-threaded analysis")

    print("\n🚀 Next Steps:")
    print("  1. Run with real capture: Set use_real_capture=True")
    print("  2. Train with real data: Load NSL-KDD dataset")
    print("  3. Start web dashboard: python dashboard.py")
    print("  4. Configure alerts: Edit alert_config.json")
    print("  5. View logs: Check nids_realtime.db")

    print("\n📖 Full Documentation: See REALTIME_README.md")
    print("="*70 + "\n")


def live_capture_demo():
    """Demo with real packet capture (requires permissions)"""

    print("\n" + "="*70)
    print("🛡️  REAL-TIME NIDS - LIVE CAPTURE MODE")
    print("="*70)
    print("⚠️  WARNING: Requires administrator/root privileges")
    print("="*70)

    response = input("\nDo you have the required permissions? (yes/no): ")
    if response.lower() != 'yes':
        print("\n[!] Aborting. Run as administrator or use simulation mode.")
        return

    # Initialize
    print("\n[*] Initializing system...")
    nids = RealTimeNIDS()

    # Train model
    print("[*] Training ML model...")
    X_train, y_train = make_classification(
        n_samples=1000,
        n_features=8,
        n_classes=2,
        random_state=42
    )
    nids.train_ml_model(X_train, y_train)

    # Configure for live capture
    interface = input("\nEnter network interface (or press Enter for all): ").strip()
    if interface:
        nids.config['capture']['interface'] = interface

    filter_bpf = input("Enter BPF filter (or press Enter for none): ").strip()
    if filter_bpf:
        nids.config['capture']['filter_bpf'] = filter_bpf

    duration = input("Duration in seconds (or press Enter for 60): ").strip()
    duration = int(duration) if duration else 60

    print(f"\n[*] Starting live capture for {duration} seconds...")
    print("[*] Monitoring network traffic...")
    print("-" * 70 + "\n")

    try:
        nids.start(use_real_capture=True, duration=duration)
    except KeyboardInterrupt:
        print("\n\n[!] Stopped by user")
        nids.stop()
    except Exception as e:
        print(f"\n[!] Error: {e}")
        print("[*] Tip: Make sure you have proper permissions")


def dashboard_info():
    """Show dashboard information"""

    print("\n" + "="*70)
    print("🖥️  WEB DASHBOARD QUICK START")
    print("="*70)

    print("""
To run the web dashboard:

1. Open a new terminal

2. Run the dashboard:
   python dashboard.py

3. Open your browser:
   http://localhost:5000

4. You'll see:
   • Real-time statistics
   • Live alerts
   • Attack distribution charts
   • WebSocket updates every 2 seconds

The dashboard connects to the database to show:
   • Total packets processed
   • Intrusions detected
   • Alert severity distribution
   • Recent security alerts

""")
    print("="*70 + "\n")


def main_menu():
    """Interactive menu"""

    while True:
        print("\n" + "="*70)
        print("🛡️  REAL-TIME NIDS - QUICK START MENU")
        print("="*70)
        print("\nChoose an option:")
        print("  1. Quick Demo (60 seconds, simulation mode)")
        print("  2. Live Capture Demo (requires admin/root)")
        print("  3. Web Dashboard Info")
        print("  4. Run Full Test Suite")
        print("  5. Exit")
        print()

        choice = input("Enter choice (1-5): ").strip()

        if choice == '1':
            quick_demo()
        elif choice == '2':
            live_capture_demo()
        elif choice == '3':
            dashboard_info()
        elif choice == '4':
            print("\n[*] Running full test suite...")
            import test_realtime
            test_realtime.main()
        elif choice == '5':
            print("\n👋 Goodbye!\n")
            break
        else:
            print("\n[!] Invalid choice. Please try again.")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo':
            quick_demo()
        elif sys.argv[1] == '--live':
            live_capture_demo()
        elif sys.argv[1] == '--dashboard':
            dashboard_info()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python quickstart.py [--demo|--live|--dashboard]")
    else:
        main_menu()
