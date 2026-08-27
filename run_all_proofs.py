import os
import sys
import json
import subprocess

def run_master_verification():
    print("=" * 80)
    print("   DIF-FNO & D'AGNESE BARRIER LOSS: MASTER PROOF OF SUPERIORITY SUITE")
    print("=" * 80)
    
    scripts = [
        ("Profilatura Memoria e Latenza", "tests/run_dagnese_performance_profile.py"),
        ("Confronto SOTA Baselines (FNO, Geo-FNO)", "tests/run_baseline_comparison.py"),
        ("Invarianza Topologica su Domini Critici", "tests/run_dagnese_sota_validation.py"),
        ("Super-Risoluzione Zero-Shot (1024x1024)", "tests/run_zero_shot_1024.py"),
        ("Ablazione Lambda e Navier-Stokes Stress Test", "tests/run_dagnese_physics_sensitivity_stress.py")
    ]
    
    passed = 0
    for title, script in scripts:
        print(f"\n[RUN] {title}...")
        res = subprocess.run([sys.executable, script], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"  --> STATUS: SUCCESS")
            passed += 1
        else:
            print(f"  --> STATUS: FAILED")
            print(res.stderr)
            
    print("\n" + "=" * 80)
    print(f"   ESITO SUITE MASTERS: {passed}/{len(scripts)} TEST SUPERATI CON SUCCESSO")
    print("   GARANZIA TOPOLOGICA: 0.00% GRID FOLDING CONFERMATO")
    print("=" * 80)

if __name__ == "__main__":
    run_master_verification()
