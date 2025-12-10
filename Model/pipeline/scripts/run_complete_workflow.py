# scripts/run_complete_workflow.py
#!/usr/bin/env python3
import os
import sys
import time
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import ResourceTracker, ILMetrics, plot_resource_usage, plot_il_metrics

def run_complete_workflow():
    print("🚀 STARTING COMPLETE WORKFLOW")
    tracker = ResourceTracker(); il_metrics = ILMetrics(); rlog = {}
    os.makedirs("results/overall", exist_ok=True)

    try:
        print("\n📚 SESSION 0"); tracker.start()
        from session0 import session0_initial_training
        models0, loader, pipeline0, acc_s0 = session0_initial_training()
        rlog['Case 0'] = tracker.stop()
        il_metrics.record(0, 0, acc_s0)
        
        print("\n🔍 SESSION 1"); tracker.start()
        from session1 import session1_workflow
        pipeline1, accs_s1 = session1_workflow()
        rlog['Case 1'] = tracker.stop()
        il_metrics.record(1, 0, accs_s1['S0']); il_metrics.record(1, 1, accs_s1['S1'])
        
        print("\n🎯 SESSION 2"); tracker.start()
        from session2 import session2_workflow
        pipeline2, accs_s2 = session2_workflow()
        rlog['Case 2'] = tracker.stop()
        il_metrics.record(2, 0, accs_s2['S0']); il_metrics.record(2, 1, accs_s2['S1']); il_metrics.record(2, 2, accs_s2['S2'])
        
        plot_resource_usage(rlog, "results/overall/resources.png")
        plot_il_metrics(il_metrics, "results/overall/il_metrics.png")
        print(f"\nFinal BWT: {il_metrics.calculate_bwt(2):.4f}")
        print("🎉 DONE!")
    except Exception as e: print(f"💥 Error: {e}"); import traceback; traceback.print_exc()

if __name__ == "__main__": run_complete_workflow()