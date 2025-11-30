# scripts/run_complete_workflow.py
#!/usr/bin/env python3
import os
import sys
import time
import joblib
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import ResourceTracker, ILMetrics, plot_resource_usage, plot_il_matrix

def run_complete_workflow():
    print("🚀 STARTING COMPLETE WORKFLOW WITH IL METRICS & RESOURCE TRACKING")
    print("=" * 70)
    
    tracker = ResourceTracker()
    il_metrics = ILMetrics()
    resource_log = {}
    
    os.makedirs("results/overall", exist_ok=True)

    try:
        # --- SESSION 0 ---
        print("\n📚 SESSION 0: Initial Training")
        tracker.start()
        from session0 import session0_initial_training
        # Session 0 trả về models, loader, pipeline, và accuracy trên test set S0
        models0, loader, pipeline0, acc_s0 = session0_initial_training()
        resource_log['Session 0'] = tracker.stop()
        il_metrics.record(0, 0, acc_s0)
        
        # --- SESSION 1 ---
        print("\n🔍 SESSION 1: Detection (Label 3) & IL")
        tracker.start()
        from session1 import session1_workflow
        # Session 1 trả về pipeline mới và list accuracy [acc_on_S0, acc_on_S1]
        pipeline1, accs_s1 = session1_workflow()
        resource_log['Session 1'] = tracker.stop()
        
        il_metrics.record(1, 0, accs_s1['S0']) # Test lại trên data cũ
        il_metrics.record(1, 1, accs_s1['S1']) # Test trên data mới
        
        # --- SESSION 2 ---
        print("\n🎯 SESSION 2: Detection (Label 4) & IL")
        tracker.start()
        from session2 import session2_workflow
        pipeline2, accs_s2 = session2_workflow()
        resource_log['Session 2'] = tracker.stop()
        
        il_metrics.record(2, 0, accs_s2['S0'])
        il_metrics.record(2, 1, accs_s2['S1'])
        il_metrics.record(2, 2, accs_s2['S2'])
        
        # --- SUMMARY & VISUALIZATION ---
        print("\n📊 GENERATING FINAL REPORTS...")
        
        # 1. Resource Usage
        plot_resource_usage(resource_log, "results/overall/resource_usage.png")
        
        # 2. IL Metrics (Forgetting & Transfer)
        plot_il_matrix(il_metrics, "results/overall/il_accuracy_matrix.png")
        
        bwt = il_metrics.calculate_bwt(2)
        print(f"\n=== FINAL STATISTICS ===")
        print(f"Catastrophic Forgetting (BWT): {bwt:.4f}")
        print("(Negative value means forgetting, close to 0 is good)")
        
        print("=" * 70)
        print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__": run_complete_workflow()