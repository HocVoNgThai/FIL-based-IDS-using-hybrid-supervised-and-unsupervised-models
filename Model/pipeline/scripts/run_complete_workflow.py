# scripts/run_complete_workflow.py
#!/usr/bin/env python3
import os
import sys
import time
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    ResourceTracker, ILMetrics, 
    plot_resource_usage, # [OK] Đã có trong utils
    plot_il_matrix, # [OK] Đã sửa tên
    plot_il_metrics_trends, # [New]
    plot_detailed_resource_usage, # [New]
    plot_all_models_performance, # [New]
    plot_unknown_detection_performance # [New]
)

def run_complete_workflow():
    print("🚀 STARTING COMPLETE WORKFLOW")
    tracker = ResourceTracker()
    il_metrics = ILMetrics()
    global_history = {} # Lưu metrics của tất cả models qua các Scenario
    unknown_stats = {}  # Lưu metrics unknown
    os.makedirs("results/overall", exist_ok=True)

    try:
        # === Scenario 0 ===
        print("\n📚 Scenario 0"); tracker.start()
        from session0 import session0_initial_training
        models0, loader, pipeline0, acc_s0, metrics_s0 = session0_initial_training()
        res0 = tracker.stop()
        
        il_metrics.record(0, 0, acc_s0)
        il_metrics.calculate_metrics(0)
        global_history['Scenario 0'] = metrics_s0
        plot_detailed_resource_usage(res0['history'], "Scenario 0", "results/Scenario0/resource_detail.png")
        
        # === Scenario 1 ===
        print("\n🔍 Scenario 1"); tracker.start()
        from session1 import session1_workflow
        pipeline1, results_s1 = session1_workflow()
        res1 = tracker.stop()
        
        il_metrics.record(1, 0, results_s1['acc_s0'])
        il_metrics.record(1, 1, results_s1['acc_s1'])
        il_metrics.calculate_metrics(1)
        global_history['Scenario 1'] = results_s1['metrics']
        unknown_stats['Scenario 1'] = results_s1['unknown_stats']
        plot_detailed_resource_usage(res1['history'], "Scenario 1", "results/Scenario1/resource_detail.png")
        
        # === Scenario 2 ===
        print("\n🎯 Scenario 2"); tracker.start()
        from session2 import session2_workflow
        pipeline2, results_s2 = session2_workflow()
        res2 = tracker.stop()
        
        il_metrics.record(2, 0, results_s2['acc_s0'])
        il_metrics.record(2, 1, results_s2['acc_s1'])
        il_metrics.record(2, 2, results_s2['acc_s2'])
        il_metrics.calculate_metrics(2)
        global_history['Scenario 2'] = results_s2['metrics']
        unknown_stats['Scenario 2'] = results_s2['unknown_stats']
        plot_detailed_resource_usage(res2['history'], "Scenario 2", "results/Scenario2/resource_detail.png")
        
        # === PLOTTING GLOBAL ===
        print("\n📊 Generating Global Charts...")
        plot_il_metrics_trends(il_metrics, "results/overall/il_trends.png")
        plot_il_matrix(il_metrics, "results/overall/il_matrix.png")
        plot_all_models_performance(global_history, "results/overall")
        plot_unknown_detection_performance(unknown_stats, "results/overall")
        
        rlog = {'Scenario 0': res0, 'Scenario 1': res1, 'Scenario 2': res2}
        plot_resource_usage(rlog, "results/overall/resources_summary.png")
        
        print(f"\nFinal BWT: {il_metrics.history['BWT'][-1]:.4f}")
        print("🎉 DONE!")
        
    except Exception as e: print(f"💥 Error: {e}"); import traceback; traceback.print_exc()

if __name__ == "__main__": run_complete_workflow()