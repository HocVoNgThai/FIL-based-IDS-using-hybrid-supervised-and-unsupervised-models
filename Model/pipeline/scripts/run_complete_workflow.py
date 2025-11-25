# scripts/run_complete_workflow.py
#!/usr/bin/env python3
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_complete_workflow():
    print("🚀 STARTING COMPLETE 3-SESSION WORKFLOW (HYBRID LOGIC)")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # ==========================================
        # SESSION 0: Initial Training
        # ==========================================
        print("\n📚 SESSION 0: Initial Training (Labels 0,1,2)")
        print("-" * 50)
        from session0 import session0_initial_training
        
        # Nhận đủ 3 giá trị trả về
        models_session0, data_loader, pipeline0 = session0_initial_training()
        
        # ==========================================
        # SESSION 1: Unknown Detection & IL (Label 3)
        # ==========================================
        print("\n🔍 SESSION 1: Unknown Detection & IL (Label 3)")
        print("-" * 50)
        from session1 import session1_workflow
        
        # Session 1 chỉ cần trả về pipeline đã update
        pipeline1 = session1_workflow()
        
        # ==========================================
        # SESSION 2: Unknown Detection & IL (Label 4)
        # ==========================================
        print("\n🎯 SESSION 2: Unknown Detection & IL (Label 4)")  
        print("-" * 50)
        from session2 import session2_workflow
        
        # Session 2 chỉ cần trả về pipeline đã update
        pipeline2 = session2_workflow()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("=" * 70)
        print("🎉 3-SESSION WORKFLOW COMPLETED SUCCESSFULLY!")
        print(f"⏱️ Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print("=" * 70)
        
        print("\n📊 WORKFLOW SUMMARY:")
        print("  ✅ Session 0: Trained Unsup on Benign, Sup on All.")
        print("  ✅ Session 1: Detected Label 3 as Unknown -> Replayed IL.") 
        print("  ✅ Session 2: Detected Label 4 as Unknown -> Replayed IL.")
        print("  ✅ Final System: Hybrid Pipeline (XGBoost + AE/IF fallback).")
        
    except Exception as e:
        print(f"\n💥 Error in workflow: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # Tạo thư mục results nếu chưa có để tránh lỗi lưu ảnh
    os.makedirs("results", exist_ok=True)
    
    success = run_complete_workflow()
    if success:
        print("\n✨ All sessions completed successfully!")
    else:
        print("\n❌ Workflow failed. Please check the errors above.")