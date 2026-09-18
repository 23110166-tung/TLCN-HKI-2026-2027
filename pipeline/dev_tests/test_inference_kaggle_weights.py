# -*- coding: utf-8 -*-
import os
import sys
import json

import sys
sys.path.insert(0, r"c:\Users\ztung\Downloads\TLCN")

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pipeline.inference_engine import DocumentLayoutAndOCR

def main():
    yolo_weight = r"c:\Users\ztung\Downloads\TLCN\HCMUTE_DocAI_YOLOv11_VietOCR_Artifacts\runs\detect\hcmute_yolo11_layout\weights\best.pt"
    engine = DocumentLayoutAndOCR(yolo_model_path=yolo_weight, device="cpu", conf_threshold=0.35)

    test_synthetic = r"c:\Users\ztung\Downloads\TLCN\dataset_yolo\images\test\ban_sao_sample_046.png"
    test_real = r"c:\Users\ztung\Downloads\TLCN\dataset_yolo_real_world\images\REAL_001_IMG_2531.jpg"

    print("\n================== 1. SYNTHETIC TEST SAMPLE ==================")
    res_syn = engine.process_document(test_synthetic, save_visualization_path="syn_test_vis.png")
    det_count_syn = len(res_syn["raw_detections"])
    print(f"Detected {det_count_syn} elements:")
    print("Entities:")
    for k, v in res_syn["entities"].items():
        print(f"  {k}: {v}")
    print("Verification:", json.dumps(res_syn["verification"], ensure_ascii=False))

    print("\n================== 2. REAL CAMERA TEST SAMPLE ==================")
    res_real = engine.process_document(test_real, save_visualization_path="real_test_vis.png")
    det_count_real = len(res_real["raw_detections"])
    print(f"Detected {det_count_real} elements:")
    print("Entities:")
    for k, v in res_real["entities"].items():
        print(f"  {k}: {v}")
    print("Verification:", json.dumps(res_real["verification"], ensure_ascii=False))

if __name__ == "__main__":
    main()
