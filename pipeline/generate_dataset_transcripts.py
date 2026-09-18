# -*- coding: utf-8 -*-
"""
Full Dataset Generator for Academic Transcripts (Bảng Điểm Học Tập)
Generates 50 samples with 80% Train (40), 10% Val (5), 10% Test (5).
Covers cohorts K20 to K26 across 6 diverse engineering & business majors.
"""

import os
import sys
import time
import random
import yaml
import shutil
from selenium import webdriver
from selenium.webdriver.edge.options import Options

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline.mock_data import LAST_NAMES, MIDDLE_NAMES, FIRST_NAMES_MALE, FIRST_NAMES_FEMALE
from pipeline.transcript_generator import generate_transcript_sample_with_driver, EDGE_EXE
from pipeline.ontology import CLASS_NAMES

def generate_random_name():
    last = random.choice(LAST_NAMES)
    mid = random.choice(MIDDLE_NAMES)
    first = random.choice(FIRST_NAMES_MALE + FIRST_NAMES_FEMALE)
    return f"{last} {mid} {first}"

def main():
    start_time = time.time()
    sys.stdout.reconfigure(encoding='utf-8')
    
    dataset_root = os.path.join(BASE_DIR, "dataset_yolo_transcripts")
    temp_dir = os.path.join(dataset_root, ".temp_transcripts")
    
    splits = ["train", "val", "test"]
    for s in splits:
        os.makedirs(os.path.join(dataset_root, "images", s), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, "labels", s), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, "visualizations", s), exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    num_train = 40
    num_val = 5
    num_test = 5
    total_samples = num_train + num_val + num_test
    
    majors = ["AI", "CNTT", "DDT", "CK", "OTO", "QTKD"]
    cohort_pool = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    
    print("=" * 80)
    print("🚀 BẮT ĐẦU SINH 50 DATASET BẢNG ĐIỂM HỌC TẬP TRỰC TUYẾN (YOLOV11)")
    print(f"📊 Phân chia: Train ({num_train}) | Val ({num_val}) | Test ({num_test})")
    print(f"🎓 Bao phủ các khóa: K20, K21, K22, K23, K24, K25, K26")
    print(f"🏢 Các ngành: AI, CNTT, DDT, CK, OTO, QTKD")
    print("=" * 80)
    
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.binary_location = EDGE_EXE
    
    driver = webdriver.Edge(options=options)
    total_bboxes = 0
    sample_records = []
    
    try:
        for idx in range(1, total_samples + 1):
            if idx <= num_train:
                split = "train"
            elif idx <= num_train + num_val:
                split = "val"
            else:
                split = "test"
                
            sample_id = f"transcript_sample_{idx:03d}"
            random.seed(idx * 7919 + 104729)
            
            # Select cohort and major
            cohort = cohort_pool[(idx - 1) % len(cohort_pool)]
            major = majors[(idx - 1) % len(majors)]
            is_clc = random.choice([True, False])
            
            name = generate_random_name()
            mssv = f"{str(cohort)[2:]}{random.randint(110000, 149999)}"
            
            res = generate_transcript_sample_with_driver(
                driver=driver,
                student_name=name,
                student_id=mssv,
                cohort_year=cohort,
                major_key=major,
                output_base_name=sample_id,
                output_dir=temp_dir,
                is_clc=is_clc
            )
            
            img_src = res["image"]
            lbl_src = res["label"]
            vis_src = res["visualization"]
            num_box = res["num_bboxes"]
            
            img_dst = os.path.join(dataset_root, "images", split, os.path.basename(img_src))
            lbl_dst = os.path.join(dataset_root, "labels", split, os.path.basename(lbl_src))
            vis_dst = os.path.join(dataset_root, "visualizations", split, os.path.basename(vis_src))
            
            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)
            shutil.copy2(vis_src, vis_dst)
            
            total_bboxes += num_box
            sample_records.append({
                "sample_id": sample_id,
                "split": split,
                "cohort": f"K{str(cohort)[2:]}",
                "major": major,
                "student": name,
                "mssv": mssv,
                "bboxes": num_box,
                "size": res["dimensions"]
            })
            
            print(f" [{idx:02d}/{total_samples:02d}] Split: {split.upper():<5} | {sample_id} | K{str(cohort)[2:]} - {major:<5} | SV: {name:<22} | MSSV: {mssv} | Boxes: {num_box:03d}")
            
    finally:
        driver.quit()
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    yaml_content = {
        "path": dataset_root.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": {i: name for i, name in enumerate(CLASS_NAMES)}
    }
    
    yaml_path = os.path.join(dataset_root, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f_y:
        yaml.dump(yaml_content, f_y, allow_unicode=True, sort_keys=False)
        
    elapsed = time.time() - start_time
    print("=" * 80)
    print("🎉 HOÀN THÀNH TẠO 50 BẢNG ĐIỂM HỌC TẬP DATASET!")
    print(f"📁 Thư mục dataset: {dataset_root}")
    print(f"   - Train: {num_train} ảnh & nhãn")
    print(f"   - Val:   {num_val} ảnh & nhãn")
    print(f"   - Test:  {num_test} ảnh & nhãn")
    print(f"🏷️ Tổng số Bounding Boxes: {total_bboxes}")
    print(f"⏱️ Tổng thời gian thực thi: {elapsed:.2f} giây")
    print("=" * 80)

if __name__ == "__main__":
    main()
