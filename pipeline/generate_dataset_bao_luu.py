# -*- coding: utf-8 -*-
"""
Dataset Generator for Template 5: Don xin bao luu.docx
Generates 50 image samples with 80% Train (40), 10% Val (5), 10% Test (5).
"""
import os
import sys
import time
import random
import yaml
import shutil
import fitz

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline.mock_data import generate_mock_student
from pipeline.docx_filler import fill_document
from pipeline.pdf_converter import PDFConverter
from pipeline.yolo_extractor import extract_and_generate_dataset_sample
from pipeline.ontology import CLASS_NAMES

def main():
    start_time = time.time()
    sys.stdout.reconfigure(encoding='utf-8')
    
    template_path = os.path.join(BASE_DIR, "sample", "Don xin bao luu.docx")
    dataset_root = os.path.join(BASE_DIR, "dataset_yolo")
    temp_dir = os.path.join(dataset_root, ".temp_bao_luu")
    
    splits = ["train", "val", "test"]
    for s in splits:
        os.makedirs(os.path.join(dataset_root, "images", s), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, "labels", s), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, "visualizations", s), exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    target_train = 40
    target_val = 5
    target_test = 5
    total_target = target_train + target_val + target_test
    
    print("=" * 80)
    print("🚀 BẮT ĐẦU SINH DATASET YOLOV11 - MẪU 5: ĐƠN XIN BẢO LƯU ĐIỂM")
    print(f"📄 Mẫu nguồn: {template_path}")
    print(f"📊 Tổng số ảnh mục tiêu: {total_target} (Train: {target_train} | Val: {target_val} | Test: {target_test})")
    print("=" * 80)
    
    converter = PDFConverter()
    total_bboxes = 0
    generated_counts = {"train": 0, "val": 0, "test": 0}
    run_idx = 0
    
    try:
        while sum(generated_counts.values()) < total_target:
            run_idx += 1
            sample_id = f"bao_luu_doc_{run_idx:03d}"
            random.seed(run_idx * 439 + 17)
            
            # Step 1: Mock Data & Word Filling
            st = generate_mock_student()
            doc, rec = fill_document(template_path, st)
            
            temp_docx = os.path.join(temp_dir, f"{sample_id}.docx")
            temp_pdf = os.path.join(temp_dir, f"{sample_id}.pdf")
            doc.save(temp_docx)
            
            # Step 2: Convert to PDF
            converter.convert(temp_docx, temp_pdf)
            
            # Step 3: YOLO Extraction & Annotation
            gen_files = extract_and_generate_dataset_sample(
                pdf_path=temp_pdf,
                recorded_fields=rec,
                output_base_name=f"bao_luu_sample_{run_idx:03d}",
                output_dir=temp_dir,
                dpi=300
            )
            
            for g in gen_files:
                if generated_counts["train"] < target_train:
                    split = "train"
                elif generated_counts["val"] < target_val:
                    split = "val"
                elif generated_counts["test"] < target_test:
                    split = "test"
                else:
                    break
                    
                img_src = g["image"]
                lbl_src = g["label"]
                vis_src = g["visualization"]
                num_box = g["num_bboxes"]
                
                img_dst = os.path.join(dataset_root, "images", split, os.path.basename(img_src))
                lbl_dst = os.path.join(dataset_root, "labels", split, os.path.basename(lbl_src))
                vis_dst = os.path.join(dataset_root, "visualizations", split, os.path.basename(vis_src))
                
                shutil.copy2(img_src, img_dst)
                shutil.copy2(lbl_src, lbl_dst)
                shutil.copy2(vis_src, vis_dst)
                
                total_bboxes += num_box
                generated_counts[split] += 1
                curr_total = sum(generated_counts.values())
                print(f" [{curr_total:02d}/{total_target:02d}] Split: {split.upper():<5} | {os.path.basename(img_src)} | SV: {st['student_name']:<22} | MSSV: {st['student_id']} | Boxes: {num_box:02d}")
            
    finally:
        converter.close()
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
    print("🎉 HOÀN THÀNH TẠO DATASET CHO MẪU 5 (50 MẪU)!")
    print(f"📁 Thư mục dataset: {dataset_root}")
    print(f"   - Train samples: {generated_counts['train']} ảnh & labels")
    print(f"   - Val samples:   {generated_counts['val']} ảnh & labels")
    print(f"   - Test samples:  {generated_counts['test']} ảnh & labels")
    print(f"🏷️ Tổng số Bounding Boxes: {total_bboxes}")
    print(f"⏱️ Tổng thời gian thực thi: {elapsed:.2f} giây")
    print("=" * 80)

if __name__ == "__main__":
    main()
