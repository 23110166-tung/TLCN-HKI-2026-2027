# -*- coding: utf-8 -*-
"""
Dataset Generator for Template 8: BM12_QT_PĐT_XCBHTBTHCNTN_LSX02_NHL15052020_ThayThe.docx
Generates 50 samples with 80% Train (40), 10% Val (5), 10% Test (5).
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
    
    template_path = os.path.join(BASE_DIR, "sample", "BM12_QT_PĐT_XCBHTBTHCNTN_LSX02_NHL15052020_ThayThe.docx")
    dataset_root = os.path.join(BASE_DIR, "dataset_yolo")
    temp_dir = os.path.join(dataset_root, ".temp_thay_the")
    
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
    
    print("=" * 80)
    print("🚀 BẮT ĐẦU SINH DATASET YOLOV11 - MẪU 8: ĐƠN XIN HỌC MÔN THAY THẾ (BM12)")
    print(f"📄 Mẫu nguồn: {template_path}")
    print(f"📊 Tổng số mẫu: {total_samples} (Train: {num_train} | Val: {num_val} | Test: {num_test})")
    print("=" * 80)
    
    converter = PDFConverter()
    sample_records = []
    total_bboxes = 0
    
    try:
        for idx in range(1, total_samples + 1):
            if idx <= num_train:
                split = "train"
            elif idx <= num_train + num_val:
                split = "val"
            else:
                split = "test"
                
            sample_id = f"thay_the_sample_{idx:03d}"
            random.seed(idx * 757 + 619)
            
            # Step 1: Mock Data & Word Filling
            st = generate_mock_student()
            doc, rec = fill_document(template_path, st)
            
            temp_docx = os.path.join(temp_dir, f"{sample_id}.docx")
            temp_pdf = os.path.join(temp_dir, f"{sample_id}.pdf")
            doc.save(temp_docx)
            
            # Step 2: Convert to PDF
            converter.convert(temp_docx, temp_pdf)
            
            # Verify single page
            doc_fitz = fitz.open(temp_pdf)
            page_count = len(doc_fitz)
            doc_fitz.close()
            
            # Step 3: YOLO Extraction & Annotation
            gen_files = extract_and_generate_dataset_sample(
                pdf_path=temp_pdf,
                recorded_fields=rec,
                output_base_name=sample_id,
                output_dir=temp_dir,
                dpi=300
            )
            
            for g in gen_files:
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
                sample_records.append({
                    "sample_id": sample_id,
                    "split": split,
                    "student": st['student_name'],
                    "mssv": st['student_id'],
                    "bboxes": num_box,
                    "pages": page_count
                })
                
            print(f" [{idx:02d}/{total_samples:02d}] Split: {split.upper():<5} | {sample_id} | SV: {st['student_name']:<22} | MSSV: {st['student_id']} | Boxes: {gen_files[0]['num_bboxes']:02d} | Trang: {page_count}")
            
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
    print("🎉 HOÀN THÀNH TẠO DATASET CHO MẪU 8 (50 MẪU)!")
    print(f"📁 Thư mục dataset: {dataset_root}")
    print(f"   - Train samples: {num_train} ảnh & labels")
    print(f"   - Val samples:   {num_val} ảnh & labels")
    print(f"   - Test samples:  {num_test} ảnh & labels")
    print(f"🏷️ Tổng số Bounding Boxes: {total_bboxes}")
    print(f"⏱️ Tổng thời gian thực thi: {elapsed:.2f} giây")
    print("=" * 80)

if __name__ == "__main__":
    main()
