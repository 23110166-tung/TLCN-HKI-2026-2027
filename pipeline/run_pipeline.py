# -*- coding: utf-8 -*-
"""
Main Pipeline Runner for Academic Forms YOLOv11 Dataset Generation
Executes 2 samples per DOCX template, generates scan images, YOLO labels, and visualizations.
"""

import os
import sys
import time
import zipfile
import shutil
import yaml

# Add parent directory to sys.path
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
    
    sample_dir = os.path.join(BASE_DIR, "sample")
    output_dir = os.path.join(BASE_DIR, "output")
    filled_docx_dir = os.path.join(output_dir, "filled_docx")
    pdf_dir = os.path.join(output_dir, "pdf")
    
    os.makedirs(filled_docx_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    
    # 1. Discover templates
    templates = sorted([
        f for f in os.listdir(sample_dir)
        if f.endswith(".docx") and not f.startswith("~$")
    ])
    
    print("=" * 75, flush=True)
    print("🚀 KHỞI ĐỘNG PIPELINE XỬ LÝ BIỂU MẪU VÀ TẠO DATASET YOLOV11", flush=True)
    print(f"📂 Thư mục mẫu: {sample_dir}", flush=True)
    print(f"📄 Tổng số biểu mẫu DOCX tìm thấy: {len(templates)}", flush=True)
    print(f"🔢 Số lượng mẫu sinh cho mỗi biểu mẫu: 2 mẫu (Tổng: {len(templates) * 2} bộ tài liệu)", flush=True)
    print("=" * 75, flush=True)
    
    stats = []
    total_samples = 0
    total_bboxes = 0
    errors = []
    
    # Initialize high speed batch PDF converter
    pdf_converter = PDFConverter()
    
    try:
        # Process each template
        for t_idx, t_name in enumerate(templates, 1):
            t_path = os.path.join(sample_dir, t_name)
            base_name_clean = os.path.splitext(t_name)[0]
            safe_name = "".join(c for c in base_name_clean if c.isalnum() or c in ('_', '-')).strip()
            if len(safe_name) > 40:
                safe_name = safe_name[:40]
                
            print(f"\n[{t_idx:02d}/{len(templates):02d}] 📋 Đang xử lý biểu mẫu: {t_name}", flush=True)
            
            for sample_num in [1, 2]:
                sample_id = f"{safe_name}_sample{sample_num}"
                print(f"  --> Chạy mẫu {sample_num} (ID: {sample_id})...", flush=True)
                
                try:
                    # Phase 1: Mock Data Generation & Docx Filling
                    student_data = generate_mock_student()
                    filled_doc, recorded_fields = fill_document(t_path, student_data)
                    
                    filled_docx_path = os.path.join(filled_docx_dir, f"{sample_id}_filled.docx")
                    filled_doc.save(filled_docx_path)
                    print(f"      [1/4] Word Filled: {len(recorded_fields)} trường dữ liệu đã điền.", flush=True)
                    
                    # Phase 2: PDF Conversion
                    pdf_path = os.path.join(pdf_dir, f"{sample_id}.pdf")
                    pdf_converter.convert(filled_docx_path, pdf_path)
                    print(f"      [2/4] PDF Converted: {os.path.basename(pdf_path)} ({os.path.getsize(pdf_path):,} bytes)", flush=True)
                    
                    # Phase 3 & 4: YOLO Extraction, Scan Simulation, Stamp & Signature
                    gen_results = extract_and_generate_dataset_sample(
                        pdf_path=pdf_path,
                        recorded_fields=recorded_fields,
                        output_base_name=sample_id,
                        output_dir=output_dir,
                        dpi=200
                    )
                    
                    sample_bboxes = sum(r["num_bboxes"] for r in gen_results)
                    total_bboxes += sample_bboxes
                    total_samples += len(gen_results)
                    
                    print(f"      [3/4 & 4/4] Hoàn tất ảnh scan & nhãn YOLO: {len(gen_results)} trang, {sample_bboxes} bounding boxes.", flush=True)
                    
                    stats.append({
                        "template": t_name,
                        "sample": sample_num,
                        "fields_inserted": len(recorded_fields),
                        "pages": len(gen_results),
                        "bboxes": sample_bboxes,
                        "status": "SUCCESS"
                    })
                    
                except Exception as e:
                    import traceback
                    err_msg = f"Lỗi tại {t_name} (Sample {sample_num}): {str(e)}"
                    print(f"      [ERROR] ❌ {err_msg}", flush=True)
                    traceback.print_exc()
                    errors.append(err_msg)
                    stats.append({
                        "template": t_name,
                        "sample": sample_num,
                        "fields_inserted": 0,
                        "pages": 0,
                        "bboxes": 0,
                        "status": f"FAILED: {str(e)}"
                    })
    finally:
        pdf_converter.close()

    # Write classes.txt and dataset.yaml
    classes_path = os.path.join(output_dir, "classes.txt")
    with open(classes_path, "w", encoding="utf-8") as f_cls:
        for cname in CLASS_NAMES:
            f_cls.write(f"{cname}\n")
            
    yaml_data = {
        "path": output_dir,
        "train": "images",
        "val": "images",
        "nc": len(CLASS_NAMES),
        "names": {i: name for i, name in enumerate(CLASS_NAMES)}
    }
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f_yaml:
        yaml.dump(yaml_data, f_yaml, allow_unicode=True, sort_keys=False)

    print("\n" + "=" * 75, flush=True)
    print("📦 ĐANG ĐÓNG GÓI TẬP DỮ LIỆU THÀNH FILE .ZIP...", flush=True)
    
    zip_output_path = os.path.join(BASE_DIR, "dataset_output.zip")
    with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                abs_file = os.path.join(root, file)
                rel_path = os.path.relpath(abs_file, BASE_DIR)
                zipf.write(abs_file, rel_path)
                
    zip_size_mb = os.path.getsize(zip_output_path) / (1024 * 1024)
    elapsed = time.time() - start_time
    
    print(f"✅ Đã tạo file zip thành công: {zip_output_path} ({zip_size_mb:.2f} MB)", flush=True)
    print("=" * 75, flush=True)
    print("📊 BÁO CÁO THỐNG KÊ KẾT QUẢ:", flush=True)
    print(f"  - Tổng số biểu mẫu xử lý: {len(templates)}", flush=True)
    print(f"  - Tổng số mẫu scan sinh ra: {total_samples} trang ảnh", flush=True)
    print(f"  - Tổng số Bounding Box (YOLOv11 labels): {total_bboxes} nhãn", flush=True)
    print(f"  - Tổng thời gian thực thi: {elapsed:.2f} giây", flush=True)
    print(f"  - Trạng thái lỗi: {len(errors)} lỗi", flush=True)
    if errors:
        for err in errors:
            print(f"    * {err}", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    main()
