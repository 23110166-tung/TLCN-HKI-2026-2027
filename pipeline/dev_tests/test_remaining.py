# -*- coding: utf-8 -*-
"""
Inspect and test remaining 4 templates:
1. BAN SAO TU SO GOC.docx
2. BM10_QT_PĐT_QLĐ_LSX00_NHL15052020_DonXinChuyenDiemNgoaiNgu.docx
3. BM12_QT_PĐT_XCBHTBTHCNTN_LSX02_NHL150520_ThayThe.docx
4. Don mien thi.docx
"""
import sys, os, random, shutil
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')
import fitz
from pipeline.mock_data import generate_mock_student
from pipeline.docx_filler import fill_document
from pipeline.pdf_converter import PDFConverter
from pipeline.yolo_extractor import extract_and_generate_dataset_sample

templates = [
    "sample/BAN SAO TU SO GOC.docx",
    "sample/BM10_QT_PĐT_QLĐ_LSX00_NHL15052020_DonXinChuyenDiemNgoaiNgu.docx",
    "sample/BM12_QT_PĐT_XCBHTBTHCNTN_LSX02_NHL15052020_ThayThe.docx",
    "sample/Don mien thi.docx"
]

out_base = "output_test_remaining"
os.makedirs(out_base, exist_ok=True)
converter = PDFConverter()

for t_path in templates:
    t_name = os.path.splitext(os.path.basename(t_path))[0]
    t_dir = os.path.join(out_base, t_name)
    os.makedirs(t_dir, exist_ok=True)
    print(f"\n=================== TESTING: {t_name} ===================")
    
    for sample_idx in [1, 2]:
        random.seed(sample_idx * 100 + 42)
        st = generate_mock_student()
        doc, rec = fill_document(t_path, st)
        docx_out = os.path.join(t_dir, f"sample_{sample_idx}.docx")
        pdf_out = os.path.join(t_dir, f"sample_{sample_idx}.pdf")
        doc.save(docx_out)
        converter.convert(docx_out, pdf_out)
        
        doc_fitz = fitz.open(pdf_out)
        print(f"[{t_name}] Sample {sample_idx} -> Page count: {len(doc_fitz)}")
        doc_fitz.close()
        
        gen_files = extract_and_generate_dataset_sample(pdf_out, rec, f"sample_{sample_idx}", t_dir)
        for g in gen_files:
            print(f"   [OK] {g['sample_name']}: {g['num_bboxes']} bboxes -> {g['visualization']}")

converter.close()
print("\nALL REMAINING TEMPLATES TESTED!")
