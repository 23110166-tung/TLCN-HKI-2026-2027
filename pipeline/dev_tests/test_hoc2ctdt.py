# -*- coding: utf-8 -*-
"""
Test script for 'TBaovaMaudangky_hoc2CTDT.docx'
"""
import sys, os, random
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')
import fitz
from pipeline.mock_data import generate_mock_student
from pipeline.docx_filler import fill_document
from pipeline.pdf_converter import PDFConverter
from pipeline.yolo_extractor import extract_and_generate_dataset_sample

template_path = r"sample/TBaovaMaudangky_hoc2CTDT.docx"
output_dir = r"output_test_hoc2ctdt"
os.makedirs(output_dir, exist_ok=True)

converter = PDFConverter()

for sample_idx in [1, 2]:
    random.seed(sample_idx * 100 + 42)
    st = generate_mock_student()
    doc, rec = fill_document(template_path, st)
    docx_out = os.path.join(output_dir, f"filled_sample{sample_idx}.docx")
    pdf_out = os.path.join(output_dir, f"filled_sample{sample_idx}.pdf")
    doc.save(docx_out)

    converter.convert(docx_out, pdf_out)

    doc_fitz = fitz.open(pdf_out)
    print(f"Sample {sample_idx} -> Resulting PDF Page count: {len(doc_fitz)}")
    page = doc_fitz[0]
    for kw in ['Ý kiến PĐT', 'Người làm đơn', 'Tp.HCM', 'Phòng Đào tạo']:
        rects = page.search_for(kw)
        print(f"   Keyword '{kw}': count={len(rects)}")
        for r in rects:
            print(f"      rect: y0={r.y0:.1f}, x0={r.x0:.1f}")
    doc_fitz.close()

    gen_files = extract_and_generate_dataset_sample(pdf_out, rec, f"sample_{sample_idx}", output_dir)
    for g in gen_files:
        print(f"   [OK] {g['sample_name']}: {g['num_bboxes']} bboxes -> {g['visualization']}")

converter.close()
print("TEST COMPLETED!")
