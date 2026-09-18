# -*- coding: utf-8 -*-
import sys, os, random
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')
import fitz
from pipeline.mock_data import generate_mock_student
from pipeline.docx_filler import fill_document
from pipeline.pdf_converter import PDFConverter
from pipeline.yolo_extractor import extract_and_generate_dataset_sample

template_path = r"sample/BM10_QT_PĐT_QLĐ_LSX00_NHL15052020_DonXinChuyenDiemNgoaiNgu.docx"
output_dir = r"output_test_bm10"
os.makedirs(output_dir, exist_ok=True)

converter = PDFConverter()

for sample_idx in [1, 2]:
    random.seed(sample_idx * 100 + 42)
    st = generate_mock_student()
    doc, rec = fill_document(template_path, st)
    docx_out = os.path.join(output_dir, f"sample_{sample_idx}.docx")
    pdf_out = os.path.join(output_dir, f"sample_{sample_idx}.pdf")
    doc.save(docx_out)

    converter.convert(docx_out, pdf_out)

    doc_fitz = fitz.open(pdf_out)
    print(f"Sample {sample_idx} -> Resulting PDF Page count: {len(doc_fitz)}")
    doc_fitz.close()

    gen_files = extract_and_generate_dataset_sample(pdf_out, rec, f"sample_{sample_idx}", output_dir)
    for g in gen_files:
        print(f"   [OK] {g['sample_name']}: {g['num_bboxes']} bboxes -> {g['visualization']}")

converter.close()
print("BM10 TEST COMPLETED!")
