# -*- coding: utf-8 -*-
import docx, sys, os, random
from docx.shared import Inches, Pt

sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

from pipeline.mock_data import generate_mock_student
from pipeline.docx_filler import fill_document
from pipeline.pdf_converter import PDFConverter
from pipeline.yolo_extractor import extract_and_generate_dataset_sample
import fitz

template_path = r"sample/BM12_QT_PĐT_XCBHTBTHCNTN_LSX02_NHL15052020_ThayThe.docx"
output_dir = r"output_test_bm12"
os.makedirs(output_dir, exist_ok=True)

# Let's test filling BM12
converter = PDFConverter()
for s_idx in [1, 2]:
    random.seed(s_idx * 100 + 42)
    st = generate_mock_student()
    doc, rec = fill_document(template_path, st)
    docx_out = os.path.join(output_dir, f"sample_{s_idx}.docx")
    pdf_out = os.path.join(output_dir, f"sample_{s_idx}.pdf")
    doc.save(docx_out)
    converter.convert(docx_out, pdf_out)
    
    doc_fitz = fitz.open(pdf_out)
    print(f"Sample {s_idx} -> Page count: {len(doc_fitz)}")
    doc_fitz.close()
    
    gen_files = extract_and_generate_dataset_sample(pdf_out, rec, f"sample_{s_idx}", output_dir)
    for g in gen_files:
        print(f"   [OK] {g['sample_name']}: {g['num_bboxes']} bboxes -> {g['visualization']}")

converter.close()
print("BM12 INITIAL TEST COMPLETED!")
