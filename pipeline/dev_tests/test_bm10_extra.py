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

converter = PDFConverter()

for s_idx in [3, 4]:
    random.seed(s_idx * 150 + 7)
    st = generate_mock_student()
    if s_idx == 3:
        # Test TOEIC 550
        st["cert_type"] = "TOEIC"
        st["cert_name"] = "TOEIC Quốc tế (IIG Việt Nam)"
        st["cert_listening"] = "290"
        st["cert_speaking"] = "---"
        st["cert_reading"] = "260"
        st["cert_writing"] = "---"
        st["cert_total"] = "550"
    elif s_idx == 4:
        # Test IELTS 5.5
        st["cert_type"] = "IELTS"
        st["cert_name"] = "IELTS Academic (IDP Education)"
        st["cert_listening"] = "6.0"
        st["cert_speaking"] = "5.5"
        st["cert_reading"] = "5.5"
        st["cert_writing"] = "5.0"
        st["cert_total"] = "5.5"

    doc, rec = fill_document(template_path, st)
    docx_out = os.path.join(output_dir, f"sample_{s_idx}.docx")
    pdf_out = os.path.join(output_dir, f"sample_{s_idx}.pdf")
    doc.save(docx_out)
    converter.convert(docx_out, pdf_out)

    doc_fitz = fitz.open(pdf_out)
    print(f"Sample {s_idx} -> Resulting PDF Page count: {len(doc_fitz)}")
    doc_fitz.close()

    gen_files = extract_and_generate_dataset_sample(pdf_out, rec, f"sample_{s_idx}", output_dir)
    for g in gen_files:
        print(f"   [OK] {g['sample_name']}: {g['num_bboxes']} bboxes -> {g['visualization']}")

converter.close()
