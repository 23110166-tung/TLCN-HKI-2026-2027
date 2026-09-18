# -*- coding: utf-8 -*-
import docx, sys, os, random
from docx.shared import Inches, Pt
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

from pipeline.mock_data import generate_mock_student
from pipeline.pdf_converter import PDFConverter
from pipeline.yolo_extractor import extract_and_generate_dataset_sample
import fitz

template_path = r"sample/BM10_QT_PĐT_QLĐ_LSX00_NHL15052020_DonXinChuyenDiemNgoaiNgu.docx"
output_dir = r"output_test_bm10_clean"
os.makedirs(output_dir, exist_ok=True)

def fill_bm10_perfect(template_path, st):
    doc = docx.Document(template_path)
    rec = []

    # Section margins
    for sec in doc.sections:
        sec.top_margin = Inches(0.55)
        sec.bottom_margin = Inches(0.5)
        sec.left_margin = Inches(0.65)
        sec.right_margin = Inches(0.65)

    # Paragraphs formatting & filling
    for i, p in enumerate(doc.paragraphs):
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(1.5)
        p.paragraph_format.line_spacing = 1.0
        t = p.text
        if "Họ tên sinh viên:" in t:
            p.text = f"Họ tên sinh viên:  {st['student_name']}"
            rec.append({"text": st['student_name'], "class_name": "student_name"})
        elif "MSSV:" in t and "Điện thoại:" in t:
            p.text = f"MSSV:  {st['student_id']} \t\tĐiện thoại:  {st['phone']}"
            rec.append({"text": st['student_id'], "class_name": "student_id"})
            rec.append({"text": st['phone'], "class_name": "phone"})
        elif "Lớp:" in t and "Khoa:" in t:
            p.text = f"Lớp:  {st['class_name']} \t\tKhoa:  {st['faculty']}"
            rec.append({"text": st['class_name'], "class_name": "class_name"})
            rec.append({"text": st['faculty'], "class_name": "faculty"})
        elif "Đã thi đạt chứng chỉ:" in t:
            p.text = f"Đã thi đạt chứng chỉ: TOEIC Quốc tế (IIG Việt Nam)"
            rec.append({"text": "TOEIC Quốc tế", "class_name": "degree_info"})
        elif "Với số điểm:" in t or "Nghe:" in t:
            p.text = f"Với số điểm: Nghe: {st['cert_listening']} Nói: {st['cert_speaking']} Đọc: {st['cert_reading']} Viết: {st['cert_writing']} Tổng điểm: {st['cert_total']}"
            rec.append({"text": st['cert_listening'], "class_name": "score"})
            rec.append({"text": st['cert_speaking'], "class_name": "score"})
            rec.append({"text": st['cert_reading'], "class_name": "score"})
            rec.append({"text": st['cert_writing'], "class_name": "score"})
            rec.append({"text": str(st['cert_total']), "class_name": "score"})
        elif "Tại:" in t and "Ngày cấp chứng chỉ:" in t:
            p.text = f"Tại: {st['cert_place']} Ngày cấp chứng chỉ: {st['cert_date']}"
            rec.append({"text": st['cert_place'], "class_name": "pob"})
            rec.append({"text": st['cert_date'], "class_name": "date"})
        elif "Tp. HCM, ngày" in t:
            p.text = f"                                                                      Tp. HCM, ngày {st['day']} tháng {st['month']} năm {st['year']}"
            rec.append({"text": f"ngày {st['day']} tháng {st['month']} năm {st['year']}", "class_name": "date"})
        elif "Xác nhận của Phòng Đào tạo" in t:
            p.text = "Xác nhận của Phòng Đào tạo                                         Sinh viên ký tên"
        elif not t.strip():
            p.text = ""
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = Pt(1)

    t = doc.tables[0]
    
    eng_courses = [
        ("Anh văn 1", "ENGL130137", "3", f"{random.randint(7, 9)}.{random.randint(0, 9)}"),
        ("Anh văn 2", "ENGL230237", "3", f"{random.randint(7, 9)}.{random.randint(0, 9)}"),
        ("Anh văn 3", "ENGL330337", "3", f"{random.randint(7, 9)}.{random.randint(0, 9)}")
    ]

    for row_idx in range(1, len(t.rows)):
        row = t.rows[row_idx]
        c_idx = row_idx - 1
        
        # Set trHeight to 300 dxa (approx 15pt)
        if row._tr.trPr is not None:
            for child in list(row._tr.trPr):
                if child.tag.endswith('trHeight'):
                    row._tr.trPr.remove(child)
        row._tr.get_or_add_trPr().append(parse_xml(r'<w:trHeight %s w:val="340" w:hRule="atLeast"/>' % nsdecls('w')))

        if c_idx < len(eng_courses):
            c_name, c_code, c_cred, c_score = eng_courses[c_idx]
            vals = [str(row_idx), c_name, c_code, c_cred, c_score]
            classes = [None, "course_name", "course_code", "credits", "score"]
        else:
            vals = [str(row_idx), "", "", "", ""]
            classes = [None, None, None, None, None]

        for col_idx in range(5):
            cell = row.cells[col_idx]
            val = vals[col_idx]
            cls = classes[col_idx]
            
            p = cell.paragraphs[0]
            p.text = val
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            if col_idx in [0, 2, 3, 4]:
                p.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(8.5)
            
            if val and cls:
                rec.append({"text": val, "class_name": cls})

    return doc, rec

converter = PDFConverter()
for s_idx in [1, 2]:
    random.seed(s_idx * 100 + 42)
    st = generate_mock_student()
    doc, rec = fill_bm10_perfect(template_path, st)
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
print("BM10 CLEAN TEST COMPLETED!")
