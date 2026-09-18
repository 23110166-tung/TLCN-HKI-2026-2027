# -*- coding: utf-8 -*-
import docx, fitz, sys
sys.stdout.reconfigure(encoding='utf-8')

doc_docx = docx.Document(r'sample/Đơn xin nhận và trả điểm I.docx')
for i, p in enumerate(doc_docx.paragraphs):
    print(f"P{i:02d}: {repr(p.text)}")

print("\n--- PDF Search ---")
doc_pdf = fitz.open(r'output_test_diem_i/filled_sample1.pdf')
page = doc_pdf[0]
for kw in ['TRƯỞNG KHOA', 'TRƯỞNG PHÒNG ĐÀO TẠO', 'GV chấm thi 1', 'GV chấm thi 2', 'Sinh viên', 'Bộ môn', 'Xác nhận điểm:']:
    rects = page.search_for(kw)
    print(f"kw '{kw}':")
    for r in rects:
        print(f"   x0={r.x0:.1f}, y0={r.y0:.1f}, x1={r.x1:.1f}, y1={r.y1:.1f}")
