import sys, fitz
sys.stdout.reconfigure(encoding='utf-8')
doc = fitz.open(r'output_test_diem_i/filled_sample1.pdf')
page = doc[0]
for kw in ['Sinh viên', '(Ký và ghi rõ họ tên)', 'Ký và ghi rõ họ tên', 'TRƯỞNG KHOA', 'TRƯỞNG PHÒNG ĐÀO TẠO', 'Tp.HCM', 'Xác nhận điểm']:
    rects = page.search_for(kw)
    print(f'Keyword "{kw}": count={len(rects)}')
    for r in rects:
        print(f'   rect: x0={r.x0:.1f}, y0={r.y0:.1f}, x1={r.x1:.1f}, y1={r.y1:.1f}')
