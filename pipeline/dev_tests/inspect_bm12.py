# -*- coding: utf-8 -*-
import fitz, sys
sys.stdout.reconfigure(encoding='utf-8')

doc = fitz.open(r'output_test_bm12/sample_1.pdf')
page = doc[0]
print(f"Page width={page.rect.width}, height={page.rect.height}")
for kw in ['Kính đơn', 'Kính', 'đơn', 'Bộ môn quản môn học', 'Bộ môn', 'Khoa quản môn học', 'Phòng Đào tạo', 'Đồng ý', '(Ghi rõ ý kiến', 'Lưu ý']:
    rects = page.search_for(kw)
    print(f'kw "{kw}":')
    for r in rects:
        print(f'   x0={r.x0:.1f}, y0={r.y0:.1f}, x1={r.x1:.1f}, y1={r.y1:.1f}')
