import sys, fitz
sys.stdout.reconfigure(encoding='utf-8')
doc = fitz.open(r'output_test_remaining/BAN SAO TU SO GOC/sample_1.pdf')
page = doc[0]
print(f"Page rect: width={page.rect.width}, height={page.rect.height}")
for kw in ['Chân thành cám ơn', 'Chân thành', 'Người làm đơn', 'Tp. Hồ Chí Minh', 'xét và giải quyết']:
    rects = page.search_for(kw)
    print(f'Keyword "{kw}":')
    for r in rects:
        print(f'   rect: x0={r.x0:.1f}, y0={r.y0:.1f}, x1={r.x1:.1f}, y1={r.y1:.1f}')
