# -*- coding: utf-8 -*-
import sys, fitz
sys.stdout.reconfigure(encoding='utf-8')

doc = fitz.open(r'output_test_diem_i/filled_sample1.pdf')
page = doc[0]
zoom = 200 / 72.0
pw, ph = page.rect.width, page.rect.height

student_sig_anchors = [
    "(Ký và ghi rõ họ tên)",
    "(Ký, ghi rõ họ tên)",
    "Ký và ghi rõ họ tên",
    "(Ký tên)",
    "Người làm đơn",
    "Sinh viên ký tên",
    "Người viết đơn",
    "Người đề nghị",
    "Sinh viên"
]

found_anchor = None
for kw in student_sig_anchors:
    rects = page.search_for(kw)
    if rects:
        # Filter for rects in lower half of page (y0 > 0.5 * ph)
        bottom_rects = [r for r in rects if r.y0 > 0.5 * ph]
        if bottom_rects:
            # If "Sinh viên", prefer one on the right side (x0 > 0.4 * pw)
            if kw == "Sinh viên":
                right_rects = [r for r in bottom_rects if r.x0 > 0.4 * pw]
                if right_rects:
                    found_anchor = right_rects[-1]
                    print(f"Matched '{kw}' (right bottom): {found_anchor}")
                    break
            else:
                found_anchor = bottom_rects[-1]
                print(f"Matched '{kw}': {found_anchor}")
                break

if found_anchor:
    sw, sh = 180, 80
    anchor_center_x = (found_anchor.x0 + found_anchor.x1) / 2.0 * zoom
    sx = int(anchor_center_x - sw / 2.0)
    sy = int(found_anchor.y1 * zoom + 10)
    print(f"Placed signature at: sx={sx}, sy={sy}, sw={sw}, sh={sh}")
