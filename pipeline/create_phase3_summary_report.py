# -*- coding: utf-8 -*-
"""
Script to generate Phase 3 Summary Report for HCMUTE DocAI Thesis.
Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh.
Giai đoạn 3: Nghiên cứu, Thử nghiệm Mô hình Học sâu & Thẩm định Quy chế Học vụ.
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline.pdf_converter import PDFConverter

SCHOOL_NAME = "Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh"


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=50, bottom=50, left=70, right=70):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)


def set_cell_border(cell, top="CCCCCC", bottom="CCCCCC", left=None, right=None):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = ['<w:tcBorders ' + nsdecls('w') + '>']
    borders.append(f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{top}"/>' if top else '<w:top w:val="none"/>')
    borders.append(f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="{bottom}"/>' if bottom else '<w:bottom w:val="none"/>')
    borders.append(f'<w:left w:val="single" w:sz="4" w:space="0" w:color="{left}"/>' if left else '<w:left w:val="none"/>')
    borders.append(f'<w:right w:val="single" w:sz="4" w:space="0" w:color="{right}"/>' if right else '<w:right w:val="none"/>')
    borders.append('</w:tcBorders>')
    tcPr.append(parse_xml(''.join(borders)))


def fmt(p, sb=0, sa=2, ls=1.1, align=WD_ALIGN_PARAGRAPH.LEFT):
    p.paragraph_format.space_before = Pt(sb)
    p.paragraph_format.space_after = Pt(sa)
    p.paragraph_format.line_spacing = ls
    p.alignment = align


def heading1(doc, text):
    p = doc.add_paragraph()
    fmt(p, sb=8, sa=3, align=WD_ALIGN_PARAGRAPH.LEFT)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 51, 102)
    return p


def heading2(doc, text):
    p = doc.add_paragraph()
    fmt(p, sb=5, sa=2, align=WD_ALIGN_PARAGRAPH.LEFT)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(10.5)
    run.font.bold = True
    run.font.color.rgb = RGBColor(16, 44, 87)
    return p


def body(doc, text, bold=False, italic=False, bullet=False, sz=9.5):
    p = doc.add_paragraph()
    fmt(p, sb=0, sa=2, ls=1.1)
    if bullet:
        p.paragraph_format.left_indent = Inches(0.18)
        rb = p.add_run("• ")
        rb.font.name = 'Times New Roman'
        rb.font.bold = True
        rb.font.color.rgb = RGBColor(0, 51, 102)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(sz)
    run.font.bold = bold
    run.font.italic = italic
    return p


def table_header_cell(cell, text, width_in, bg="003366"):
    cell.width = Inches(width_in)
    set_cell_background(cell, bg)
    set_cell_margins(cell, top=45, bottom=45, left=60, right=60)
    set_cell_border(cell, top=bg, bottom=bg, left=bg, right=bg)
    p = cell.paragraphs[0]
    fmt(p, sb=0, sa=0, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(9)
    r.font.bold = True
    r.font.color.rgb = RGBColor(255, 255, 255)


def table_data_cell(cell, text, width_in, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.LEFT, bg="FFFFFF"):
    cell.width = Inches(width_in)
    set_cell_background(cell, bg)
    set_cell_margins(cell, top=35, bottom=35, left=55, right=55)
    set_cell_border(cell, top="E0E0E0", bottom="E0E0E0", left="E0E0E0", right="E0E0E0")
    p = cell.paragraphs[0]
    fmt(p, sb=0, sa=0, align=align)
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(8.5)
    r.font.bold = bold
    r.font.italic = italic


def main():
    doc = docx.Document()

    # Set page margins (0.7 inch)
    for section in doc.sections:
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)
        section.left_margin = Inches(0.7)
        section.right_margin = Inches(0.7)

    out_docx = os.path.join(BASE_DIR, "Bao_Cao_Tong_Ket_Giai_Doan_3_HCMUTE.docx")
    out_pdf = os.path.join(BASE_DIR, "Bao_Cao_Tong_Ket_Giai_Doan_3_HCMUTE.pdf")

    # Header Top
    tbl_top = doc.add_table(rows=1, cols=2)
    tbl_top.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_top.autofit = False

    c_left = tbl_top.rows[0].cells[0]
    c_left.width = Inches(3.8)
    p_l = c_left.paragraphs[0]
    fmt(p_l, sb=0, sa=1, align=WD_ALIGN_PARAGRAPH.CENTER)
    for txt, sz, bold, italic in [
        (f"{SCHOOL_NAME.upper()}\n", 8.5, True, False),
        ("KHOA CÔNG NGHỆ THÔNG TIN\n", 8.5, True, False),
        ("BỘ MÔN TRÍ TUỆ NHÂN TẠO\n", 8.5, False, False),
        ("---------------", 8, False, False),
    ]:
        r = p_l.add_run(txt)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(sz)
        r.font.bold = bold
        r.font.italic = italic

    c_right = tbl_top.rows[0].cells[1]
    c_right.width = Inches(3.8)
    p_r = c_right.paragraphs[0]
    fmt(p_r, sb=0, sa=1, align=WD_ALIGN_PARAGRAPH.CENTER)
    for txt, sz, bold, italic in [
        ("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n", 8.5, True, False),
        ("Độc lập – Tự do – Hạnh phúc\n", 8.5, True, False),
        ("-----------------------", 8, False, False),
    ]:
        r = p_r.add_run(txt)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(sz)
        r.font.bold = bold
        r.font.italic = italic

    # Title
    p_title = doc.add_paragraph()
    fmt(p_title, sb=8, sa=2, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p_title.add_run("BÁO CÁO TỔNG KẾT TIẾN ĐỘ THỰC HIỆN GIAI ĐOẠN 3")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0, 51, 102)

    p_sub = doc.add_paragraph()
    fmt(p_sub, sb=0, sa=5, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p_sub.add_run("NGHIÊN CỨU, THỬ NGHIỆM MÔ HÌNH HỌC SÂU & THẨM ĐỊNH QUY CHẾ NGHIỆP VỤ")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(10.5)
    r.font.bold = True
    r.font.color.rgb = RGBColor(16, 44, 87)

    # General info
    t_info = doc.add_table(rows=4, cols=2)
    t_info.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_info.autofit = False

    info_data = [
        ("Tên đề tài:", "Xây dựng hệ thống nhận dạng, trích xuất và kiểm tra tính hợp lệ của thông tin từ đơn học vụ và bảng điểm sinh viên bằng học sâu"),
        ("Giảng viên hướng dẫn:", "TS. Bùi Mạnh Quân"),
        ("Sinh viên thực hiện:", "Nguyễn Bách Tùng (MSSV: 23110166 - Nhóm trưởng)\nNguyễn Hưng Nguyên (MSSV: 23110035) – Trần Cẩm Long (MSSV: 23110122)"),
        ("Thời gian thực hiện:", "Học kỳ I, Năm học 2026 - 2027 (Hoàn tất toàn bộ Giai đoạn 3)")
    ]
    for r_idx, (label, val) in enumerate(info_data):
        c1 = t_info.rows[r_idx].cells[0]
        c1.width = Inches(1.8)
        set_cell_background(c1, "F2F5F8")
        set_cell_margins(c1, 30, 30, 40, 40)
        set_cell_border(c1, "CCCCCC", "CCCCCC", "CCCCCC", "CCCCCC")
        p = c1.paragraphs[0]
        fmt(p, 0, 0)
        r = p.add_run(label)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(9)
        r.font.bold = True

        c2 = t_info.rows[r_idx].cells[1]
        c2.width = Inches(5.8)
        set_cell_background(c2, "FFFFFF")
        set_cell_margins(c2, 30, 30, 40, 40)
        set_cell_border(c2, "CCCCCC", "CCCCCC", "CCCCCC", "CCCCCC")
        p = c2.paragraphs[0]
        fmt(p, 0, 0)
        r = p.add_run(val)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(9)

    # Section 1: Overview
    heading1(doc, "I. TỔNG QUAN MỤC TIÊU VÀ KẾT QUẢ GIAI ĐOẠN 3")
    body(doc, "Trong Giai đoạn 3, nhóm đã tập trung giải quyết trọn vẹn 2 Câu hỏi Nghiên cứu (Research Questions) cốt lõi của đề tài theo đề cương đã duyệt, đồng thời phát triển hoàn chỉnh Academic Rule Engine và ứng dụng Web tương tác:")
    body(doc, "Câu hỏi Nghiên cứu 1 (RQ1): Đánh giá khả năng cải thiện độ chính xác nhận dạng văn bản tiếng Việt học vụ khi Fine-tuning mô hình Seq2Seq VietOCR (VGG-Transformer) trên tập dữ liệu đặc thù của HCMUTE.", bullet=True)
    body(doc, "Câu hỏi Nghiên cứu 2 (RQ2): Giải quyết hiện tượng suy giảm hiệu năng do dịch chuyển miền dữ liệu (Domain Shift) khi chuyển từ tài liệu tổng hợp (Synthetic) sang 120 ảnh chụp smartphone thực tế có biến dạng phối cảnh, ánh sáng và bóng đổ.", bullet=True)
    body(doc, "Academic Rule Engine độc lập: Xây dựng bộ luật 5 nhóm (R1x-R5x) kiểm tra tính hợp pháp, tính nhất quán niên khóa/thời gian và đối soát chữ ký/con dấu cho 10 loại biểu mẫu học vụ (T01-T10).", bullet=True)
    body(doc, "Ứng dụng Web tương tác (Streamlit UI): Phát triển giao diện Web hoàn chỉnh cho phép chuyên viên kéo thả ảnh, xem trực quan bounding box và bảng điều khiển thẩm định hồ sơ theo thời gian thực.", bullet=True)

    # Section 2: RQ1 VietOCR
    heading1(doc, "II. KẾT QUẢ GIẢI QUYẾT CÂU HỎI NGHIÊN CỨU 1 (RQ1 - VIETOCR)")
    body(doc, "Nhóm đã tự động trích xuất và gán nhãn 4.896 ảnh crop văn bản từ biểu mẫu thực tế, chia tách độc lập Train/Val/Test (867 ảnh test). Tiến hành Fine-tuning mô hình VGG-Transformer trong 15 epochs với AdamW.")

    t_rq1 = doc.add_table(rows=5, cols=4)
    t_rq1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_rq1.autofit = False

    rq1_headers = ["Chỉ số Đánh giá", "VietOCR Baseline", "VietOCR Fine-tuned", "Chênh lệch (Δ)"]
    for i, h in enumerate(rq1_headers):
        table_header_cell(t_rq1.rows[0].cells[i], h, [2.4, 1.7, 1.8, 1.7][i])

    rq1_rows = [
        ("Tỷ lệ khớp chính xác toàn từ (Exact Match)", "51.67%", "69.78%", "+18.11% (Tăng vượt bậc)"),
        ("Tỷ lệ lỗi ký tự (Character Error Rate - CER)", "8.34%", "3.82%", "-4.52% (Giảm hơn một nửa)"),
        ("Tỷ lệ lỗi từ (Word Error Rate - WER)", "47.46%", "48.10%", "+0.64% (Ổn định)"),
        ("Số lượng mẫu đúng tuyệt đối trên tập test", "448 / 867 mẫu", "605 / 867 mẫu", "+157 mẫu chính xác")
    ]
    for r_idx, row in enumerate(rq1_rows, start=1):
        bg = "F9FBFD" if r_idx % 2 == 1 else "FFFFFF"
        table_data_cell(t_rq1.rows[r_idx].cells[0], row[0], 2.4, bold=True, bg=bg)
        table_data_cell(t_rq1.rows[r_idx].cells[1], row[1], 1.7, align=WD_ALIGN_PARAGRAPH.CENTER, bg=bg)
        table_data_cell(t_rq1.rows[r_idx].cells[2], row[2], 1.8, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, bg=bg)
        table_data_cell(t_rq1.rows[r_idx].cells[3], row[3], 1.7, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, bg=bg)

    body(doc, "Nhận xét: VietOCR sau khi fine-tuning đã khắc phục triệt để lỗi mất dấu tiếng Việt và nhận diện chuẩn xác các từ viết tắt chuyên môn như 'ĐKTH', 'CTĐT', 'TBCHK', 'MSSV'...", sz=9, italic=True)

    # Section 3: RQ2 YOLOv11 Domain Adaptation
    heading1(doc, "III. KẾT QUẢ GIẢI QUYẾT CÂU HỎI NGHIÊN CỨU 2 (RQ2 - DOMAIN ADAPTATION)")
    body(doc, "Nhóm đã thu thập 120 ảnh chụp smartphone thực tế từ 14 mẫu biểu khác nhau, phân chia nhóm nghiêm ngặt theo Document ID (Zero Data Leakage: 85 train, 18 val, 17 test).")
    body(doc, "Đóng góp Data-Centric AI quan trọng: Phát hiện và khử triệt để 6.240 nhãn phantom/duplicate ở các cột bảng điểm; áp dụng Albumentations nâng cao (Perspective, Arm/Phone Shadow, Sensor Noise); nâng độ phân giải lên imgsz=1280 và kế thừa trọng số Pretrained Synthetic (Stage 2).")

    t_rq2 = doc.add_table(rows=5, cols=5)
    t_rq2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_rq2.autofit = False

    rq2_headers = ["Chỉ số Hiệu năng", "Zero-shot Thô", "Baseline Synthetic", "Domain-Adapted v2", "Mức cải thiện"]
    for i, h in enumerate(rq2_headers):
        table_header_cell(t_rq2.rows[0].cells[i], h, [1.8, 1.4, 1.4, 1.5, 1.5][i])

    rq2_rows = [
        ("Precision (Độ chính xác)", "23.27%", "29.19%", "40.48%", "+17.21% (Tăng 1.74x)"),
        ("Recall (Độ bao phủ)", "12.91%", "14.89%", "34.11%", "+21.20% (Tăng 2.64x)"),
        ("mAP@0.5 (Độ chuẩn)", "6.89%", "10.53%", "34.72%", "+27.83% (Tăng 5.04x)"),
        ("mAP@0.5:0.95 (Chặt chẽ)", "1.66%", "2.66%", "12.22%", "+10.56% (Tăng 7.36x)")
    ]
    for r_idx, row in enumerate(rq2_rows, start=1):
        bg = "F9FBFD" if r_idx % 2 == 1 else "FFFFFF"
        table_data_cell(t_rq2.rows[r_idx].cells[0], row[0], 1.8, bold=True, bg=bg)
        table_data_cell(t_rq2.rows[r_idx].cells[1], row[1], 1.4, align=WD_ALIGN_PARAGRAPH.CENTER, bg=bg)
        table_data_cell(t_rq2.rows[r_idx].cells[2], row[2], 1.4, align=WD_ALIGN_PARAGRAPH.CENTER, bg=bg)
        table_data_cell(t_rq2.rows[r_idx].cells[3], row[3], 1.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, bg=bg)
        table_data_cell(t_rq2.rows[r_idx].cells[4], row[4], 1.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, bg=bg)

    body(doc, "Hiệu năng chi tiết trên các trường thông tin sinh viên cốt lõi: Khoa đào tạo đạt Precision 74.1% (mAP 65.9%), Chuyên ngành đạt 65.5% (mAP 63.7%), Ngày sinh đạt mAP 63.8%, Họ tên sinh viên đạt 52.0% (mAP 59.4%), MSSV đạt 57.8% (mAP 48.2%).", sz=9, italic=True)

    # Section 4: Academic Rule Engine & Web UI
    heading1(doc, "IV. ACADEMIC RULE ENGINE VÀ ỨNG DỤNG WEB DEMO (STREAMLIT UI)")
    heading2(doc, "1. Academic Rule Engine (Kiểm tra Tính hợp lệ Nghiệp vụ)")
    body(doc, "Xây dựng độc lập module pipeline/rule_engine.py gồm 5 nhóm luật bám sát quy chế đào tạo HCMUTE:", bold=True)
    body(doc, "R1x - Định danh sinh viên: Regex MSSV chuẩn 8 chữ số ^2\\d{7}$, định dạng mã lớp, định dạng email trường @student.hcmute.edu.vn, số điện thoại liên hệ.", bullet=True)
    body(doc, "R2x - Thời gian & Niên khóa: Kiểm tra tính logic của ngày làm đơn (trước ngày thẩm định), kiểm tra niên khóa nhập học theo 2 số đầu MSSV.", bullet=True)
    body(doc, "R3x - Tính đầy đủ hồ sơ: Yêu cầu con dấu tròn đỏ và số lượng chữ ký bắt buộc theo từng loại mẫu T01–T10 (ví dụ T01 cần 5 chữ ký, T08 cần 3 chữ ký và 2 con dấu).", bullet=True)
    body(doc, "R4x - Học phần & Điểm số: Chuẩn hóa mã học phần, đối soát thang điểm 10 ↔ 4 ↔ chữ (khớp với thang quy chế đào tạo HCMUTE).", bullet=True)
    body(doc, "R5x - Chất lượng nhận dạng OCR: Cảnh báo chuyên viên đối chiếu thủ công với các trường trọng yếu có độ tin cậy OCR < 75%.", bullet=True)
    body(doc, "Kết quả kiểm thử: Bộ Unit Test tự động (test_rule_engine.py) đạt 8/8 test cases Passed (100%).", bold=True)

    heading2(doc, "2. Ứng dụng Web Tương tác (Streamlit UI - app.py)")
    body(doc, "Đã triển khai hoàn chỉnh ứng dụng Web hiện đại (app.py) với 4 tabs chức năng:", bold=True)
    body(doc, "Tab 1 (Layout Inspection): Trực quan hóa Bounding Box phân màu theo Ontology chuẩn của trường; xem ảnh trước và sau nhận dạng.", bullet=True)
    body(doc, "Tab 2 (Structured Data): Trích xuất thông tin sinh viên có cấu trúc; trích xuất bảng điểm đa cột gom cụm theo trục y; hỗ trợ xuất JSON và CSV.", bullet=True)
    body(doc, "Tab 3 (Rule Engine Dashboard): Hiển thị thang điểm tuân thủ (Compliance Score 0-100%); gắn cờ HỢP LỆ (xanh), CẢNH BÁO (vàng), VI PHẠM (đỏ); bảng chi tiết từng điều luật vi phạm.", bullet=True)
    body(doc, "Tab 4 (Benchmark Hub): Tích hợp toàn bộ bảng số liệu và biểu đồ đối chứng RQ1 & RQ2 phục vụ thuyết minh khoa học.", bullet=True)

    # Section 5: Next Steps
    heading1(doc, "V. KẾ HOẠCH NGHIỆM THU VÀ BƯỚC TIẾP THEO (GIAI ĐOẠN 4)")
    body(doc, "1. Hoàn thiện giao diện Web: Kiểm thử mở rộng trên toàn bộ các biểu mẫu thực tế của sinh viên.", bullet=True)
    body(doc, "2. Đóng gói ứng dụng: Hỗ trợ triển khai Docker Container hoặc chạy trực tiếp bằng lệnh 'streamlit run app.py'.", bullet=True)
    body(doc, "3. Biên soạn báo cáo tổng kết tiểu luận: Viết chi tiết Chương 3 (Mô hình đề xuất) và Chương 4 (Thực nghiệm & Đánh giá) cho cuốn báo cáo chính thức.", bullet=True)

    # Signature Table
    p_sp = doc.add_paragraph()
    fmt(p_sp, sb=8, sa=3)

    sig = doc.add_table(rows=1, cols=2)
    sig.alignment = WD_TABLE_ALIGNMENT.CENTER
    sig.autofit = False

    cs1 = sig.rows[0].cells[0]
    cs1.width = Inches(3.8)
    ps1 = cs1.paragraphs[0]
    fmt(ps1, sb=0, sa=1, align=WD_ALIGN_PARAGRAPH.CENTER)
    for txt, sz, bold, italic in [
        ("Ý KIẾN CỦA GIẢNG VIÊN HƯỚNG DẪN\n", 10, True, False),
        ("(Ký và ghi rõ họ tên)\n\n\n\n\n", 9, False, True),
        ("TS. Bùi Mạnh Quân", 10, True, False)
    ]:
        r = ps1.add_run(txt)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(sz)
        r.font.bold = bold
        r.font.italic = italic

    cs2 = sig.rows[0].cells[1]
    cs2.width = Inches(3.8)
    ps2 = cs2.paragraphs[0]
    fmt(ps2, sb=0, sa=1, align=WD_ALIGN_PARAGRAPH.CENTER)
    for txt, sz, bold, italic in [
        ("Tp. Hồ Chí Minh, ngày 02 tháng 11 năm 2026\n", 9, False, True),
        ("ĐẠI DIỆN NHÓM SINH VIÊN\n", 10, True, False),
        ("(Ký và ghi rõ họ tên)\n\n\n\n\n", 9, False, True),
        ("Nguyễn Bách Tùng\nNguyễn Hưng Nguyên – Trần Cẩm Long", 10, True, False)
    ]:
        r = ps2.add_run(txt)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(sz)
        r.font.bold = bold
        r.font.italic = italic

    # Save Word document
    try:
        doc.save(out_docx)
    except PermissionError:
        out_docx = out_docx.replace(".docx", "_v2.docx")
        out_pdf = out_pdf.replace(".pdf", "_v2.pdf")
        doc.save(out_docx)

    print(f"✅ Đã tạo file Word: {out_docx}")

    # Convert to PDF
    try:
        conv = PDFConverter()
        conv.convert(out_docx, out_pdf)
        conv.close()
        print(f"✅ Đã xuất PDF: {out_pdf}")
    except Exception as e:
        print(f"⚠️ Lỗi convert PDF: {e}")


if __name__ == "__main__":
    main()
