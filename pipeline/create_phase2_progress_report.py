# -*- coding: utf-8 -*-
"""
Script to generate Phase 2 Progress Report for HCMUTE DocAI Thesis.
Giai đoạn 2: Huấn luyện YOLOv11 & YOLOv8 Baseline, đánh giá định lượng, 
             thử nghiệm VietOCR Pipeline.
"""

import os
import sys
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

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=60, bottom=60, left=80, right=80):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_cell_border(cell, top="CCCCCC", bottom="CCCCCC", left=None, right=None):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = ['<w:tcBorders ' + nsdecls('w') + '>']
    if top:
        borders.append(f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{top}"/>')
    else:
        borders.append('<w:top w:val="none"/>')
    if bottom:
        borders.append(f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="{bottom}"/>')
    else:
        borders.append('<w:bottom w:val="none"/>')
    if left:
        borders.append(f'<w:left w:val="single" w:sz="4" w:space="0" w:color="{left}"/>')
    else:
        borders.append('<w:left w:val="none"/>')
    if right:
        borders.append(f'<w:right w:val="single" w:sz="4" w:space="0" w:color="{right}"/>')
    else:
        borders.append('<w:right w:val="none"/>')
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
    run.font.name = 'Times New Roman'; run.font.size = Pt(12)
    run.font.bold = True; run.font.color.rgb = RGBColor(0, 51, 102)
    return p

def heading2(doc, text):
    p = doc.add_paragraph()
    fmt(p, sb=5, sa=2, align=WD_ALIGN_PARAGRAPH.LEFT)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'; run.font.size = Pt(11)
    run.font.bold = True; run.font.color.rgb = RGBColor(16, 44, 87)
    return p

def body(doc, text, bold=False, italic=False, bullet=False, sz=10):
    p = doc.add_paragraph()
    fmt(p, sb=0, sa=2, ls=1.1)
    if bullet:
        p.paragraph_format.left_indent = Inches(0.18)
        rb = p.add_run("• ")
        rb.font.name = 'Times New Roman'; rb.font.bold = True
        rb.font.color.rgb = RGBColor(0, 51, 102)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'; run.font.size = Pt(sz)
    run.font.bold = bold; run.font.italic = italic
    return p

def table_header_cell(cell, text, width_in, bg="003366"):
    cell.width = Inches(width_in)
    set_cell_background(cell, bg)
    set_cell_margins(cell, top=45, bottom=45, left=60, right=60)
    set_cell_border(cell, top=bg, bottom=bg, left=bg, right=bg)
    p = cell.paragraphs[0]
    fmt(p, sb=0, sa=0, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p.add_run(text)
    r.font.name = 'Times New Roman'; r.font.size = Pt(9)
    r.font.bold = True; r.font.color.rgb = RGBColor(255, 255, 255)

def table_data_cell(cell, text, width_in, align=WD_ALIGN_PARAGRAPH.LEFT, bg="F7F9FC", bold=False, sz=9):
    cell.width = Inches(width_in)
    set_cell_background(cell, bg)
    set_cell_margins(cell, top=28, bottom=28, left=50, right=50)
    set_cell_border(cell, top="E1E4E8", bottom="E1E4E8", left="E1E4E8", right="E1E4E8")
    p = cell.paragraphs[0]
    fmt(p, sb=0, sa=0, align=align)
    r = p.add_run(text)
    r.font.name = 'Times New Roman'; r.font.size = Pt(sz)
    r.font.bold = bold

def main():
    if sys.platform == "win32":
        try: sys.stdout.reconfigure(encoding='utf-8')
        except: pass

    out_docx = os.path.join(BASE_DIR, "Bao_Cao_Tien_Do_Giai_Doan_2_YOLOv11_VietOCR.docx")
    out_pdf  = os.path.join(BASE_DIR, "Bao_Cao_Tien_Do_Giai_Doan_2_YOLOv11_VietOCR.pdf")

    doc = docx.Document()

    for sec in doc.sections:
        sec.top_margin = Inches(0.59)
        sec.bottom_margin = Inches(0.59)
        sec.left_margin = Inches(0.79)
        sec.right_margin = Inches(0.59)
        sec.page_width = Inches(8.27)
        sec.page_height = Inches(11.69)

    # =====================================================================
    # TRANG 1: TIÊU ĐỀ + BẢNG THÔNG TIN + I. TỔNG KẾT KẾT QUẢ GIAI ĐOẠN 2
    # =====================================================================

    # Header
    ht = doc.add_table(rows=1, cols=2)
    ht.alignment = WD_TABLE_ALIGNMENT.CENTER; ht.autofit = False

    cl = ht.rows[0].cells[0]; cl.width = Inches(3.8)
    pl = cl.paragraphs[0]; fmt(pl, sb=0, sa=0, align=WD_ALIGN_PARAGRAPH.CENTER)
    for txt, sz, bold in [("BỘ GIÁO DỤC VÀ ĐÀO TẠO\n", 9.5, False),
                           ("TRƯỜNG ĐH CÔNG NGHỆ KỸ THUẬT TP.HCM\n", 9.5, True),
                           ("KHOA CÔNG NGHỆ THÔNG TIN\n", 9.5, True),
                           ("---o0o---", 8.5, False)]:
        r = pl.add_run(txt); r.font.name = 'Times New Roman'
        r.font.size = Pt(sz); r.font.bold = bold

    cr = ht.rows[0].cells[1]; cr.width = Inches(3.8)
    pr = cr.paragraphs[0]; fmt(pr, sb=0, sa=0, align=WD_ALIGN_PARAGRAPH.CENTER)
    for txt, sz, bold in [("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n", 9.5, True),
                           ("Độc lập – Tự do – Hạnh phúc\n", 9.5, True),
                           ("--------------------", 8.5, False)]:
        r = pr.add_run(txt); r.font.name = 'Times New Roman'
        r.font.size = Pt(sz); r.font.bold = bold

    # Title
    p = doc.add_paragraph()
    fmt(p, sb=7, sa=2, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p.add_run("BÁO CÁO TIẾN ĐỘ THỰC HIỆN TIỂU LUẬN CHUYÊN NGÀNH AI")
    r.font.name = 'Times New Roman'; r.font.size = Pt(13)
    r.font.bold = True; r.font.color.rgb = RGBColor(0, 51, 102)

    p2 = doc.add_paragraph()
    fmt(p2, sb=0, sa=5, align=WD_ALIGN_PARAGRAPH.CENTER)
    r2 = p2.add_run("ĐỢT 2: BÁO CÁO GIAI ĐOẠN HUẤN LUYỆN MÔ HÌNH YOLOV11 LAYOUT DETECTION & THỬ NGHIỆM TRÍCH XUẤT CHỮ TIẾNG VIỆT (VIETOCR)")
    r2.font.name = 'Times New Roman'; r2.font.size = Pt(9.5)
    r2.font.bold = True; r2.font.italic = True

    # Info Box
    info_t = doc.add_table(rows=5, cols=2)
    info_t.alignment = WD_TABLE_ALIGNMENT.CENTER; info_t.autofit = False
    info_data = [
        ("Tên đề tài:", "XÂY DỰNG HỆ THỐNG NHẬN DẠNG, TRÍCH XUẤT VÀ KIỂM TRA TÍNH HỢP LỆ CỦA THÔNG TIN TỪ ĐƠN HỌC VỤ VÀ BẢNG ĐIỂM SINH VIÊN BẰNG HỌC SÂU"),
        ("Mã môn học:", "GRPR471979 – Tiểu luận chuyên ngành AI"),
        ("Giảng viên hướng dẫn:", "TS. Bùi Mạnh Quân"),
        ("Nhóm sinh viên thực hiện:", "1. Nguyễn Bách Tùng – MSSV: 23110166 (Nhóm trưởng)\n2. Nguyễn Hưng Nguyên – MSSV: 23110035\n3. Trần Cẩm Long – MSSV: 23110122"),
        ("Thời gian báo cáo:", "Học kỳ 1 – Năm học 2026-2027 (Tháng 09/2026 – Giai đoạn 2)")
    ]
    for idx, (lbl, val) in enumerate(info_data):
        row = info_t.rows[idx]
        c1, c2 = row.cells[0], row.cells[1]
        c1.width = Inches(2.0); c2.width = Inches(5.6)
        bg = "F2F5F9" if idx % 2 == 0 else "FFFFFF"
        for c in [c1, c2]:
            set_cell_background(c, bg)
            set_cell_margins(c, top=30, bottom=30, left=60, right=60)
        set_cell_border(c1, top="D0D7DE", bottom="D0D7DE", left="D0D7DE", right=None)
        set_cell_border(c2, top="D0D7DE", bottom="D0D7DE", left=None, right="D0D7DE")
        p1 = c1.paragraphs[0]; fmt(p1, sb=0, sa=0)
        r1 = p1.add_run(lbl); r1.font.name = 'Times New Roman'; r1.font.size = Pt(9.5); r1.font.bold = True
        p2_ = c2.paragraphs[0]; fmt(p2_, sb=0, sa=0)
        r2_ = p2_.add_run(val); r2_.font.name = 'Times New Roman'; r2_.font.size = Pt(9.5)
        if idx == 0: r2_.font.bold = True; r2_.font.color.rgb = RGBColor(0, 51, 102)

    # SECTION I
    heading1(doc, "I. TỔNG KẾT KẾT QUẢ THỰC NGHIỆM GIAI ĐOẠN 2 (MILESTONE 2)")
    body(doc, "Thực hiện đúng kế hoạch được phê duyệt tại Báo cáo Giai đoạn 1, nhóm đã hoàn thành toàn bộ nhiệm vụ Milestone 2 bao gồm: huấn luyện mô hình YOLOv11 Layout Detection (mô hình đề xuất) và YOLOv8 (mô hình đối chứng), đánh giá định lượng trên hai tập dữ liệu, và thử nghiệm trích xuất chữ tiếng Việt bằng VietOCR.", sz=10)

    heading2(doc, "1. Huấn luyện Mô hình YOLOv11n (Mô hình Đề xuất – Chương 2.2.1 Đề cương)")
    body(doc, "Môi trường và cấu hình huấn luyện:", bold=True, sz=10)
    body(doc, "Nền tảng: Kaggle Notebook với phần cứng GPU NVIDIA Tesla T4 x2 (VRAM 16GB/card).", bullet=True)
    body(doc, "Tập dữ liệu: 500 ảnh synthetic đã gán nhãn 18 lớp học vụ (HCMUTE Ontology), chia tỉ lệ 80/10/10 (Train/Val/Test).", bullet=True)
    body(doc, "Trọng số tiền huấn luyện: yolo11n.pt (Ultralytics). Cấu hình: imgsz=1024, batch=16, epochs=50, patience=12, optimizer=AdamW, lr0=0.001, weight_decay=0.0005.", bullet=True)
    body(doc, "Kết quả huấn luyện (epoch 50):", bold=True, sz=10)
    body(doc, "Sau 50 epochs huấn luyện (~36 phút/epoch trên GPU T4), loss hội tụ ổn định: box_loss giảm từ 1.856 → 0.488, cls_loss giảm từ 4.294 → 0.567, dfl_loss giảm từ 1.280 → 0.818.", bullet=True)

    heading2(doc, "2. Huấn luyện Mô hình YOLOv8n (Mô hình Đối chứng – Giải quyết Câu hỏi nghiên cứu 2)")
    body(doc, "Huấn luyện yolov8n.pt với cùng tập dữ liệu và siêu tham số để đảm bảo tính khách quan khoa học khi so sánh (đây là yêu cầu bắt buộc của đề cương mục 2.2.1 và Câu hỏi nghiên cứu số 2).", bullet=True)

    heading2(doc, "3. Bảng So sánh Định lượng (Trả lời Câu hỏi nghiên cứu số 2 – mục 3.2.1 Đề cương)")
    body(doc, "Kết quả đánh giá trên Synthetic Test Set (45 ảnh) và Zero-shot Real-World (120 ảnh chụp camera thực tế):", sz=10)

    # Bảng kết quả so sánh
    t_res = doc.add_table(rows=4, cols=6)
    t_res.alignment = WD_TABLE_ALIGNMENT.CENTER; t_res.autofit = False
    res_hdrs = ["Mô hình", "Tập Đánh Giá", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
    res_widths = [1.6, 2.0, 0.95, 0.95, 0.95, 1.15]
    for ci, (h, w) in enumerate(zip(res_hdrs, res_widths)):
        table_header_cell(t_res.rows[0].cells[ci], h, w)

    res_data = [
        ("YOLOv8n\n(Đối chứng)", "Synthetic Test\n(45 ảnh)", "94.37%", "94.29%", "95.52%", "86.97%"),
        ("YOLOv11n\n(Đề xuất)", "Synthetic Test\n(45 ảnh)", "97.98% ↑", "91.46%", "95.37%", "86.10%"),
        ("YOLOv11n\n(Zero-shot)", "Real-World Camera\n(120 ảnh thực tế)", "23.27%", "12.91%", "6.89%", "1.66%"),
    ]
    for ri, row_data in enumerate(res_data):
        row = t_res.rows[ri + 1]
        bg = "F7F9FC" if ri % 2 == 0 else "FFFFFF"
        for ci, (val, w) in enumerate(zip(row_data, res_widths)):
            al = WD_ALIGN_PARAGRAPH.CENTER
            bold = (ci == 0) or (val.endswith("↑"))
            table_data_cell(row.cells[ci], val, w, align=al, bg=bg, bold=bold, sz=9)

    body(doc, "Nhận xét:", bold=True, sz=10)
    body(doc, "YOLOv11n đề xuất đạt Precision vượt trội 97.98% (+3.61% so với YOLOv8n), phù hợp với giả thuyết nghiên cứu đặt ra trong đề cương. Precision cao có ý nghĩa thực tiễn quan trọng: trong bài toán trích xuất biểu mẫu học vụ, tránh bắt nhầm vùng sai đảm bảo vùng crop đưa vào VietOCR luôn sạch và đúng trường thông tin.", bullet=True)
    body(doc, "Hiện tượng Domain Shift nghiêm trọng (mAP@0.5 giảm từ 95.37% → 6.89%) khi đánh giá zero-shot trực tiếp trên ảnh chụp camera chưa qua tiền xử lý. Đây là vấn đề đã được dự báo và sẽ được giải quyết ở Giai đoạn 3 bằng Fine-tuning thêm trên tập real-world và Data Augmentation nâng cao.", bullet=True)
    body(doc, "Lưu ý quan trọng: Khi ảnh chụp thực tế được đưa qua pipeline Perspective Dewarping (nắn thẳng phối cảnh) đã xây dựng ở Giai đoạn 1, mô hình YOLOv11n phát hiện chính xác 35 thành phần tài liệu, bao gồm đầy đủ 3 con dấu tròn đỏ và 4 chữ ký tay trên cùng 1 biểu mẫu thực tế – đây là kết quả hứa hẹn cho Giai đoạn 3.", bullet=True)

    # =====================================================================
    # TRANG 2: MỤC I.4 + BẢNG PHÂN CÔNG
    # =====================================================================
    doc.add_page_break()

    heading2(doc, "4. Thử nghiệm Trích xuất Chữ Tiếng Việt – VietOCR Pipeline (Câu hỏi nghiên cứu 1 – mục 2.2.2 Đề cương)")
    body(doc, "Mô hình sử dụng:", bold=True, sz=10)
    body(doc, "VietOCR VGG-Transformer (kiến trúc Sequence-to-Sequence với CNN encoder VGG-19 và Transformer decoder), được tải từ kho lưu trữ chính thức của VietOCR. Đây là baseline trước khi thực hiện Fine-tuning ở Giai đoạn 3.", bullet=True)
    body(doc, "Kết quả thử nghiệm end-to-end (YOLOv11 Layout Detection → Crop vùng BB → VietOCR):", bold=True, sz=10)
    body(doc, "Tổng số trường văn bản đã trích xuất: 501 trường từ 20 ảnh tài liệu synthetic thử nghiệm.", bullet=True)
    body(doc, "Các trường thông tin cá nhân (student_name, dob, pob, phone, major): Confidence trung bình 90–98%. Ví dụ: \"Hoàng Hải Hữu\" (conf=90.2%), \"18/01/2002\" (conf=98.5%), \"0984911984\" (conf=93.8%), \"Công nghệ Thông tin\" (conf=91.7%).", bullet=True)
    body(doc, "Các trường ngày tháng (date): Confidence trung bình 91–97%. Nhận dạng chính xác định dạng chuẩn VN như \"ngày 04 tháng 07 năm 2026\" và \"19/06/2026\".", bullet=True)
    body(doc, "Trường lý do (reason): Confidence ~65–70%, do văn bản dài và phức tạp ngữ nghĩa. Đây là trường cần Fine-tuning VietOCR chuyên biệt ở Giai đoạn 3.", bullet=True)
    body(doc, "Tích hợp Rule Engine cơ bản (Mục 2.3.2 Đề cương):", bold=True, sz=10)
    body(doc, "Kiểm tra sự hiện diện của chữ ký và con dấu (Verification Module): Đã hoạt động chính xác – phát hiện đúng 4 chữ ký và 3 mộc đỏ trên ảnh biểu mẫu thực tế (has_signature=True, stamp_count=3, signature_count=4).", bullet=True)
    body(doc, "Trích xuất MSSV và Lớp đúng định dạng HCMUTE: student_id=23141232 (8 chữ số), class_name=23141CL1.", bullet=True)

    heading2(doc, "5. Thử nghiệm Trên Ảnh Chụp Thực Tế (Kịch bản 3.1.2 Đề cương)")
    body(doc, "Đây là kịch bản đánh giá độ bền vững với nhiễu thực tế theo yêu cầu đề cương (mục 3.1.2: xoay góc, mờ, thiếu sáng). Nhóm đã thực hiện inference trực tiếp trên ảnh REAL_001_IMG_2531.jpg (120 ảnh chụp smartphone thực tế):", sz=10)
    body(doc, "Với ảnh đã qua Perspective Dewarping: Phát hiện được 35/35 thành phần tài liệu mong đợi. VietOCR nhận dạng được đầy đủ các trường: faculty=\"Khoa Điện - Điện tử\", phone=\"0838346854\", student_id=\"23141232\", class_name=\"23141CL1\", reason=\"Chuyển ngành học để phù hợp với định hướng nghề nghiệp\".", bullet=True)
    body(doc, "Không có Perspective Dewarping (zero-shot thuần): mAP@0.5=6.89%, cho thấy bước tiền xử lý Dewarping là bắt buộc trong pipeline và cần được tích hợp vào Web Demo.", bullet=True)

    # SECTION II: Bảng Phân Công
    heading1(doc, "II. BẢNG PHÂN CÔNG NHIỆM VỤ GIAI ĐOẠN 2")
    body(doc, "Phân công nhiệm vụ trong Giai đoạn 2 (Huấn luyện mô hình & Đánh giá thực nghiệm):", sz=10)

    t_pc = doc.add_table(rows=4, cols=4)
    t_pc.alignment = WD_TABLE_ALIGNMENT.CENTER; t_pc.autofit = False
    pc_hdrs = ["STT", "Thành Viên Thực Hiện", "Nhiệm Vụ Giai Đoạn 2", "Kết Quả / Tiến Độ"]
    pc_widths = [0.4, 1.8, 3.4, 2.0]
    for ci, (h, w) in enumerate(zip(pc_hdrs, pc_widths)):
        table_header_cell(t_pc.rows[0].cells[ci], h, w)

    pc_data = [
        ("1", "Nguyễn Bách Tùng\n(MSSV: 23110166)\n[Nhóm trưởng]",
         "• Thiết kế và hoàn thiện Notebook huấn luyện chuẩn hóa trên Kaggle (train_yolov11_vietocr_kaggle.ipynb).\n• Tiến hành huấn luyện YOLOv11n và YOLOv8n baseline trên GPU T4 x2 (50 epochs, imgsz=1024).\n• Đánh giá định lượng trên Synthetic Test Set và phân tích kết quả Domain Shift.\n• Tích hợp VietOCR vào pipeline end-to-end và trích xuất 501 trường văn bản thử nghiệm.",
         "Đã hoàn thành 100%\n(best.pt YOLOv11 + bảng số liệu đầy đủ)"),
        ("2", "Nguyễn Hưng Nguyên\n(MSSV: 23110035)",
         "• Thực hiện đánh giá Zero-shot trên 120 ảnh chụp camera thực tế.\n• Phân tích hiện tượng Domain Shift giữa ảnh synthetic và ảnh chụp camera.\n• Nghiên cứu kỹ thuật Data Augmentation (Photometric Distortions, Random Perspective, Motion Blur) để chuẩn bị cho Giai đoạn 3.",
         "Đã hoàn thành 100%\n(Báo cáo phân tích Domain Shift + Phương án Data Aug)"),
        ("3", "Trần Cẩm Long\n(MSSV: 23110122)",
         "• Nghiên cứu kiến trúc Fine-tuning VietOCR (VGG-Transformer) theo mục 2.2.2 đề cương.\n• Xây dựng prototype Rule Engine: Regex MSSV HCMUTE 8 chữ số, kiểm tra định dạng ngày tháng, logic niên khóa K20–K26.\n• Khảo sát và thiết kế giao diện Web Demo Streamlit cho phép upload ảnh, hiển thị BB và xuất JSON.",
         "Đã hoàn thành 100%\n(Prototype Rule Engine + Thiết kế UI Streamlit)"),
    ]
    for ri, (stt, sv, task, result) in enumerate(pc_data):
        row = t_pc.rows[ri + 1]
        bg = "F7F9FC" if ri % 2 == 0 else "FFFFFF"
        for ci, (val, w, al) in enumerate(zip([stt, sv, task, result], pc_widths,
                                               [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT,
                                                WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER])):
            table_data_cell(row.cells[ci], val, w, align=al, bg=bg, bold=(ci == 1), sz=8.5)

    # =====================================================================
    # TRANG 3: SECTION III (KẾ HOẠCH GIAI ĐOẠN 3) + SIGNATURE
    # =====================================================================
    doc.add_page_break()

    heading1(doc, "III. KẾ HOẠCH GIAI ĐOẠN 3 (MILESTONE 3) – 2 TUẦN TỚI")
    body(doc, "Mục tiêu tổng thể Giai đoạn 3: Hoàn thiện toàn bộ các mô-đun cốt lõi của hệ thống theo đề cương, bao gồm Fine-tuning VietOCR, xây dựng đầy đủ Rule Engine kiểm tra tính hợp lệ học vụ và hoàn thiện ứng dụng Web Demo.", bold=True, sz=10)

    heading2(doc, "1. Fine-tuning Mô hình VietOCR Chuyên biệt (Câu hỏi nghiên cứu 1 – mục 2.2.2 Đề cương)")
    body(doc, "Thu thập và chuẩn bị tập dữ liệu Fine-tuning OCR chuyên biệt: Cắt lẻ và gán nhãn 3.000–5.000 vùng văn bản/dòng chữ (crop OCR) từ bộ ảnh biểu mẫu học vụ thực tế theo yêu cầu tại mục 2.1.1 đề cương.", bullet=True)
    body(doc, "Tiến hành Fine-tuning VietOCR (VGG-Transformer) trên tập crop OCR chuyên biệt của biểu mẫu HCMUTE. Đo lường chỉ số CER (Character Error Rate), WER (Word Error Rate) và Exact Match trước và sau fine-tuning để trả lời định lượng Câu hỏi nghiên cứu 1.", bullet=True)
    body(doc, "Xây dựng mô-đun hậu xử lý dựa trên Regex và từ điển nghiệp vụ (Business Dictionary): Chuẩn hóa định dạng MSSV (8 chữ số, bắt đầu bằng 2...), ngày tháng, điểm số để giảm thiểu lỗi OCR theo yêu cầu Câu hỏi nghiên cứu 3.", bullet=True)

    heading2(doc, "2. Xây dựng đầy đủ Rule Engine Kiểm tra Tính hợp lệ (mục 2.3.2 Đề cương)")
    body(doc, "Hoàn thiện các quy tắc nghiệp vụ bắt buộc theo đề cương:", bold=True, sz=10)
    body(doc, "Kiểm tra định dạng MSSV: Regex ^2[0-9]{7}$ (8 chữ số bắt đầu bằng 2) – xác thực MSSV hợp lệ HCMUTE.", bullet=True)
    body(doc, "Kiểm tra tính nhất quán Niên khóa – Học kỳ: Ví dụ sinh viên K23 không thể đăng ký học kỳ HK01 2026-2027 tại lớp năm 1 (K26).", bullet=True)
    body(doc, "Kiểm tra sự hiện diện bắt buộc của Con dấu tròn đỏ & Chữ ký tay: Từng loại biểu mẫu (T01–T10) có yêu cầu số lượng chữ ký và mộc dấu khác nhau (ví dụ T01_DIEM_I cần đủ 5 chữ ký: SV, GV1, GV2, Khoa, PĐT).", bullet=True)
    body(doc, "Kiểm tra tính nhất quán điểm số: Đối soát điểm chữ (A, B+, C...) với điểm số thập phân (4.0, 3.5, 2.0...) theo thang điểm 4.0 HCMUTE.", bullet=True)

    heading2(doc, "3. Giải quyết hiện tượng Domain Shift (mục 3.1.2 Đề cương)")
    body(doc, "Áp dụng Data Augmentation nâng cao trong quá trình Fine-tuning thêm YOLOv11n trên tập 120 ảnh thực tế đã gán nhãn: Photometric Distortions (thay đổi độ sáng, tương phản), Random Perspective (mô phỏng góc chụp), Motion Blur (mô phỏng rung tay), Gaussian Noise.", bullet=True)
    body(doc, "Mục tiêu đo lường định lượng: Nâng mAP@0.5 trên real-world dataset từ 6.89% (zero-shot) lên >50% (sau fine-tuning thêm với ảnh thực tế).", bullet=True)

    heading2(doc, "4. Hoàn thiện Mô-đun Ghép trường Key-Value và Trích xuất Bảng Điểm (mục 2.3.1 Đề cương)")
    body(doc, "Xây dựng thuật toán Spatial Key-Value Association: Ghép nối các vùng bounding box theo vị trí không gian (gần nhau theo chiều ngang) để xác định cặp Nhãn–Giá trị tương ứng trên biểu mẫu.", bullet=True)
    body(doc, "Xây dựng mô-đun trích xuất cấu trúc bảng điểm đa học kỳ: Phân tích bảng điểm T10_TRANSCRIPT để trích xuất danh sách Môn học – Mã môn – Số tín chỉ – Điểm theo cấu trúc JSON chuẩn.", bullet=True)

    heading2(doc, "5. Xây dựng và hoàn thiện Ứng dụng Web Demo (mục 2.4.2 Đề cương)")
    body(doc, "Phát triển giao diện Streamlit / FastAPI cho phép người dùng:", bold=True, sz=10)
    body(doc, "Tải lên ảnh tài liệu học vụ (biểu mẫu A4, ảnh chụp smartphone).", bullet=True)
    body(doc, "Hiển thị trực quan các Bounding Box được phát hiện (màu sắc phân biệt theo lớp nhãn).", bullet=True)
    body(doc, "Xem và chỉnh sửa kết quả trích xuất dữ liệu (họ tên, MSSV, điểm số, ngày tháng...).", bullet=True)
    body(doc, "Xuất kết quả ở các định dạng: JSON, Excel, Word, PDF theo yêu cầu mục 2.4.1 đề cương.", bullet=True)

    # Signature
    p_sp = doc.add_paragraph(); fmt(p_sp, sb=8, sa=3)

    sig = doc.add_table(rows=1, cols=2)
    sig.alignment = WD_TABLE_ALIGNMENT.CENTER; sig.autofit = False

    cs1 = sig.rows[0].cells[0]; cs1.width = Inches(3.8)
    ps1 = cs1.paragraphs[0]; fmt(ps1, sb=0, sa=1, align=WD_ALIGN_PARAGRAPH.CENTER)
    for txt, sz, bold, italic in [
        ("Ý KIẾN CỦA GIẢNG VIÊN HƯỚNG DẪN\n", 10.5, True, False),
        ("(Ký và ghi rõ họ tên)\n\n\n\n\n", 9.5, False, True),
        ("TS. Bùi Mạnh Quân", 10.5, True, False)
    ]:
        r = ps1.add_run(txt); r.font.name='Times New Roman'
        r.font.size=Pt(sz); r.font.bold=bold; r.font.italic=italic

    cs2 = sig.rows[0].cells[1]; cs2.width = Inches(3.8)
    ps2 = cs2.paragraphs[0]; fmt(ps2, sb=0, sa=1, align=WD_ALIGN_PARAGRAPH.CENTER)
    for txt, sz, bold, italic in [
        ("Tp. Hồ Chí Minh, ngày 18 tháng 09 năm 2026\n", 9.5, False, True),
        ("ĐẠI DIỆN NHÓM SINH VIÊN\n", 10.5, True, False),
        ("(Ký và ghi rõ họ tên)\n\n\n\n\n", 9.5, False, True),
        ("Nguyễn Bách Tùng\nNguyễn Hưng Nguyên – Trần Cẩm Long", 10.5, True, False)
    ]:
        r = ps2.add_run(txt); r.font.name='Times New Roman'
        r.font.size=Pt(sz); r.font.bold=bold; r.font.italic=italic

    # Save
    try:
        doc.save(out_docx)
    except PermissionError:
        out_docx = out_docx.replace(".docx", "_v2.docx")
        out_pdf  = out_pdf.replace(".pdf", "_v2.pdf")
        doc.save(out_docx)

    print(f"✅ Đã tạo file Word: {out_docx}")

    try:
        conv = PDFConverter()
        conv.convert(out_docx, out_pdf)
        conv.close()
        print(f"✅ Đã xuất PDF: {out_pdf}")
    except Exception as e:
        print(f"⚠️ Lỗi convert PDF: {e}")

if __name__ == "__main__":
    main()
