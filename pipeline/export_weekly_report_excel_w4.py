# -*- coding: utf-8 -*-
"""
Export Báo cáo tiến độ Tuần 4 - Tiểu luận CNTT to Excel (.xlsx)
Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh
"""

import os
import sys
import openpyxl

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_EXCEL = os.path.join(BASE_DIR, "Bao_Cao_Tien_Do_Tuan_4_TLCN.xlsx")


def create_progress_report_excel(output_path=OUTPUT_EXCEL):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Báo cáo tiến độ Tuần 4"

    # Ensure grid lines are visible
    ws.views.sheetView[0].showGridLines = True

    # Palette
    NAVY_TITLE = "1B365D"
    SECTION_FILL = "2E4A7D"
    HEADER_FILL = "203764"
    WHITE = "FFFFFF"
    BORDER_COLOR = "BFBFBF"
    LIGHT_GRAY_FILL = "F2F2F2"

    font_title = Font(name="Calibri", size=15, bold=True, color=NAVY_TITLE)
    font_section = Font(name="Calibri", size=11, bold=True, color=WHITE)
    font_tbl_header = Font(name="Calibri", size=10, bold=True, color=WHITE)
    font_bold = Font(name="Calibri", size=10, bold=True)
    font_regular = Font(name="Calibri", size=10)
    font_italic = Font(name="Calibri", size=9, italic=True)

    fill_section = PatternFill(start_color=SECTION_FILL, end_color=SECTION_FILL, fill_type="solid")
    fill_tbl_header = PatternFill(start_color=HEADER_FILL, end_color=HEADER_FILL, fill_type="solid")

    thin_border = Border(
        left=Side(style="thin", color=BORDER_COLOR),
        right=Side(style="thin", color=BORDER_COLOR),
        top=Side(style="thin", color=BORDER_COLOR),
        bottom=Side(style="thin", color=BORDER_COLOR)
    )

    align_center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    align_left = Alignment(horizontal="left", vertical="center", wrap_text=True)
    align_top_left = Alignment(horizontal="left", vertical="top", wrap_text=True)
    align_top_center = Alignment(horizontal="center", vertical="top", wrap_text=True)

    # 1. Title Row 1
    ws.merge_cells("A1:H1")
    cell_a1 = ws["A1"]
    cell_a1.value = "BÁO CÁO TIẾN ĐỘ TUẦN 4 - TIỂU LUẬN CNTT"
    cell_a1.font = font_title
    cell_a1.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 32

    # Row 2: Empty
    ws.row_dimensions[2].height = 10

    # 2. Section: THÔNG TIN CHUNG (Row 3)
    ws.merge_cells("A3:H3")
    sec1 = ws["A3"]
    sec1.value = " THÔNG TIN CHUNG"
    sec1.font = font_section
    sec1.fill = fill_section
    sec1.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[3].height = 22

    # Info Rows 4-7
    info_data = [
        (4, "Tên đề tài:", "XÂY DỰNG HỆ THỐNG NHẬN DẠNG, TRÍCH XUẤT VÀ KIỂM TRA TÍNH HỢP LỆ CỦA THÔNG TIN TỪ ĐƠN HỌC VỤ VÀ BẢNG ĐIỂM SINH VIÊN BẰNG HỌC SÂU"),
        (5, "Giảng viên HD:", "TS. Bùi Mạnh Quân"),
        (6, "Sinh viên TH:", "Nguyễn Bách Tùng (23110166), Nguyễn Hưng Nguyên (23110035), Trần Cẩm Long (23110122)"),
        (7, "Thời gian:", "Tuần 4")
    ]
    for r_idx, label, val in info_data:
        ws.cell(row=r_idx, column=1, value=label).font = font_bold
        ws.cell(row=r_idx, column=1).alignment = Alignment(horizontal="left", vertical="center")
        ws.merge_cells(start_row=r_idx, start_column=2, end_row=r_idx, end_column=8)
        val_cell = ws.cell(row=r_idx, column=2, value=val)
        val_cell.font = font_regular
        val_cell.alignment = Alignment(horizontal="left", vertical="center")
        ws.row_dimensions[r_idx].height = 19

    # Row 8: Empty
    ws.row_dimensions[8].height = 10

    # 3. Section: MỤC TIÊU & KẾ HOẠCH TUẦN 4 (Row 9)
    ws.merge_cells("A9:H9")
    sec2 = ws["A9"]
    sec2.value = " MỤC TIÊU & KẾ HOẠCH TUẦN 4"
    sec2.font = font_section
    sec2.fill = fill_section
    sec2.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[9].height = 22

    # Objectives (Rows 10-14)
    objectives = [
        "Xây dựng hoàn chỉnh ứng dụng Web App tương tác người dùng (Streamlit Dashboard) gồm 4 Tab chức năng: Trực quan hóa Layout (YOLOv11), Dữ liệu Cấu trúc & OCR (VietOCR), Thẩm định Quy chế (Academic Rule Engine), và Benchmark Nghiên cứu (RQ1 & RQ2 Hub).",
        "Nghiên cứu và khắc phục triệt để các lỗi nhận diện trên ảnh chụp camera thực tế (Smartphone Domain Shift): hiện tượng nhận diện tràn/ảo trên các dòng bảng trống, cắt xén số điện thoại, trùng lặp thông tin Khoa, nhầm lẫn ngày lập đơn và ngày thi.",
        "Phát triển thuật toán bù góc nghiêng dốc của bảng (Table Tilt Slope Correction) và Dynamic Adaptive Padding cho bounding box hẹp nhằm nâng cao độ chính xác trích xuất điểm số và số điện thoại.",
        "Tích hợp tri thức miền danh mục đào tạo HCMUTE (Knowledge Base Grounding & Normalization) nhằm tự động chuẩn hóa mã môn học, tên môn học và phát hiện chữ ký mực thực tế (Handwritten Ink Blob Detector).",
        "Nghiên cứu và phát triển tính năng Human-in-the-Loop (HITL) CRUD Interface cho phép người dùng kiểm tra, thêm, sửa, xóa trực tiếp dữ liệu OCR sai lệch và tự động kích hoạt tính toán lại quy chế học vụ tức thì (Re-validation)."
    ]
    for idx, text in enumerate(objectives, start=10):
        ws.cell(row=idx, column=1, value=idx - 9).alignment = align_top_center
        ws.cell(row=idx, column=1).font = font_regular
        ws.cell(row=idx, column=1).border = thin_border
        ws.merge_cells(start_row=idx, start_column=2, end_row=idx, end_column=8)
        c = ws.cell(row=idx, column=2, value=text)
        c.font = font_regular
        c.alignment = align_top_left
        for col_i in range(2, 9):
            ws.cell(row=idx, column=col_i).border = thin_border
        ws.row_dimensions[idx].height = 28

    # Row 15: Empty
    ws.row_dimensions[15].height = 10

    # 4. Section: PHÂN CÔNG CHI TIẾT CÔNG VIỆC TUẦN 4 (Row 16)
    ws.merge_cells("A16:H16")
    sec3 = ws["A16"]
    sec3.value = " PHÂN CÔNG CHI TIẾT CÔNG VIỆC TUẦN 4"
    sec3.font = font_section
    sec3.fill = fill_section
    sec3.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[16].height = 22

    # Tasks Header (Row 17)
    headers_t4 = ["STT", "Hạng mục / Module", "Đầu việc chi tiết", "Người phụ trách", "Dự kiến", "Trạng thái", "Kết quả / Deliverables", "Ghi chú"]
    for col_idx, h_name in enumerate(headers_t4, start=1):
        cell = ws.cell(row=17, column=col_idx, value=h_name)
        cell.font = font_tbl_header
        cell.fill = fill_tbl_header
        cell.alignment = align_center
        cell.border = thin_border
    ws.row_dimensions[17].height = 26

    # Tasks Data (Rows 18-22)
    tasks_data = [
        (
            1,
            "Phát triển Web App Streamlit & Dashboard UI",
            "Thiết kế giao diện Web tương tác Streamlit app.py chuẩn UI/UX hiện đại; cấu trúc 4 Tabs chuyên biệt (Trực quan layout YOLOv11, Dữ liệu cấu trúc OCR, Thẩm định quy chế học vụ, Benchmark RQ1 & RQ2 Hub); bổ sung sidebar tùy chỉnh ngưỡng conf/IoU, chọn thiết bị CPU/GPU và bộ chọn ảnh kiểm thử thực tế.",
            "Nguyễn Bách Tùng",
            "Tuần 4",
            "Đã hoàn thành",
            "Ứng dụng web Streamlit hoàn chỉnh (app.py) hoạt động ổn định; thời gian tải trực quan mượt mà, phân loại trực quan màu sắc nhãn thực thể, hiển thị thẻ thẩm định hợp lệ/cảnh báo/vi phạm trực quan.",
            "Hiện thực hóa trọn vẹn sản phẩm demo người dùng cuối theo đúng cam kết kế hoạch tuần 3."
        ),
        (
            2,
            "Khắc phục Lỗi Bounding Box & Bảng trống trên Ảnh Thực tế",
            "Phân tích nguyên nhân lỗi sinh bounding box ảo trên các dòng trống 4-10 ở ảnh chụp camera thực tế; cải tiến thuật toán gom cụm dòng (Y-clustering) với Table Grid Alignment và Early-Stop Logic; sửa lỗi trùng lặp Khoa đào tạo; tách biệt ngày lập đơn ở footer và ngày thi trong bảng điểm.",
            "Trần Cẩm Long",
            "Tuần 4",
            "Đã hoàn thành",
            "Loại bỏ 100% hiện tượng hallucination (box rác) trên các dòng bảng chưa điền; trích xuất chính xác ngày văn bản ('01/07/2026') thay vì lấy nhầm ngày thi; sửa dứt điểm lỗi lặp chuỗi 'Khoa Điện - Điện tử'.",
            "Giải quyết triệt để vấn đề domain shift trên các form mẫu có cấu trúc dòng kẻ bảng phức tạp."
        ),
        (
            3,
            "Thuật toán Bù Dốc Bảng (Tilt Slope) & Dynamic Padding",
            "Nghiên cứu góc nghiêng chụp thực tế dẫn đến độ lệch dòng Y giữa các cột; xây dựng công thức bù dốc tuyến tính Delta_y = slope * (x - x_ref) khi crop các ô điểm số; phát triển Dynamic Adaptive Padding mở rộng lề trái (+25px) cho ô số điện thoại và đồng bộ tọa độ visual bounding box.",
            "Nguyễn Hưng Nguyên,\nNguyễn Bách Tùng",
            "Tuần 4",
            "Đã hoàn thành",
            "Đọc chính xác 100% các ô điểm thi (6.4, 8.6, 6.0) không còn bị cắt viền trên/dưới; nhận diện trọn vẹn 10 chữ số điện thoại '0837958902' thay vì bị cụt mất tiền tố '08'; tọa độ hộp bounding box hiển thị khớp hoàn hảo với vùng crop.",
            "Nâng cao độ bền vững (robustness) của pipeline trước biến dạng phối cảnh ảnh chụp smartphone."
        ),
        (
            4,
            "Tích hợp Knowledge Base Grounding & Nhận diện Chữ ký",
            "Tích hợp từ điển danh mục đào tạo chính quy HCMUTE (Course Knowledge Base) để đối soát fuzzy matching và chuẩn hóa mã môn/tên môn; phát triển bộ lọc thị giác vết mực chữ ký thủ công (Handwritten Ink Blob Detector) ở vùng xác nhận cuối đơn.",
            "Nguyễn Hưng Nguyên",
            "Tuần 4",
            "Đã hoàn thành",
            "Chuẩn hóa tự động các môn học (MOBI330980, MATH141701, PRTE230385) đạt độ chính xác 100%; phát hiện chính xác chữ ký viết tay và ghi nhận trạng thái pháp lý hợp lệ của hồ sơ sinh viên.",
            "Tăng cường độ tin cậy thực tế nhờ kết hợp giữa thị giác máy tính và tri thức chuyên ngành."
        ),
        (
            5,
            "Phát triển Tính năng Human-in-the-Loop (HITL) CRUD",
            "Xây dựng cơ chế tương tác Người - Máy (HITL): chế độ chỉnh sửa thông tin sinh viên, bảng điểm động bằng st.data_editor (thêm dòng, sửa trực tiếp, xóa dòng); nút Lưu thay đổi tự động kích hoạt Rule Engine thẩm định lại quy chế tức thì (Re-validation) và cập nhật Compliance Score.",
            "Trần Cẩm Long,\nNguyễn Bách Tùng",
            "Tuần 4",
            "Đã hoàn thành",
            "Giao diện HITL CRUD trực quan, hoạt động mượt mà; hỗ trợ chuyên viên phòng đào tạo can thiệp xử lý ngoại lệ nhanh chóng; điểm tuân thủ quy chế tự động cập nhật ngay sau khi lưu dữ liệu chỉnh sửa.",
            "Đáp ứng hoàn hảo yêu cầu tính ứng dụng thực tế và khả năng vận hành linh hoạt trong môi trường giáo dục."
        )
    ]

    for r_idx, row_values in enumerate(tasks_data, start=18):
        for col_idx, val in enumerate(row_values, start=1):
            cell = ws.cell(row=r_idx, column=col_idx, value=val)
            cell.font = font_regular
            cell.border = thin_border
            if col_idx in [1, 4, 5, 6]:
                cell.alignment = align_top_center
            else:
                cell.alignment = align_top_left
            if col_idx == 6:  # Trạng thái
                cell.font = font_bold
        ws.row_dimensions[r_idx].height = 68

    # Row 23: Empty
    ws.row_dimensions[23].height = 10

    # 5. Section: KHÓ KHĂN, VƯỚNG MẮC & GIẢI PHÁP ĐỀ XUẤT (Row 24)
    ws.merge_cells("A24:H24")
    sec4 = ws["A24"]
    sec4.value = " KHÓ KHĂN, VƯỚNG MẮC & GIẢI PHÁP ĐỀ XUẤT"
    sec4.font = font_section
    sec4.fill = fill_section
    sec4.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[24].height = 22

    # Issue Table Header (Row 25)
    ws.cell(row=25, column=1, value="STT").alignment = align_center
    ws.cell(row=25, column=2, value="Khó khăn gặp phải").alignment = align_center
    ws.cell(row=25, column=3, value="Hướng giải quyết").alignment = align_center
    ws.cell(row=25, column=4, value="Xin ý kiến GVHD").alignment = align_center
    ws.merge_cells("E25:H25")
    ws.cell(row=25, column=5, value="Ghi chú").alignment = align_center

    for col_i in range(1, 9):
        c = ws.cell(row=25, column=col_i)
        c.font = font_tbl_header
        c.fill = fill_tbl_header
        c.border = thin_border
    ws.row_dimensions[25].height = 26

    # Issue Data (Row 26)
    ws.cell(row=26, column=1, value=1).alignment = align_top_center
    ws.cell(row=26, column=1).font = font_regular
    ws.cell(row=26, column=1).border = thin_border

    issue_b = (
        "Ảnh chụp từ điện thoại thực tế có nhiều biến dạng phi tuyến tính: góc chụp nghiêng dốc (table tilt slope) "
        "làm các ô điểm số ở các cột bên phải bị lệch trục Y so với cột đầu; bounding box của nhãn hẹp như số điện thoại ('phone') "
        "thường bị YOLO cắt sát mép làm mất số '08' ở đầu; các dòng bảng trống phía dưới dễ bị thuật toán gom dòng nhận diện nhầm thành các môn học ảo."
    )
    ws.cell(row=26, column=2, value=issue_b).alignment = align_top_left
    ws.cell(row=26, column=2).font = font_regular
    ws.cell(row=26, column=2).border = thin_border

    issue_c = (
        "1. Xây dựng thuật toán bù góc nghiêng dốc của bảng (Table Tilt Slope Correction) tự động điều chỉnh tọa độ Y theo phương ngang cột.\n"
        "2. Áp dụng Dynamic Adaptive Padding (+25px về bên trái) cho trường phone và đồng bộ với tọa độ visual box.\n"
        "3. Thiết lập cơ chế chặn dừng sớm (Early Stop) khi dòng không có mã/tên môn hợp lệ kết hợp đối soát HCMUTE Course Knowledge Base.\n"
        "4. Bổ sung tính năng HITL CRUD cho phép người dùng hiệu chỉnh thủ công các trường hợp biên (edge cases)."
    )
    ws.cell(row=26, column=3, value=issue_c).alignment = align_top_left
    ws.cell(row=26, column=3).font = font_regular
    ws.cell(row=26, column=3).border = thin_border

    issue_d = (
        "Báo cáo với Thầy về sản phẩm Web App hoàn chỉnh tích hợp trọn vẹn pipeline AI + Rule Engine + HITL CRUD; "
        "xin ý kiến Thầy về việc triển khai đóng gói hệ thống (Docker / local deployment) và định hướng kiểm thử mở rộng trên các mẫu đơn phức tạp khác trong Giai đoạn 5."
    )
    ws.cell(row=26, column=4, value=issue_d).alignment = align_top_left
    ws.cell(row=26, column=4).font = font_regular
    ws.cell(row=26, column=4).border = thin_border

    ws.merge_cells("E26:H26")
    issue_e = ws.cell(row=26, column=5, value="Hệ thống đã đạt độ hoàn thiện cao, giải quyết trọn vẹn các thách thức thực tế từ ảnh chụp camera; sẵn sàng nghiệm thu và viết báo cáo tổng kết đề tài.")
    issue_e.alignment = align_top_left
    issue_e.font = font_regular
    for col_i in range(5, 9):
        ws.cell(row=26, column=col_i).border = thin_border
    ws.row_dimensions[26].height = 110

    # Row 27-31: Empty padding
    for r in range(27, 32):
        ws.row_dimensions[r].height = 8

    # 6. Section: KẾ HOẠCH DỰ KIẾN TUẦN 5 (Row 32)
    ws.merge_cells("A32:H32")
    sec5 = ws["A32"]
    sec5.value = " KẾ HOẠCH DỰ KIẾN TUẦN 5"
    sec5.font = font_section
    sec5.fill = fill_section
    sec5.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[32].height = 22

    # Week 5 Header (Row 33)
    ws.cell(row=33, column=1, value="STT").alignment = align_center
    ws.cell(row=33, column=2, value="Hạng mục").alignment = align_center
    ws.cell(row=33, column=3, value="Đầu việc chuẩn bị").alignment = align_center
    ws.cell(row=33, column=4, value="Dự kiến").alignment = align_center
    ws.merge_cells("E33:H33")
    ws.cell(row=33, column=5, value="Ghi chú").alignment = align_center

    for col_i in range(1, 9):
        c = ws.cell(row=33, column=col_i)
        c.font = font_tbl_header
        c.fill = fill_tbl_header
        c.border = thin_border
    ws.row_dimensions[33].height = 26

    # Week 5 Plan Data (Rows 34-36)
    w5_plans = [
        (
            1,
            "Tối ưu hóa Hiệu năng & Đóng gói Hệ thống (Deployment)",
            "Tối ưu hóa tốc độ suy luận của mô hình (Inference Latency) bằng định dạng ONNX/TensorRT hoặc lượng tử hóa (Quantization); đóng gói toàn bộ ứng dụng thành container Docker hoặc gói cài đặt tự động phục vụ demo.",
            "Tuần 5",
            "Đảm bảo hệ thống vận hành ổn định trên máy tính cá nhân không yêu cầu GPU rời mạnh."
        ),
        (
            2,
            "Kiểm thử Mở rộng & Đánh giá Toàn diện (Stress-Testing)",
            "Thu thập thêm các trường hợp biên (edge cases) từ nhiều góc chụp, điều kiện ánh sáng yếu, bóng tay che khuất; đánh giá độ trễ và độ chính xác end-to-end trên toàn bộ 10 loại biểu mẫu học vụ T01-T10.",
            "Tuần 5",
            "Hoàn thiện bảng số liệu thực nghiệm định lượng phục vụ phần kết quả và thảo luận của khóa luận."
        ),
        (
            3,
            "Hoàn thiện Cuốn Báo cáo Tiểu luận & Slide Nghiệm thu",
            "Viết chi tiết các chương báo cáo: Cơ sở lý thuyết, Phương pháp đề xuất (Pipeline YOLOv11 + VietOCR + Rule Engine + HITL), Thực nghiệm và Kết quả; thiết kế bộ slide trình bày nghiệm thu cuối kỳ trước hội đồng.",
            "Tuần 5",
            "Chuẩn bị đầy đủ tài liệu học thuật theo đúng quy chuẩn định dạng của Trường Đại học Công nghệ Kỹ thuật TP.HCM."
        )
    ]

    for r_idx, (stt, item, prep, due, note) in enumerate(w5_plans, start=34):
        ws.cell(row=r_idx, column=1, value=stt).alignment = align_top_center
        ws.cell(row=r_idx, column=1).font = font_regular
        ws.cell(row=r_idx, column=1).border = thin_border

        ws.cell(row=r_idx, column=2, value=item).alignment = align_top_left
        ws.cell(row=r_idx, column=2).font = font_regular
        ws.cell(row=r_idx, column=2).border = thin_border

        ws.cell(row=r_idx, column=3, value=prep).alignment = align_top_left
        ws.cell(row=r_idx, column=3).font = font_regular
        ws.cell(row=r_idx, column=3).border = thin_border

        ws.cell(row=r_idx, column=4, value=due).alignment = align_top_center
        ws.cell(row=r_idx, column=4).font = font_regular
        ws.cell(row=r_idx, column=4).border = thin_border

        ws.merge_cells(start_row=r_idx, start_column=5, end_row=r_idx, end_column=8)
        note_cell = ws.cell(row=r_idx, column=5, value=note)
        note_cell.alignment = align_top_left
        note_cell.font = font_regular
        for col_i in range(5, 9):
            ws.cell(row=r_idx, column=col_i).border = thin_border
        ws.row_dimensions[r_idx].height = 42

    # Column Widths Setup
    col_widths = {
        "A": 7.0,
        "B": 26.0,
        "C": 42.0,
        "D": 19.0,
        "E": 12.0,
        "F": 16.0,
        "G": 38.0,
        "H": 26.0
    }
    for col_letter, width in col_widths.items():
        ws.column_dimensions[col_letter].width = width

    wb.save(output_path)
    print(f"✅ Đã xuất thành công file Excel Báo cáo tiến độ Tuần 4 tại: {output_path}")
    return output_path


if __name__ == "__main__":
    create_progress_report_excel()

