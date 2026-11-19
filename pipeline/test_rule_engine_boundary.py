# -*- coding: utf-8 -*-
"""
Bộ Kịch bản Kiểm thử Độc lập Rule Engine (Unit & Boundary Test Suite)
Đề tài Tiểu luận Chuyên ngành AI - HCMUTE
GVHD: TS. Bùi Mạnh Quân

Kiểm thử riêng biệt tầng suy luận logic Rule Engine (AcademicRuleEngine) bằng dữ liệu
chuẩn hóa (Ground Truth Test Suite) độc lập hoàn toàn với tầng thị giác máy tính.
Bao gồm:
  - Các ca biên điểm số (ngay dưới 3.9, đúng bằng 4.0, ngay trên 4.1; 5.4 - 5.5 - 5.6; 6.9 - 7.0; 8.4 - 8.5)
  - Các ca khuyết thiếu dữ liệu bắt buộc (thiếu MSSV, họ tên, chữ ký, mộc đỏ, bảng điểm trống)
  - Các ca mâu thuẫn quy chế (quy đổi điểm 10 -> 4/chữ sai, mã môn - tên môn không khớp, năm học sai)
  - Các ca niên đại tên trường lịch sử (có ngày hợp lệ, thiếu ngày, mốc chuyển tiếp)
  - ĐẶC BIỆT: Đo lường chính xác Tỷ lệ Chấp thuận Sai (False Acceptance Rate: Hồ sơ không đủ điều kiện nhưng bị kết luận ĐỦ điều kiện).
"""

import os
import sys
import csv
from datetime import date

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline.rule_engine import AcademicRuleEngine, STATUS_VALID, STATUS_WARNING, STATUS_INVALID


def build_boundary_test_suite():
    """Xây dựng 60 ca kiểm thử biên và mâu thuẫn dữ liệu chi tiết."""
    tests = []

    # =========================================================================
    # NHÓM 1: CA BIÊN ĐIỂM SỐ HỌC PHẦN (SCORE BOUNDARY & CONVERSION)
    # =========================================================================
    # TC01: 3.9 ngay dưới ngưỡng qua môn (4.0/D)
    tests.append({
        "id": "TC01", "group": "Điểm số biên",
        "name": "Điểm 3.9 ngay dưới ngưỡng qua môn D (4.0)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "score": "3.9 0.0 F", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_VALID, # Điểm F vẫn là điểm hợp lệ về quy chế (học phần rớt môn)
        "should_be_valid": True,
        "note": "Điểm 3.9 quy đổi 0.0 F đúng quy chế thang 10-4-chữ."
    })

    # TC02: 4.0 đúng bằng ngưỡng qua môn D
    tests.append({
        "id": "TC02", "group": "Điểm số biên",
        "name": "Điểm 4.0 đúng bằng ngưỡng qua môn D",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "score": "4.0 1.0 D", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_VALID,
        "should_be_valid": True,
        "note": "4.0 -> 1.0 D chuẩn ngưỡng tối thiểu qua môn."
    })

    # TC03: 4.1 ngay trên ngưỡng qua môn D
    tests.append({
        "id": "TC03", "group": "Điểm số biên",
        "name": "Điểm 4.1 ngay trên ngưỡng qua môn D",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "score": "4.1 1.0 D", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_VALID,
        "should_be_valid": True,
        "note": "4.1 -> 1.0 D hợp lệ."
    })

    # TC04: 5.4 ngay dưới ngưỡng C (5.5) -> thuộc D+ (1.5)
    tests.append({
        "id": "TC04", "group": "Điểm số biên",
        "name": "Điểm 5.4 ngay dưới ngưỡng C (5.5), thuộc D+",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "MATH141701", "course_name": "Toán cao cấp 1", "score": "5.4 1.5 D+", "credits": "4"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_VALID,
        "should_be_valid": True,
        "note": "5.4 -> 1.5 D+ đúng biên quy chế."
    })

    # TC05: 5.5 đúng bằng ngưỡng C
    tests.append({
        "id": "TC05", "group": "Điểm số biên",
        "name": "Điểm 5.5 đúng bằng ngưỡng C (2.0)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "MATH141701", "course_name": "Toán cao cấp 1", "score": "5.5 2.0 C", "credits": "4"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_VALID,
        "should_be_valid": True,
        "note": "5.5 -> 2.0 C chuẩn."
    })

    # TC06: 6.9 ngay dưới ngưỡng B (7.0) -> thuộc C+ (2.5)
    tests.append({
        "id": "TC06", "group": "Điểm số biên",
        "name": "Điểm 6.9 ngay dưới ngưỡng B (7.0), thuộc C+",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "score": "6.9 2.5 C+", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_VALID,
        "should_be_valid": True,
        "note": "6.9 -> 2.5 C+ chuẩn."
    })

    # TC07: 7.0 đúng bằng ngưỡng B (3.0)
    tests.append({
        "id": "TC07", "group": "Điểm số biên",
        "name": "Điểm 7.0 đúng bằng ngưỡng B (3.0)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "score": "7.0 3.0 B", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_VALID,
        "should_be_valid": True,
        "note": "7.0 -> 3.0 B chuẩn."
    })

    # TC08: 8.4 ngay dưới ngưỡng A (8.5) -> thuộc B+ (3.5)
    tests.append({
        "id": "TC08", "group": "Điểm số biên",
        "name": "Điểm 8.4 ngay dưới ngưỡng A (8.5), thuộc B+",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "score": "8.4 3.5 B+", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_VALID,
        "should_be_valid": True,
        "note": "8.4 -> 3.5 B+ chuẩn."
    })

    # TC09: 8.5 đúng bằng ngưỡng A (3.7)
    tests.append({
        "id": "TC09", "group": "Điểm số biên",
        "name": "Điểm 8.5 đúng bằng ngưỡng A (3.7)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "score": "8.5 3.7 A", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_VALID,
        "should_be_valid": True,
        "note": "8.5 -> 3.7 A chuẩn."
    })

    # TC10: 9.0 đúng bằng ngưỡng A+ (4.0)
    tests.append({
        "id": "TC10", "group": "Điểm số biên",
        "name": "Điểm 9.0 đúng bằng ngưỡng A+ (4.0)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "score": "9.0 4.0 A+", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_VALID,
        "should_be_valid": True,
        "note": "9.0 -> 4.0 A+ chuẩn."
    })

    # TC11: 10.0 biên tối đa của thang điểm 10
    tests.append({
        "id": "TC11", "group": "Điểm số biên",
        "name": "Điểm 10.0 biên tối đa của thang 10",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "score": "10.0 4.0 A+", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_VALID,
        "should_be_valid": True,
        "note": "10.0 -> 4.0 A+ chuẩn."
    })

    # TC12: 0.0 biên tối thiểu của thang điểm 10
    tests.append({
        "id": "TC12", "group": "Điểm số biên",
        "name": "Điểm 0.0 biên tối thiểu của thang 10",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "score": "0.0 0.0 F", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_VALID,
        "should_be_valid": True,
        "note": "0.0 -> 0.0 F chuẩn."
    })

    # TC13: Điểm âm (-0.5) vượt ngoài khoảng cho phép
    tests.append({
        "id": "TC13", "group": "Điểm số biên",
        "name": "Điểm âm (-0.5) ngoài thang [0, 10]",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "score": "-0.5", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R46 (Điểm < 0)."
    })

    # TC14: Điểm lớn hơn 10 (10.5)
    tests.append({
        "id": "TC14", "group": "Điểm số biên",
        "name": "Điểm 10.5 vượt thang điểm 10",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "score": "10.5", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R46 (Điểm > 10)."
    })

    # TC15: Lỗi OCR nhầm dấu phẩy thành 60 (6.0 -> 60)
    tests.append({
        "id": "TC15", "group": "Điểm số biên",
        "name": "Điểm 60.0 do OCR nhầm dấu thập phân",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "score": "60.0", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R46 (Điểm = 60 > 10)."
    })

    # TC16: Mâu thuẫn quy đổi điểm 10 -> điểm 4 (8.5 nhưng ghi điểm 4 là 3.0)
    tests.append({
        "id": "TC16", "group": "Mâu thuẫn quy đổi điểm",
        "name": "Mâu thuẫn quy đổi 10 -> 4 (8.5 nhưng ghi 3.0 thay vì 3.7)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "score": "8.5 3.0 A", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R49 (Quy đổi điểm 10 sang 4 sai lệch)."
    })

    # TC17: Mâu thuẫn quy đổi điểm 10 -> điểm chữ (8.5 nhưng ghi điểm chữ B)
    tests.append({
        "id": "TC17", "group": "Mâu thuẫn quy đổi điểm",
        "name": "Mâu thuẫn quy đổi 10 -> chữ (8.5 nhưng ghi B thay vì A)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "score": "8.5 3.7 B", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R4A (Quy đổi điểm 10 sang chữ sai lệch)."
    })

    # TC18: Điểm chữ không hợp lệ (ghi điểm Z)
    tests.append({
        "id": "TC18", "group": "Mâu thuẫn quy đổi điểm",
        "name": "Điểm chữ không hợp lệ (điểm chữ 'Z')",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "score": "8.5 3.7 Z", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R48 (Điểm chữ ngoài danh mục)."
    })

    # TC19: Môn chỉ xét Đạt (điểm chữ P)
    tests.append({
        "id": "TC19", "group": "Điểm số biên",
        "name": "Môn GDTC / GDQP chỉ xét Đạt (điểm chữ P)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "score": "P", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_VALID,
        "should_be_valid": True,
        "note": "Điểm P hợp lệ cho môn điều kiện."
    })

    # =========================================================================
    # NHÓM 2: CA KHUYẾT THIẾU DỮ LIỆU BẮT BUỘC (MISSING MANDATORY FIELDS)
    # =========================================================================
    # TC20: Thiếu hoàn toàn MSSV
    tests.append({
        "id": "TC20", "group": "Khuyết thiếu dữ liệu",
        "name": "Thiếu hoàn toàn MSSV trong hồ sơ",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R30 (Thiếu trường bắt buộc MSSV)."
    })

    # TC21: Thiếu hoàn toàn họ tên sinh viên
    tests.append({
        "id": "TC21", "group": "Khuyết thiếu dữ liệu",
        "name": "Thiếu hoàn toàn họ và tên sinh viên",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R30 (Thiếu trường bắt buộc Họ tên)."
    })

    # TC22: Thiếu chữ ký người làm đơn (signature_count = 0)
    tests.append({
        "id": "TC22", "group": "Khuyết thiếu dữ liệu",
        "name": "Thiếu chữ ký người làm đơn (signature_count = 0)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 0, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R32 (Thiếu chữ ký bắt buộc)."
    })

    # TC23: Thiếu mộc đỏ xác nhận của Phòng Đào tạo (stamp_count = 0)
    tests.append({
        "id": "TC23", "group": "Khuyết thiếu dữ liệu",
        "name": "Thiếu con dấu mộc đỏ Phòng Đào tạo (stamp_count = 0)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 0}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R33 (Thiếu con dấu bắt buộc)."
    })

    # TC24: Đơn xin chuyển điểm nhưng bảng môn học rỗng
    tests.append({
        "id": "TC24", "group": "Khuyết thiếu dữ liệu",
        "name": "Đơn chuyển điểm nhưng không kê khai môn học nào",
        "input": {
            "document_file": "chuyen_diem.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T04_CHUYEN_DIEM",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R30 (Thiếu course_code / course_name)."
    })

    # TC25: Bảng điểm sinh viên nhưng không có môn học nào
    tests.append({
        "id": "TC25", "group": "Khuyết thiếu dữ liệu",
        "name": "Bảng điểm sinh viên nhưng danh sách môn học trống",
        "input": {
            "document_file": "bang_diem.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166"},
            "courses_table": [],
            "verification": {"signature_count": 0, "stamp_count": 0}
        },
        "form_type": "T10_BANG_DIEM",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R31 (Số học phần tối thiểu >= 1)."
    })

    # TC26: Thiếu cả MSSV, cả chữ ký, cả con dấu
    tests.append({
        "id": "TC26", "group": "Khuyết thiếu dữ liệu",
        "name": "Hồ sơ trống: thiếu MSSV, chữ ký và con dấu",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng"},
            "courses_table": [],
            "verification": {"signature_count": 0, "stamp_count": 0}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm đa luật nghiêm trọng R30, R32, R33."
    })

    # =========================================================================
    # NHÓM 3: CA NGÀY THÁNG, NIÊN ĐẠI & HỌC KỲ
    # =========================================================================
    # TC27: Ngày làm đơn ở tương lai (30/12/2030)
    tests.append({
        "id": "TC27", "group": "Niên đại & Thời gian",
        "name": "Ngày lập đơn ở tương lai (30/12/2030)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "30/12/2030"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R23 (Ngày văn bản sau ngày thẩm định)."
    })

    # TC28: Ngày làm đơn trước năm nhập học (K23 nhập học 2023 nhưng đơn ghi 2020)
    tests.append({
        "id": "TC28", "group": "Niên đại & Thời gian",
        "name": "Ngày lập đơn (2020) trước năm nhập học K23 (2023)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "10/05/2020"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_WARNING,
        "should_be_valid": False,
        "note": "Vi phạm R24 (Ngày đơn trước năm tuyển sinh -> Cảnh báo kiểm tra thủ công)."
    })

    # TC29: Ngày tháng sai lịch học vụ (ngày 31/02/2026 không tồn tại)
    tests.append({
        "id": "TC29", "group": "Niên đại & Thời gian",
        "name": "Ngày không hợp lệ trong lịch (31/02/2026)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "31/02/2026"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_WARNING,
        "should_be_valid": False,
        "note": "Vi phạm R22 (Không phân tích được ngày tháng)."
    })

    # TC30: Năm học cách nhau 2 năm (2023-2025 thay vì liên tiếp)
    tests.append({
        "id": "TC30", "group": "Niên đại & Thời gian",
        "name": "Năm học không liên tiếp (2023-2025)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2024", "reason": "Năm học 2023-2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R25 (Năm học sai quy cách)."
    })

    # TC31: Năm học bị đảo ngược (2025-2024)
    tests.append({
        "id": "TC31", "group": "Niên đại & Thời gian",
        "name": "Năm học bị đảo ngược (2025-2024)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2024", "reason": "Năm học 2025-2024"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R25 (Năm học sai quy cách)."
    })

    # TC32: Học kỳ không hợp lệ (Học kỳ 4)
    tests.append({
        "id": "TC32", "group": "Niên đại & Thời gian",
        "name": "Học kỳ không hợp lệ (Học kỳ 4)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2024", "reason": "Học kỳ: 4"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R27 (Chỉ chấp nhận HK1, HK2, HK3/Hè)."
    })

    # =========================================================================
    # NHÓM 4: CA MÂU THUẪN HỌC PHẦN, TÍN CHỈ & MÔN TRÙNG
    # =========================================================================
    # TC33: Môn học trùng lặp trong cùng một đơn
    tests.append({
        "id": "TC33", "group": "Mâu thuẫn học phần",
        "name": "Môn học trùng lặp mã trong cùng đơn",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [
                {"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "credits": "3"},
                {"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "credits": "3"}
            ],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_WARNING,
        "should_be_valid": False,
        "note": "Vi phạm R42 (Học phần lặp lại mã -> Cảnh báo rà soát học lại/cải thiện)."
    })

    # TC34: Mã học phần sai định dạng (chỉ có 2 chữ cái + 2 số)
    tests.append({
        "id": "TC34", "group": "Mâu thuẫn học phần",
        "name": "Mã học phần sai quy chuẩn định dạng (PR23)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PR23", "course_name": "Kỹ thuật lập trình", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R40 (Sai format mã học phần)."
    })

    # TC35: Mã môn và tên môn không khớp nhau
    tests.append({
        "id": "TC35", "group": "Mâu thuẫn học phần",
        "name": "Mã môn MATH141701 nhưng tên môn là Lập trình Web",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "MATH141701", "course_name": "Lập trình Web chuyên sâu", "credits": "4"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_WARNING,
        "should_be_valid": False,
        "note": "Vi phạm R43 (Tên môn không khớp mã theo CTĐT)."
    })

    # TC36: Số tín chỉ không hợp lệ = 0
    tests.append({
        "id": "TC36", "group": "Mâu thuẫn học phần",
        "name": "Số tín chỉ bằng 0",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "credits": "0"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R44 (Số tín chỉ phải từ 1-15)."
    })

    # TC37: Số tín chỉ số thập phân (2.5)
    tests.append({
        "id": "TC37", "group": "Mâu thuẫn học phần",
        "name": "Số tín chỉ là số lẻ thập phân (2.5)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "credits": "2.5"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R44 (Số tín chỉ phải là số nguyên)."
    })

    # TC38: Số tín chỉ lệch CTĐT (môn PRTE230385 là 3 TC nhưng ghi 1 TC)
    tests.append({
        "id": "TC38", "group": "Mâu thuẫn học phần",
        "name": "Số tín chỉ sai lệch so với CTĐT (ghi 1 TC thay vì 3 TC)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "credits": "1"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_WARNING,
        "should_be_valid": False,
        "note": "Vi phạm R45 (Số tín chỉ không khớp CTĐT)."
    })

    # TC39: Mã môn hoàn toàn không có trong danh mục trường
    tests.append({
        "id": "TC39", "group": "Mâu thuẫn học phần",
        "name": "Mã học phần không tồn tại trong CTĐT trường (XYZ999999)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "XYZ999999", "course_name": "Môn học lạ", "credits": "3"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_WARNING,
        "should_be_valid": False,
        "note": "Vi phạm R41 (Mã học phần ngoài danh mục)."
    })

    # =========================================================================
    # NHÓM 5: CA ĐỊNH DANH SINH VIÊN, LỚP VÀ EMAIL
    # =========================================================================
    # TC40: MSSV sai định dạng (bắt đầu bằng số 1 thay vì số 2)
    tests.append({
        "id": "TC40", "group": "Định danh sinh viên",
        "name": "MSSV bắt đầu bằng số 1 (19110166) sai quy định trường",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "19110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R10 (MSSV phải bắt đầu bằng 2)."
    })

    # TC41: MSSV chỉ có 7 chữ số
    tests.append({
        "id": "TC41", "group": "Định danh sinh viên",
        "name": "MSSV thiếu chữ số (chỉ có 7 số: 2311016)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "2311016", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R10 (MSSV phải gồm 8 số)."
    })

    # TC42: Mâu thuẫn MSSV (2 MSSV khác nhau trong cùng 1 đơn)
    tests.append({
        "id": "TC42", "group": "Định danh sinh viên",
        "name": "Tài liệu chứa 2 MSSV mâu thuẫn (23110166 và 23110035)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": ["23110166", "23110035"], "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R13 (Không nhất quán MSSV)."
    })

    # TC43: Họ tên chỉ có 1 từ
    tests.append({
        "id": "TC43", "group": "Định danh sinh viên",
        "name": "Họ và tên chỉ có 1 từ ('Tùng')",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_WARNING,
        "should_be_valid": False,
        "note": "Vi phạm R15 (Họ tên phải >= 2 từ)."
    })

    # TC44: Họ tên chứa chữ số
    tests.append({
        "id": "TC44", "group": "Định danh sinh viên",
        "name": "Họ và tên chứa số ('Nguyễn Bách Tùng 123')",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng 123", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_WARNING,
        "should_be_valid": False,
        "note": "Vi phạm R15 (Họ tên không được chứa số)."
    })

    # TC45: Mã lớp không khớp khóa tuyển sinh của MSSV
    tests.append({
        "id": "TC45", "group": "Định danh sinh viên",
        "name": "Mã lớp 21110CLC không khớp khóa 23 của MSSV 23110166",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "class_name": "21110CLC", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_WARNING,
        "should_be_valid": False,
        "note": "Vi phạm R16 (Lớp không khớp khóa tuyển sinh)."
    })

    # TC46: Email không khớp MSSV
    tests.append({
        "id": "TC46", "group": "Định danh sinh viên",
        "name": "Email sinh viên (23110035@...) không khớp MSSV (23110166)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "email": "23110035@student.hcmute.edu.vn", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_WARNING,
        "should_be_valid": False,
        "note": "Vi phạm R18 (Email lệch MSSV)."
    })

    # TC47: Số điện thoại không đủ 10 số (chỉ có 9 số)
    tests.append({
        "id": "TC47", "group": "Định danh sinh viên",
        "name": "Số điện thoại bị thiếu số (chỉ 9 số: 091234567)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "phone": "091234567", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_WARNING,
        "should_be_valid": False,
        "note": "Vi phạm R1A (SĐT phải 10 số)."
    })

    # =========================================================================
    # NHÓM 6: CA TÊN TRƯỜNG LỊCH SỬ & NIÊN ĐẠI VĂN BẢN
    # =========================================================================
    # TC48: Tên trường lịch sử + Ngày lập đơn hợp lệ (trước mốc)
    tests.append({
        "id": "TC48", "group": "Tên trường lịch sử",
        "name": "Tên lịch sử (ĐH Sư phạm Kỹ thuật TP.HCM) + Ngày đơn trước mốc",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "school_name": "Trường Đại học Sư phạm Kỹ thuật TP.HCM", "date": "15/05/2024"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_VALID,
        "should_be_valid": True,
        "note": "Hợp lệ R1B (Tên lịch sử được chấp nhận và chuẩn hóa nội bộ)."
    })

    # TC49: Tên trường lịch sử + Thiếu thông tin ngày lập đơn
    tests.append({
        "id": "TC49", "group": "Tên trường lịch sử",
        "name": "Tên lịch sử nhưng khuyết ngày lập đơn (chuyển kiểm tra thủ công)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "school_name": "Trường Đại học Sư phạm Kỹ thuật TP.HCM"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_WARNING,
        "should_be_valid": False,
        "note": "Chuyển kiểm tra thủ công R1B do thiếu mốc đối chiếu."
    })

    # TC50: Tên trường hiện hành chuẩn hóa
    tests.append({
        "id": "TC50", "group": "Tên trường lịch sử",
        "name": "Tên trường chính thức hiện hành (ĐH Công nghệ Kỹ thuật TP.HCM)",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "school_name": "Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh", "date": "15/09/2026"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_VALID,
        "should_be_valid": True,
        "note": "Khớp hoàn toàn tên chuẩn R1B."
    })

    # TC51: Tên trường hoàn toàn lạ ngoài danh mục
    tests.append({
        "id": "TC51", "group": "Tên trường lịch sử",
        "name": "Tên cơ sở đào tạo hoàn toàn không thuộc trường",
        "input": {
            "document_file": "don_bao_luu.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "school_name": "Trường Đại học Kinh tế Quốc dân", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_WARNING,
        "should_be_valid": False,
        "note": "Cảnh báo R1B tên trường lạ."
    })

    # =========================================================================
    # NHÓM 7: CÁC DẠNG BIỂU MẪU ĐẶC THÙ (T01 - T09)
    # =========================================================================
    # TC52: Đơn T01 điểm I yêu cầu >= 5 chữ ký nhưng chỉ có 2
    tests.append({
        "id": "TC52", "group": "Đặc thù loại biểu mẫu",
        "name": "Đơn điểm I (T01) yêu cầu 5 chữ ký nhưng chỉ phát hiện 2",
        "input": {
            "document_file": "don_diem_i.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 2, "stamp_count": 0}
        },
        "form_type": "T01_DIEM_I",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R32 (Thiếu chữ ký thẩm quyền đơn điểm I)."
    })

    # TC53: Đơn T02 học 2 CTĐT thiếu trường chuyên ngành (major)
    tests.append({
        "id": "TC53", "group": "Đặc thù loại biểu mẫu",
        "name": "Đơn học 2 CTĐT (T02) thiếu thông tin ngành đăng ký",
        "input": {
            "document_file": "don_hoc_2ctdt.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T02_HOC_2CTDT",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R30 (Thiếu trường ngành major bắt buộc của T02)."
    })

    # TC54: Đơn T03 chuyển hệ thiếu lý do chuyển hệ (reason)
    tests.append({
        "id": "TC54", "group": "Đặc thù loại biểu mẫu",
        "name": "Đơn chuyển hệ (T03) thiếu lý do chuyển hệ",
        "input": {
            "document_file": "don_chuyen_he.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [],
            "verification": {"signature_count": 1, "stamp_count": 2}
        },
        "form_type": "T03_CHUYEN_HE",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R30 (Thiếu trường lý do reason bắt buộc của T03)."
    })

    # TC55: Đơn T06 bản sao sổ gốc thiếu thông tin văn bằng (degree_info)
    tests.append({
        "id": "TC55", "group": "Đặc thù loại biểu mẫu",
        "name": "Đơn cấp bản sao sổ gốc (T06) thiếu thông tin văn bằng",
        "input": {
            "document_file": "don_ban_sao.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "date": "15/09/2025"},
            "courses_table": [],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T06_BAN_SAO_SO_GOC",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R30 (Thiếu degree_info)."
    })

    # TC56: Đơn T08 thay thế môn học thiếu mộc hoặc chữ ký CVHT/Trưởng khoa
    tests.append({
        "id": "TC56", "group": "Đặc thù loại biểu mẫu",
        "name": "Đơn môn thay thế (T08) chỉ có 1 chữ ký thay vì 3 chữ ký",
        "input": {
            "document_file": "don_thay_the.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 2}
        },
        "form_type": "T08_THAY_THE_MON",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R32 (T08 yêu cầu tối thiểu 3 chữ ký)."
    })

    # TC57: Đơn T09 miễn thi thiếu mộc hoặc chữ ký
    tests.append({
        "id": "TC57", "group": "Đặc thù loại biểu mẫu",
        "name": "Đơn miễn thi (T09) thiếu con dấu mộc đỏ xác nhận",
        "input": {
            "document_file": "don_mien_thi.jpg",
            "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166", "date": "15/09/2025"},
            "courses_table": [{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}],
            "verification": {"signature_count": 1, "stamp_count": 0}
        },
        "form_type": "T09_MIEN_THI",
        "expected_status": STATUS_INVALID,
        "should_be_valid": False,
        "note": "Vi phạm R33 (T09 yêu cầu mộc xác nhận)."
    })

    # =========================================================================
    # NHÓM 8: HỒ SƠ CHUẨN MỰC HỢP LỆ (CLEAN VALID CONTROL CASES)
    # =========================================================================
    # TC58: Đơn T05 bảo lưu chuẩn mực 100%
    tests.append({
        "id": "TC58", "group": "Hồ sơ chuẩn hợp lệ",
        "name": "Đơn bảo lưu (T05) hoàn hảo 100% mọi điều kiện",
        "input": {
            "document_file": "don_bao_luu_chuan.jpg",
            "entities": {
                "student_name": "Nguyễn Bách Tùng",
                "student_id": "23110166",
                "class_name": "23110CLC",
                "phone": "0837958902",
                "email": "23110166@student.hcmute.edu.vn",
                "date": "15/09/2025",
                "school_name": "Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh"
            },
            "courses_table": [
                {"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "credits": "3", "score": "8.5 3.7 A"}
            ],
            "verification": {"signature_count": 1, "stamp_count": 1}
        },
        "form_type": "T05_BAO_LUU",
        "expected_status": STATUS_VALID,
        "should_be_valid": True,
        "note": "Đầy đủ 100% trường, hợp lệ quy chế."
    })

    # TC59: Đơn T01 điểm I chuẩn mực đầy đủ 5 chữ ký
    tests.append({
        "id": "TC59", "group": "Hồ sơ chuẩn hợp lệ",
        "name": "Đơn điểm I (T01) chuẩn mực đầy đủ 5 chữ ký thẩm quyền",
        "input": {
            "document_file": "don_diem_i_chuan.jpg",
            "entities": {
                "student_name": "Nguyễn Hưng Nguyên",
                "student_id": "23110035",
                "class_name": "2311001",
                "phone": "0901234567",
                "date": "15/09/2025"
            },
            "courses_table": [
                {"course_code": "MATH141701", "course_name": "Toán cao cấp 1", "credits": "4"}
            ],
            "verification": {"signature_count": 5, "stamp_count": 0}
        },
        "form_type": "T01_DIEM_I",
        "expected_status": STATUS_VALID,
        "should_be_valid": True,
        "note": "Đủ 5 chữ ký theo quy định T01."
    })

    # TC60: Bảng điểm T10 chuẩn mực đầy đủ các môn và quy đổi điểm chính xác
    tests.append({
        "id": "TC60", "group": "Hồ sơ chuẩn hợp lệ",
        "name": "Bảng điểm (T10) chuẩn mực 3 học phần quy đổi chính xác",
        "input": {
            "document_file": "bang_diem_chuan.jpg",
            "entities": {
                "student_name": "Trần Cẩm Long",
                "student_id": "23110122",
                "class_name": "23110CLC"
            },
            "courses_table": [
                {"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "credits": "3", "score": "8.5 3.7 A"},
                {"course_code": "MATH141701", "course_name": "Toán cao cấp 1", "credits": "4", "score": "7.0 3.0 B"},
                {"course_code": "MOBI330980", "course_name": "Lập trình thiết bị di động", "credits": "3", "score": "9.0 4.0 A+"}
            ],
            "verification": {"signature_count": 0, "stamp_count": 0}
        },
        "form_type": "T10_BANG_DIEM",
        "expected_status": STATUS_VALID,
        "should_be_valid": True,
        "note": "Chuẩn mực 100% bảng điểm cổng đào tạo."
    })

    return tests


def run_rule_engine_boundary_benchmark(output_csv=None):
    """Thực thi kiểm thử 60 ca biên và tổng hợp số liệu khoa học."""
    engine = AcademicRuleEngine(reference_date=date(2026, 10, 26))
    test_suite = build_boundary_test_suite()

    results = []
    total = len(test_suite)
    matched = 0

    # Các chỉ số khoa học then chốt theo yêu cầu của GVHD:
    false_acceptances = []  # Ca KHÔNG đủ điều kiện nhưng Rule Engine kết luận VALID!
    false_rejections = []   # Ca ĐỦ điều kiện nhưng Rule Engine kết luận INVALID!
    manual_review_triggers = 0

    for tc in test_suite:
        res = engine.validate(tc["input"], form_type=tc["form_type"])
        actual_status = res["status"]
        is_auto_acceptable = res.get("is_acceptable_automatically", False)
        needs_manual = res.get("needs_manual_review", False)

        is_match = (actual_status == tc["expected_status"])
        if is_match:
            matched += 1

        if needs_manual:
            manual_review_triggers += 1

        # KIỂM SOÁT FALSE ACCEPTANCE CỰC KỲ NGHIÊM NGẶT:
        # Nếu test case là hồ sơ không hợp lệ / bất thường (should_be_valid=False)
        # NHƯNG hệ thống lại cho ra VALID hoặc cho phép tiếp nhận tự động -> ĐÂY LÀ FALSE ACCEPTANCE!
        if not tc["should_be_valid"] and (actual_status == STATUS_VALID or is_auto_acceptable):
            false_acceptances.append(tc["id"])

        # Nếu test case là hồ sơ chuẩn (should_be_valid=True) nhưng lại bị phán quyết INVALID
        if tc["should_be_valid"] and actual_status == STATUS_INVALID:
            false_rejections.append(tc["id"])

        results.append({
            "id": tc["id"],
            "group": tc["group"],
            "name": tc["name"],
            "form_type": tc["form_type"],
            "expected_status": tc["expected_status"],
            "actual_status": actual_status,
            "compliance_score": res["compliance_score"],
            "needs_manual_review": needs_manual,
            "is_acceptable_automatically": is_auto_acceptable,
            "review_categories": "|".join(res.get("review_categories", [])),
            "is_match": is_match,
            "violations_count": len(res.get("violations", [])),
            "note": tc["note"]
        })

    # Tính toán các tỷ lệ
    accuracy = (matched / total) * 100.0
    total_invalid_cases = sum(1 for tc in test_suite if not tc["should_be_valid"])
    total_valid_cases = sum(1 for tc in test_suite if tc["should_be_valid"])

    far = (len(false_acceptances) / total_invalid_cases) * 100.0 if total_invalid_cases else 0.0
    frr = (len(false_rejections) / total_valid_cases) * 100.0 if total_valid_cases else 0.0

    print("=" * 80)
    print("         BÁO CÁO KẾT QUẢ KIỂM THỬ ĐỘC LẬP RULE ENGINE (UNIT & BOUNDARY)")
    print("=" * 80)
    print(f"Tổng số ca kiểm thử biên (Boundary Cases):  {total}")
    print(f"Số ca khớp chính xác hành vi mong đợi:     {matched} / {total} ({accuracy:.2f}%)")
    print(f"Tổng số ca bất thường / vi phạm quy chế:   {total_invalid_cases}")
    print(f"Tổng số ca chuẩn mực hợp lệ:               {total_valid_cases}")
    print("-" * 80)
    print(f"🎯 TỶ LỆ CHẤP THUẬN SAI (False Acceptance Rate - FAR): {far:.2f}% ({len(false_acceptances)}/{total_invalid_cases})")
    print(f"   -> Ý nghĩa: Tỷ lệ hồ sơ vi phạm nhưng bị kết luận ĐỦ điều kiện.")
    print(f"🎯 TỶ LỆ TỪ CHỐI SAI (False Rejection Rate - FRR):      {frr:.2f}% ({len(false_rejections)}/{total_valid_cases})")
    print(f"   -> Ý nghĩa: Tỷ lệ hồ sơ chuẩn mực nhưng bị bác bỏ oan uổng.")
    print(f"🛡️ Tỷ lệ kích hoạt Kiểm tra thủ công (HITL Escalation):  {manual_review_triggers}/{total} ({manual_review_triggers/total*100:.1f}%)")
    print("=" * 80)

    if false_acceptances:
        print(f"⚠️ CẢNH BÁO: Phát hiện {len(false_acceptances)} ca False Acceptance: {false_acceptances}")
    else:
        print("✅ TUYỆT ĐỐI KHÔNG CÓ FALSE ACCEPTANCE (0.0%): Hệ thống không bỏ lọt bất kỳ hồ sơ vi phạm nào!")

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"📁 Đã xuất chi tiết 60 ca test sang file CSV: {output_csv}")

    return {
        "total_cases": total,
        "matched": matched,
        "accuracy": accuracy,
        "false_acceptances": len(false_acceptances),
        "false_acceptance_rate": far,
        "false_rejections": len(false_rejections),
        "false_rejection_rate": frr,
        "manual_review_triggers": manual_review_triggers,
        "results": results
    }


if __name__ == "__main__":
    out_csv = os.path.join(
        BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2", "rule_engine_boundary_test_results.csv"
    )
    run_rule_engine_boundary_benchmark(output_csv=out_csv)
