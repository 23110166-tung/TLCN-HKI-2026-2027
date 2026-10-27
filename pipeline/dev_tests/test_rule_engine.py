# -*- coding: utf-8 -*-
"""Unit tests cho pipeline/rule_engine.py – chạy: python -m pytest pipeline/dev_tests/test_rule_engine.py -q"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.rule_engine import (AcademicRuleEngine, split_score_tokens, parse_vn_date,
                                  normalize_digits, expected_grade)

ENGINE = AcademicRuleEngine(reference_date=date(2026, 10, 26))


def _doc(fname="bao_luu_sample_001.png", entities=None, courses=None, sig=1, stamp=1, dets=None):
    return {"document_file": fname, "entities": entities or {}, "courses_table": courses or [],
            "verification": {"signature_count": sig, "stamp_count": stamp}, "raw_detections": dets or []}


def _failed(report):
    return {v["rule_id"] for v in report["violations"]}


BASE = {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166",
        "email": "23110166@student.hcmute.edu.vn", "date": "20/08/2026", "dob": "01/05/2005"}


def test_helpers():
    assert normalize_digits("23 11O 166") == "23110166"
    assert parse_vn_date("ngày 20 tháng 08 năm 2026") == date(2026, 8, 20)
    assert parse_vn_date("31/02/2026") is None
    assert expected_grade(8.7) == (3.7, "A")
    assert split_score_tokens(["8.7", "3.7", "A"]) == (8.7, 3.7, "A")


def test_valid_form():
    r = ENGINE.validate(_doc(entities=dict(BASE)))
    assert r["form_type"] == "T05_BAO_LUU"
    assert r["status"] == "VALID", r["violations"]


def test_ocr_fixed_mssv_and_bad_mssv():
    r = ENGINE.validate(_doc(entities={**BASE, "student_id": "2311O166"}))
    assert r["normalized"]["student_id"] == "23110166" and r["status"] == "VALID"
    r = ENGINE.validate(_doc(entities={**BASE, "student_id": "1311016"}))
    assert "R10" in _failed(r) and r["status"] == "INVALID"


def test_missing_stamp_and_signatures():
    r = ENGINE.validate(_doc(stamp=0, entities=dict(BASE)))
    assert "R33" in _failed(r) and r["status"] == "INVALID"
    r = ENGINE.validate(_doc("diem_i_sample_003.png", sig=3, stamp=0,
                             entities={**BASE, "course_code": None},
                             courses=[{"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình"}]))
    assert r["form_type"] == "T01_DIEM_I" and "R32" in _failed(r) and "R33" not in _failed(r)


def test_future_date_and_email_mismatch():
    r = ENGINE.validate(_doc(entities={**BASE, "date": "15/12/2026",
                                       "email": "23110999@student.hcmute.edu.vn"}))
    assert {"R23", "R18"} <= _failed(r)


def test_academic_year_and_semester():
    dets = [{"class_name": "degree_info", "text": "Năm học: 2025-2027 Học kỳ: HK05"}]
    r = ENGINE.validate(_doc(entities=dict(BASE), dets=dets))
    assert {"R25", "R27"} <= _failed(r)


def test_transcript_grade_consistency():
    courses = [
        {"course_code": "PRTE230385", "course_name": "Kỹ thuật lập trình", "credits": "3",
         "scores": ["8.7", "3.7", "A"]},                       # đúng
        {"course_code": "DASA230179", "course_name": "Cấu trúc dữ liệu và giải thuật", "credits": "3",
         "scores": ["7.2", "3.5", "B+"]},                      # sai quy đổi
        {"course_code": "PHED110513", "course_name": "Giáo dục thể chất 1 *", "credits": "1",
         "scores": ["8.0", "3.5", "P"]},                       # môn P – bỏ qua quy đổi
        {"course_code": "OOPR230279", "course_name": "Lập trình hướng đối tượng", "credits": "4"},  # sai TC
    ]
    r = ENGINE.validate(_doc("transcript_sample_046.png", sig=0, stamp=0, courses=courses,
                             entities={"student_name": "Trần Cẩm Long", "student_id": "23110122"}))
    failed = _failed(r)
    assert r["form_type"] == "T10_BANG_DIEM"
    assert {"R49", "R4A", "R45"} <= failed
    assert "R32" not in failed and "R33" not in failed
    msgs = " ".join(v["message"] for v in r["violations"])
    assert "Dòng 3" not in msgs and "Dòng 1" not in msgs


def test_low_ocr_confidence_warning():
    dets = [{"class_name": "student_id", "text": "23110166", "ocr_confidence": 0.41}]
    r = ENGINE.validate(_doc(entities=dict(BASE), dets=dets))
    assert "R50" in _failed(r) and r["status"] == "WARNING"


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    test_funcs = [
        test_helpers,
        test_valid_form,
        test_ocr_fixed_mssv_and_bad_mssv,
        test_missing_stamp_and_signatures,
        test_future_date_and_email_mismatch,
        test_academic_year_and_semester,
        test_transcript_grade_consistency,
        test_low_ocr_confidence_warning
    ]
    
    passed = 0
    failed = 0
    print("=" * 60)
    print("🚀 BẮT ĐẦU CHẠY UNIT TEST RULE ENGINE (pipeline/rule_engine.py)")
    print("=" * 60)
    for fn in test_funcs:
        try:
            fn()
            print(f"  ✅ PASS: {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌ FAIL: {fn.__name__} - {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"📊 KẾT QUẢ: {passed}/{len(test_funcs)} test passed ({passed/len(test_funcs)*100:.1f}%)")
    if failed == 0:
        print("🎉 TẤT CẢ UNIT TEST CỦA RULE ENGINE ĐÃ HOÀN TOÀN CHÍNH XÁC!")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
