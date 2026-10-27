# -*- coding: utf-8 -*-
"""
Rule Engine – Kiểm tra tính hợp lệ nghiệp vụ cho đơn học vụ & bảng điểm sinh viên
Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh.

Module này tách biệt hoàn toàn logic nghiệp vụ khỏi phần học sâu (YOLOv11 + VietOCR):
đầu vào là kết quả có cấu trúc do `inference_engine.DocumentLayoutAndOCR` sinh ra,
đầu ra là báo cáo thẩm định gồm trạng thái tổng (VALID / WARNING / INVALID) và
danh sách chi tiết từng luật đã kiểm tra.

Nhóm luật (bám theo Đề cương – Mục 2.3.2):
    R1x – Định danh sinh viên  (MSSV, họ tên, lớp, email, SĐT, tính nhất quán)
    R2x – Thời gian            (ngày sinh, ngày làm đơn, năm học, học kỳ)
    R3x – Tính đầy đủ hồ sơ    (trường bắt buộc, chữ ký, con dấu theo loại mẫu T01–T10)
    R4x – Học phần & điểm số   (mã môn, tên môn, số tín chỉ, quy đổi điểm 10 ↔ 4 ↔ chữ)
    R5x – Chất lượng nhận dạng (độ tin cậy OCR của các trường trọng yếu)

Mức độ:
    ERROR   – vi phạm quy chế → hồ sơ INVALID
    WARNING – nghi vấn, cần cán bộ kiểm tra thủ công → hồ sơ WARNING
    INFO    – ghi chú tham khảo, không ảnh hưởng trạng thái
"""

import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from difflib import SequenceMatcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

SCHOOL_NAME = "Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh"

ERROR, WARNING, INFO = "ERROR", "WARNING", "INFO"
STATUS_VALID, STATUS_WARNING, STATUS_INVALID = "VALID", "WARNING", "INVALID"

# =====================================================================
# 1. CẤU HÌNH NGHIỆP VỤ
# =====================================================================

# Yêu cầu chữ ký / con dấu / trường bắt buộc theo từng loại biểu mẫu.
# Nguồn: mô tả nghiệp vụ trong pipeline/generate_print_dataset_100.py (TEMPLATES_9).
# "required_fields": mỗi phần tử là 1 trường hoặc tuple các trường thay thế nhau (chỉ cần 1).
FORM_REQUIREMENTS = {
    "T01_DIEM_I": {
        "name": "Đơn xin nhận và trả điểm I",
        "keywords": ["diem_i", "điểm i", "diem i"],
        "min_signatures": 5, "min_stamps": 0,
        "required_fields": ["student_name", "student_id", ("course_code", "course_name")],
        "signers": "Sinh viên, GV chấm thi 1, GV chấm thi 2, Trưởng khoa, Trưởng phòng ĐT",
    },
    "T02_HOC_2CTDT": {
        "name": "Đơn đăng ký học cùng lúc hai CTĐT",
        "keywords": ["hoc2ctdt", "2ctdt", "hai ctđt", "hai chương trình"],
        "min_signatures": 1, "min_stamps": 1,
        "required_fields": ["student_name", "student_id", "major"],
        "signers": "Sinh viên + Mộc Phòng Đào tạo",
    },
    "T03_CHUYEN_HE": {
        "name": "Đơn xin chuyển hệ đào tạo",
        "keywords": ["chuyen_he", "chuyển hệ", "chuyen he"],
        "min_signatures": 1, "min_stamps": 2,
        "required_fields": ["student_name", "student_id", "reason"],
        "signers": "Sinh viên + Mộc Ban Giám hiệu + Mộc Phòng Đào tạo",
    },
    "T04_CHUYEN_DIEM": {
        "name": "Đơn xin chuyển điểm học phần",
        "keywords": ["chuyen_diem_2018", "chuyển điểm học phần", "chuyen_diem"],
        "min_signatures": 1, "min_stamps": 1,
        "required_fields": ["student_name", "student_id", ("course_code", "course_name")],
        "signers": "Sinh viên + Mộc Phòng Đào tạo",
    },
    "T05_BAO_LUU": {
        "name": "Đơn xin bảo lưu điểm môn học",
        "keywords": ["bao_luu", "bảo lưu", "bao luu"],
        "min_signatures": 1, "min_stamps": 1,
        "required_fields": ["student_name", "student_id"],
        "signers": "Sinh viên + Mộc Phòng Đào tạo",
    },
    "T06_BAN_SAO_SO_GOC": {
        "name": "Đơn xin cấp bản sao từ sổ gốc",
        "keywords": ["ban_sao", "bản sao", "ban sao", "sổ gốc"],
        "min_signatures": 1, "min_stamps": 1,
        "required_fields": ["student_name", "degree_info"],
        "signers": "Người làm đơn + Mộc Phòng Đào tạo",
    },
    "T07_CHUYEN_DIEM_NN": {
        "name": "Đơn xin chuyển điểm Ngoại ngữ tốt nghiệp",
        "keywords": ["chuyen_diem_nn", "ngoại ngữ", "ngoai ngu"],
        "min_signatures": 1, "min_stamps": 1,
        "required_fields": ["student_name", "student_id"],
        "signers": "Sinh viên + Mộc Phòng Đào tạo",
    },
    "T08_THAY_THE_MON": {
        "name": "Đơn xin học môn thay thế",
        "keywords": ["thay_the", "thay thế", "thay the"],
        "min_signatures": 3, "min_stamps": 2,
        "required_fields": ["student_name", "student_id", ("course_code", "course_name")],
        "signers": "Sinh viên, CVHT, Trưởng khoa + Mộc Khoa + Mộc Phòng Đào tạo",
    },
    "T09_MIEN_THI": {
        "name": "Đơn xin miễn thi học phần",
        "keywords": ["mien_thi", "miễn thi", "mien thi"],
        "min_signatures": 1, "min_stamps": 1,
        "required_fields": ["student_name", "student_id", ("course_code", "course_name")],
        "signers": "Sinh viên + Mộc Phòng Đào tạo",
    },
    "T10_BANG_DIEM": {
        "name": "Bảng điểm sinh viên (Cổng thông tin đào tạo)",
        "keywords": ["transcript", "bang_diem", "bảng điểm", "bang diem"],
        "min_signatures": 0, "min_stamps": 0, "min_courses": 1,
        "required_fields": ["student_name", "student_id", "course_code"],
        "signers": "Không yêu cầu (bản in từ cổng thông tin)",
    },
}

UNKNOWN_FORM = {
    "name": "Không xác định",
    "keywords": [],
    "min_signatures": 1, "min_stamps": 0,
    "required_fields": ["student_name", "student_id"],
    "signers": "Tối thiểu chữ ký người làm đơn",
}

# Thang quy đổi điểm – khớp với pipeline/transcript_generator.get_grade_info
# (ngưỡng điểm 10 tối thiểu, điểm hệ 4, điểm chữ)
GRADE_SCALE = [
    (9.0, 4.0, "A+"),
    (8.5, 3.7, "A"),
    (8.0, 3.5, "B+"),
    (7.0, 3.0, "B"),
    (6.5, 2.5, "C+"),
    (5.5, 2.0, "C"),
    (5.0, 1.5, "D+"),
    (4.0, 1.0, "D"),
    (0.0, 0.0, "F"),
]
VALID_LETTERS = {g[2] for g in GRADE_SCALE} | {"P"}   # P = môn chỉ xét Đạt (GDTC, GDQP...)
VALID_SCORE4 = sorted({g[1] for g in GRADE_SCALE})

# Mã khoa (chữ số 3–5 của MSSV) đã biết
KNOWN_FACULTY_CODES = {"110", "115", "116", "119", "131", "141", "145", "151"}

MSSV_PATTERN = re.compile(r"^2\d{7}$")
COURSE_CODE_PATTERN = re.compile(r"^[A-Z]{2,5}\d{6}$")
PHONE_PATTERN = re.compile(r"^0[35789]\d{8}$")
EMAIL_PATTERN = re.compile(r"^([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})$")
ACADEMIC_YEAR_PATTERN = re.compile(r"(20\d{2})\s*[-–—/]\s*(20\d{2})")
SEMESTER_PATTERN = re.compile(r"(?:\bHK|Học\s*kỳ|Hoc\s*ky)\s*:?\s*([0-9]{1,2}|Hè|He)\b", re.IGNORECASE)

CRITICAL_OCR_FIELDS = {"student_id", "student_name", "course_code", "score", "credits", "date"}

# Bảng ký tự OCR hay nhầm khi trường chỉ chứa chữ số
_DIGIT_CONFUSIONS = str.maketrans({
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "I": "1", "l": "1", "|": "1", "i": "1", "!": "1",
    "Z": "2", "z": "2", "S": "5", "s": "5", "B": "8", "G": "6", "g": "9",
})


# =====================================================================
# 2. KIỂU DỮ LIỆU KẾT QUẢ
# =====================================================================

@dataclass
class RuleResult:
    rule_id: str
    name: str
    severity: str
    passed: bool
    message: str
    field: str = None
    value: object = None
    expected: object = None

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ValidationContext:
    """Trạng thái trung gian được chia sẻ giữa các luật."""
    entities: dict
    courses: list
    verification: dict
    detections: list
    form_type: str
    form_spec: dict
    normalized: dict = field(default_factory=dict)
    cohort_year: int = None


# =====================================================================
# 3. HÀM TIỆN ÍCH CHUẨN HÓA
# =====================================================================

def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [v for v in value if v not in (None, "")]
    return [value] if str(value).strip() else []


def _first(value):
    items = _as_list(value)
    return str(items[0]).strip() if items else ""


def _strip_accents(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def normalize_digits(text):
    """Chuẩn hóa trường số: bỏ khoảng trắng/dấu chấm và sửa ký tự OCR hay nhầm (O→0, I→1...)."""
    if text is None:
        return ""
    cleaned = re.sub(r"[\s.\-_,:]", "", str(text))
    # Bỏ tiền tố nhãn như "MSSV", "MSSV:" nếu OCR đọc dính
    cleaned = re.sub(r"^(MSSV|MSV|SĐT|SDT|ĐT|DT)", "", cleaned, flags=re.IGNORECASE)
    return cleaned.translate(_DIGIT_CONFUSIONS)


def normalize_course_code(text):
    code = re.sub(r"\s+", "", str(text or "")).upper()
    if len(code) > 6:
        # 6 ký tự cuối là số → sửa nhầm lẫn chữ/số ở phần đuôi
        code = code[:-6] + code[-6:].translate(_DIGIT_CONFUSIONS)
    return code


def normalize_name(text):
    text = re.sub(r"\s*\*\s*$", "", str(text or ""))   # bỏ dấu * (môn chỉ xét Đạt)
    text = _strip_accents(text).lower()
    return re.sub(r"[^a-z0-9 ]+", " ", text).split()


def parse_float(text):
    if text is None:
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", str(text))
    return float(m.group(0).replace(",", ".")) if m else None


def parse_vn_date(text):
    """Hỗ trợ: 12/08/2026, 12-8-2026, 12.08.2026, 'ngày 12 tháng 8 năm 2026'."""
    if not text:
        return None
    s = str(text)
    m = re.search(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4})", s)
    if not m:
        m = re.search(r"ng[aà]y\s*(\d{1,2}).*?th[aá]ng\s*(\d{1,2}).*?n[aă]m\s*(\d{4})",
                      s, flags=re.IGNORECASE)
    if not m:
        return None
    d, mo, y = (int(x) for x in m.groups())
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def expected_grade(score_10):
    for threshold, s4, letter in GRADE_SCALE:
        if score_10 >= threshold:
            return s4, letter
    return 0.0, "F"


def split_score_tokens(scores):
    """
    Tách danh sách box 'score' của 1 dòng (đã sắp theo trục x) thành
    (điểm hệ 10, điểm hệ 4, điểm chữ). Bảng điểm cổng đào tạo có 3 cột theo thứ tự này.
    """
    s10 = s4 = letter = None
    numerics = []
    for raw in _as_list(scores):
        tok = str(raw).strip().upper().replace(" ", "")
        if re.fullmatch(r"[A-DFP]\+?", tok):
            letter = tok
        else:
            val = parse_float(tok)
            if val is not None:
                numerics.append(val)
    if len(numerics) >= 2:
        s10, s4 = numerics[0], numerics[1]
    elif len(numerics) == 1:
        s10 = numerics[0]
    return s10, s4, letter


def build_course_catalog():
    """
    Gộp danh mục học phần từ CTĐT (curriculum_data) và ngân hàng môn (mock_data).
    Trả về {mã_môn: {"names": set[str], "credits": set[int]}}.
    """
    catalog = {}

    def _add(code, name, credits):
        entry = catalog.setdefault(code.upper(), {"names": set(), "credits": set()})
        entry["names"].add(name)
        entry["credits"].add(int(credits))

    try:
        from pipeline.curriculum_data import CURRICULUM_DATABASE
        for program in CURRICULUM_DATABASE.values():
            for _, _, courses in program["semesters"]:
                for code, name, credits in courses:
                    _add(code, name, credits)
    except Exception:
        pass
    try:
        from pipeline.mock_data import COURSES_POOL
        for c in COURSES_POOL:
            _add(c["code"], c["name"], c["credits"])
    except Exception:
        pass
    return catalog


# =====================================================================
# 4. RULE ENGINE
# =====================================================================

class AcademicRuleEngine:
    """
    Sử dụng:
        engine = AcademicRuleEngine()
        report = engine.validate(structured_result, form_type="T05_BAO_LUU")
    """

    def __init__(self, reference_date=None, course_catalog=None,
                 ocr_conf_threshold=0.75, name_similarity_threshold=0.80):
        self.reference_date = reference_date or date.today()
        self.catalog = course_catalog if course_catalog is not None else build_course_catalog()
        self.ocr_conf_threshold = ocr_conf_threshold
        self.name_similarity_threshold = name_similarity_threshold

    # -----------------------------------------------------------------
    # Xác định loại biểu mẫu
    # -----------------------------------------------------------------
    def resolve_form_type(self, form_type=None, result=None):
        """Trả về (form_type_id, nguồn xác định)."""
        result = result or {}
        if form_type:
            key = str(form_type).strip()
            if key.upper() in FORM_REQUIREMENTS:
                return key.upper(), "explicit"
            hit = self._match_keywords(key)
            if hit:
                return hit, "explicit"

        hit = self._match_keywords(result.get("document_file", ""))
        if hit:
            return hit, "filename"

        # Heuristic dựa trên bố cục đã phát hiện
        ver = result.get("verification", {})
        n_courses = len(result.get("courses_table", []))
        sigs, stamps = ver.get("signature_count", 0), ver.get("stamp_count", 0)
        if n_courses >= 6 and sigs == 0 and stamps == 0:
            return "T10_BANG_DIEM", "heuristic"
        if sigs >= 5 and stamps == 0:
            return "T01_DIEM_I", "heuristic"
        if result.get("entities", {}).get("degree_info") and not result.get("entities", {}).get("student_id"):
            return "T06_BAN_SAO_SO_GOC", "heuristic"
        return "UNKNOWN", "unknown"

    @staticmethod
    def _match_keywords(text):
        if not text:
            return None
        low = str(text).lower()
        low_plain = _strip_accents(low)
        # Ưu tiên từ khóa dài trước (vd 'chuyen_diem_nn' trước 'chuyen_diem')
        candidates = []
        for type_id, spec in FORM_REQUIREMENTS.items():
            for kw in spec["keywords"]:
                if kw in low or _strip_accents(kw) in low_plain:
                    candidates.append((len(kw), type_id))
        return max(candidates)[1] if candidates else None

    # -----------------------------------------------------------------
    # API chính
    # -----------------------------------------------------------------
    def validate(self, result, form_type=None):
        form_id, source = self.resolve_form_type(form_type, result)
        spec = FORM_REQUIREMENTS.get(form_id, UNKNOWN_FORM)

        ctx = ValidationContext(
            entities=result.get("entities", {}) or {},
            courses=result.get("courses_table", []) or [],
            verification=result.get("verification", {}) or {},
            detections=result.get("raw_detections", []) or [],
            form_type=form_id,
            form_spec=spec,
        )

        rules = []
        if form_id == "UNKNOWN":
            rules.append(RuleResult("R00", "Xác định loại biểu mẫu", WARNING, False,
                                    "Không xác định được loại biểu mẫu – áp dụng bộ luật tối thiểu."))

        for checker in (
            self._check_student_id,
            self._check_identity_consistency,
            self._check_student_name,
            self._check_class_name,
            self._check_email,
            self._check_phone,
            self._check_dob,
            self._check_document_dates,
            self._check_academic_period,
            self._check_required_fields,
            self._check_signatures,
            self._check_stamps,
            self._check_courses,
            self._check_ocr_confidence,
        ):
            rules.extend(checker(ctx))

        return self._summarize(rules, ctx, form_id, source)

    # -----------------------------------------------------------------
    # R1x – Định danh sinh viên
    # -----------------------------------------------------------------
    def _check_student_id(self, ctx):
        raw = _first(ctx.entities.get("student_id"))
        if not raw:
            return []   # thiếu trường → do R30 xử lý
        mssv = normalize_digits(raw)
        ctx.normalized["student_id"] = mssv
        out = []
        if mssv != re.sub(r"\s", "", raw):
            out.append(RuleResult("R10a", "Chuẩn hóa MSSV", INFO, True,
                                  f"Đã sửa lỗi OCR: '{raw}' → '{mssv}'.", "student_id", raw, mssv))

        if not MSSV_PATTERN.match(mssv):
            out.append(RuleResult("R10", "Định dạng MSSV", ERROR, False,
                                  "MSSV phải gồm 8 chữ số và bắt đầu bằng '2' (YY + mã khoa + STT).",
                                  "student_id", mssv, "^2[0-9]{7}$"))
            return out
        out.append(RuleResult("R10", "Định dạng MSSV", ERROR, True, "MSSV đúng định dạng 8 chữ số.",
                              "student_id", mssv))

        cohort = 2000 + int(mssv[:2])
        ctx.cohort_year = cohort
        ref_year = self.reference_date.year
        ok = ref_year - 12 <= cohort <= ref_year
        out.append(RuleResult("R11", "Khóa tuyển sinh từ MSSV", WARNING, ok,
                              f"Khóa {cohort} (K{mssv[:2]}) " +
                              ("hợp lệ." if ok else f"nằm ngoài khoảng [{ref_year - 12}, {ref_year}]."),
                              "student_id", cohort))

        fac = mssv[2:5]
        out.append(RuleResult("R12", "Mã khoa trong MSSV", INFO, fac in KNOWN_FACULTY_CODES,
                              f"Mã khoa '{fac}' " + ("thuộc danh mục." if fac in KNOWN_FACULTY_CODES
                                                    else "chưa có trong danh mục mã khoa đã biết."),
                              "student_id", fac))
        return out

    def _check_identity_consistency(self, ctx):
        out = []
        ids = {normalize_digits(v) for v in _as_list(ctx.entities.get("student_id"))}
        if len(ids) > 1:
            out.append(RuleResult("R13", "Nhất quán MSSV trong văn bản", ERROR, False,
                                  "Văn bản chứa nhiều MSSV khác nhau.", "student_id", sorted(ids)))
        names = {" ".join(normalize_name(v)) for v in _as_list(ctx.entities.get("student_name"))}
        if len(names) > 1:
            out.append(RuleResult("R14", "Nhất quán họ tên trong văn bản", WARNING, False,
                                  "Họ tên sinh viên xuất hiện với nhiều cách viết khác nhau.",
                                  "student_name", sorted(names)))
        return out

    def _check_student_name(self, ctx):
        name = _first(ctx.entities.get("student_name"))
        if not name:
            return []
        tokens = name.split()
        has_digit = bool(re.search(r"\d", name))
        bad_chars = bool(re.search(r"[^\w\s]", name))
        ok = len(tokens) >= 2 and not has_digit and not bad_chars
        msg = "Họ tên hợp lệ." if ok else "Họ tên phải có ≥ 2 từ, không chứa chữ số/ký tự đặc biệt."
        return [RuleResult("R15", "Định dạng họ tên", WARNING, ok, msg, "student_name", name)]

    def _check_class_name(self, ctx):
        cls = _first(ctx.entities.get("class_name"))
        mssv = ctx.normalized.get("student_id")
        if not cls or not mssv or not MSSV_PATTERN.match(mssv):
            return []
        cls_n = re.sub(r"\s", "", cls).upper()
        ok = cls_n.startswith(mssv[:2])
        return [RuleResult("R16", "Mã lớp khớp khóa MSSV", WARNING, ok,
                           "Mã lớp khớp khóa tuyển sinh." if ok else
                           f"Mã lớp '{cls_n}' không bắt đầu bằng khóa '{mssv[:2]}' của MSSV.",
                           "class_name", cls_n, mssv[:2] + "...")]

    def _check_email(self, ctx):
        email = _first(ctx.entities.get("email"))
        if not email:
            return []
        email = re.sub(r"\s", "", email)
        m = EMAIL_PATTERN.match(email)
        if not m:
            return [RuleResult("R17", "Định dạng email", WARNING, False,
                               "Email không đúng định dạng.", "email", email)]
        out = [RuleResult("R17", "Định dạng email", WARNING, True, "Email đúng định dạng.", "email", email)]
        local, domain = m.groups()
        mssv = ctx.normalized.get("student_id")
        if mssv and local.isdigit():
            ok = local == mssv
            out.append(RuleResult("R18", "Email khớp MSSV", WARNING, ok,
                                  "Email sinh viên khớp MSSV." if ok else
                                  f"Phần trước '@' ({local}) khác MSSV ({mssv}).", "email", local, mssv))
        if not domain.lower().endswith("hcmute.edu.vn"):
            out.append(RuleResult("R19", "Tên miền email của trường", INFO, False,
                                  "Email không dùng tên miền của trường.", "email", domain,
                                  "student.hcmute.edu.vn"))
        return out

    def _check_phone(self, ctx):
        phone = _first(ctx.entities.get("phone"))
        if not phone:
            return []
        p = normalize_digits(phone).replace("+84", "0")
        if p.startswith("84") and len(p) == 11:
            p = "0" + p[2:]
        ok = bool(PHONE_PATTERN.match(p))
        return [RuleResult("R1A", "Định dạng số điện thoại", WARNING, ok,
                           "SĐT di động hợp lệ (10 số)." if ok else "SĐT phải gồm 10 số, đầu 03/05/07/08/09.",
                           "phone", p)]

    # -----------------------------------------------------------------
    # R2x – Thời gian
    # -----------------------------------------------------------------
    def _check_dob(self, ctx):
        raw = _first(ctx.entities.get("dob"))
        if not raw:
            return []
        dob = parse_vn_date(raw)
        if dob is None:
            return [RuleResult("R20", "Ngày sinh hợp lệ", WARNING, False,
                               "Không đọc được ngày sinh (dd/mm/yyyy).", "dob", raw)]
        out = []
        age = (self.reference_date - dob).days // 365
        ok = 16 <= age <= 60
        out.append(RuleResult("R20", "Ngày sinh hợp lệ", WARNING, ok,
                              f"Tuổi hiện tại: {age}." + ("" if ok else " Ngoài khoảng hợp lý 16–60."),
                              "dob", dob.isoformat()))
        if ctx.cohort_year:
            enroll_age = ctx.cohort_year - dob.year
            ok2 = 16 <= enroll_age <= 40
            out.append(RuleResult("R21", "Tuổi nhập học khớp MSSV", WARNING, ok2,
                                  f"Tuổi khi nhập học (khóa {ctx.cohort_year}): {enroll_age}." +
                                  ("" if ok2 else " Bất thường so với khóa tuyển sinh."),
                                  "dob", enroll_age, "16–40"))
        return out

    def _check_document_dates(self, ctx):
        out = []
        for raw in _as_list(ctx.entities.get("date")):
            d = parse_vn_date(raw)
            if d is None:
                out.append(RuleResult("R22", "Ngày làm đơn đọc được", WARNING, False,
                                      "Không phân tích được ngày tháng.", "date", raw))
                continue
            not_future = d <= self.reference_date
            out.append(RuleResult("R23", "Ngày văn bản không ở tương lai", ERROR, not_future,
                                  f"Ngày {d.strftime('%d/%m/%Y')} " +
                                  ("hợp lệ." if not_future else
                                   f"sau ngày thẩm định ({self.reference_date.strftime('%d/%m/%Y')})."),
                                  "date", d.isoformat()))
            if ctx.cohort_year and ctx.form_type != "T06_BAN_SAO_SO_GOC":
                ok = d.year >= ctx.cohort_year
                out.append(RuleResult("R24", "Ngày văn bản sau năm nhập học", WARNING, ok,
                                      "Hợp lệ." if ok else
                                      f"Ngày {d.strftime('%d/%m/%Y')} trước năm nhập học {ctx.cohort_year}.",
                                      "date", d.isoformat()))
        return out

    def _check_academic_period(self, ctx):
        texts = [str(d.get("text", "")) for d in ctx.detections]
        texts += [str(v) for v in ctx.entities.values() if isinstance(v, str)]
        out, seen_years, seen_sems = [], set(), set()
        for t in texts:
            for y1, y2 in ACADEMIC_YEAR_PATTERN.findall(t):
                seen_years.add((int(y1), int(y2)))
            for s in SEMESTER_PATTERN.findall(t):
                seen_sems.add(s)

        for y1, y2 in sorted(seen_years):
            ok = (y2 == y1 + 1) and (y1 <= self.reference_date.year)
            out.append(RuleResult("R25", "Năm học hợp lệ", ERROR, ok,
                                  f"Năm học {y1}-{y2} " + ("hợp lệ." if ok else
                                                          "sai (năm sau phải = năm trước + 1, không ở tương lai)."),
                                  "academic_year", f"{y1}-{y2}"))
            if ok and ctx.cohort_year and y1 < ctx.cohort_year:
                out.append(RuleResult("R26", "Năm học sau khóa tuyển sinh", WARNING, False,
                                      f"Năm học {y1}-{y2} trước khóa nhập học {ctx.cohort_year}.",
                                      "academic_year", f"{y1}-{y2}"))
        for s in sorted(seen_sems):
            s_norm = _strip_accents(s).lower()
            ok = s_norm in {"1", "2", "3", "01", "02", "03", "he"}
            out.append(RuleResult("R27", "Học kỳ hợp lệ", ERROR, ok,
                                  f"Học kỳ '{s}' " + ("hợp lệ." if ok else "không tồn tại (chỉ HK1, HK2, HK3/Hè)."),
                                  "semester", s))
        return out

    # -----------------------------------------------------------------
    # R3x – Tính đầy đủ hồ sơ
    # -----------------------------------------------------------------
    def _field_present(self, ctx, name):
        if _as_list(ctx.entities.get(name)):
            return True
        return any(str(row.get(name, "")).strip() for row in ctx.courses)

    def _check_required_fields(self, ctx):
        out = []
        for req in ctx.form_spec["required_fields"]:
            alts = req if isinstance(req, tuple) else (req,)
            ok = any(self._field_present(ctx, f) for f in alts)
            label = " / ".join(alts)
            out.append(RuleResult("R30", f"Trường bắt buộc: {label}", ERROR, ok,
                                  "Có thông tin." if ok else "Thiếu hoặc không nhận dạng được.", label))
        min_courses = ctx.form_spec.get("min_courses")
        if min_courses:
            n = sum(1 for r in ctx.courses if r.get("course_code"))
            out.append(RuleResult("R31", "Số học phần tối thiểu", ERROR, n >= min_courses,
                                  f"Nhận dạng được {n} học phần.", "courses_table", n, f">= {min_courses}"))
        return out

    def _check_signatures(self, ctx):
        need = ctx.form_spec["min_signatures"]
        if need == 0:
            return []
        have = int(ctx.verification.get("signature_count", 0))
        ok = have >= need
        return [RuleResult("R32", "Đủ chữ ký theo loại mẫu", ERROR, ok,
                           f"Phát hiện {have}/{need} chữ ký ({ctx.form_spec['signers']})." +
                           ("" if ok else f" Thiếu {need - have} chữ ký."),
                           "signature", have, need)]

    def _check_stamps(self, ctx):
        need = ctx.form_spec["min_stamps"]
        if need == 0:
            return []
        have = int(ctx.verification.get("stamp_count", 0))
        ok = have >= need
        return [RuleResult("R33", "Đủ con dấu theo loại mẫu", ERROR, ok,
                           f"Phát hiện {have}/{need} mộc đỏ." + ("" if ok else f" Thiếu {need - have} con dấu."),
                           "stamp", have, need)]

    # -----------------------------------------------------------------
    # R4x – Học phần & điểm số
    # -----------------------------------------------------------------
    def _check_courses(self, ctx):
        out, seen = [], {}
        checked = []
        for idx, row in enumerate(ctx.courses, start=1):
            tag = f"Dòng {idx}"
            code_raw = row.get("course_code", "")
            code = normalize_course_code(code_raw) if code_raw else ""
            entry = None
            row_report = {"row": idx, "course_code": code or None}

            if code:
                if not COURSE_CODE_PATTERN.match(code):
                    out.append(RuleResult("R40", "Định dạng mã học phần", ERROR, False,
                                          f"{tag}: mã '{code}' sai định dạng (2–5 chữ cái + 6 chữ số).",
                                          "course_code", code))
                else:
                    entry = self.catalog.get(code)
                    if entry is None:
                        out.append(RuleResult("R41", "Mã học phần có trong CTĐT", WARNING, False,
                                              f"{tag}: mã '{code}' không có trong danh mục CTĐT.",
                                              "course_code", code))
                    if code in seen:
                        out.append(RuleResult("R42", "Học phần trùng lặp", WARNING, False,
                                              f"{tag}: mã '{code}' lặp lại (dòng {seen[code]}) – có thể là học lại/cải thiện.",
                                              "course_code", code))
                    seen.setdefault(code, idx)

            # Tên môn so khớp danh mục
            name = row.get("course_name", "")
            if entry and name:
                ocr_tokens = " ".join(normalize_name(name))
                best_name, best = None, 0.0
                for ref in entry["names"]:
                    r = SequenceMatcher(None, ocr_tokens, " ".join(normalize_name(ref))).ratio()
                    if r > best:
                        best_name, best = ref, r
                ok = best >= self.name_similarity_threshold
                row_report["name_similarity"] = round(best, 3)
                if not ok:
                    out.append(RuleResult("R43", "Tên học phần khớp mã", WARNING, False,
                                          f"{tag}: tên '{name}' không khớp mã {code} (độ tương đồng {best:.0%}).",
                                          "course_name", name, best_name))

            # Số tín chỉ
            cred_raw = row.get("credits", "")
            if cred_raw:
                cred = parse_float(normalize_digits(cred_raw))
                if cred is None or cred != int(cred) or not (1 <= cred <= 15):
                    out.append(RuleResult("R44", "Số tín chỉ hợp lệ", ERROR, False,
                                          f"{tag}: số tín chỉ '{cred_raw}' không hợp lệ (số nguyên 1–15).",
                                          "credits", cred_raw))
                elif entry and int(cred) not in entry["credits"]:
                    out.append(RuleResult("R45", "Số tín chỉ khớp CTĐT", WARNING, False,
                                          f"{tag}: {code} có {int(cred)} TC, CTĐT quy định {sorted(entry['credits'])}.",
                                          "credits", int(cred), sorted(entry["credits"])))
                row_report["credits"] = cred

            out.extend(self._check_scores(row, tag, row_report))
            checked.append(row_report)

        n_codes = sum(1 for r in ctx.courses if r.get("course_code"))
        if n_codes:
            n_err = sum(1 for r in out if not r.passed and r.severity == ERROR)
            out.insert(0, RuleResult("R4*", "Tổng hợp kiểm tra học phần", INFO, n_err == 0,
                                     f"Đã kiểm tra {n_codes} học phần, {n_err} lỗi nghiêm trọng."))
        ctx.normalized["courses_checked"] = checked
        return out

    def _check_scores(self, row, tag, row_report):
        out = []
        s10, s4, letter = split_score_tokens(row.get("scores") or row.get("score"))
        row_report.update({"score_10": s10, "score_4": s4, "letter": letter})
        if s10 is None and s4 is None and letter is None:
            return out   # học kỳ đang học – chưa có điểm

        if s10 is not None and not (0.0 <= s10 <= 10.0):
            out.append(RuleResult("R46", "Điểm hệ 10 trong khoảng", ERROR, False,
                                  f"{tag}: điểm hệ 10 = {s10} nằm ngoài [0, 10].", "score", s10))
            return out
        if s4 is not None and s4 not in VALID_SCORE4:
            out.append(RuleResult("R47", "Điểm hệ 4 hợp lệ", ERROR, False,
                                  f"{tag}: điểm hệ 4 = {s4} không thuộc thang {VALID_SCORE4}.", "score", s4))
        if letter is not None and letter not in VALID_LETTERS:
            out.append(RuleResult("R48", "Điểm chữ hợp lệ", ERROR, False,
                                  f"{tag}: điểm chữ '{letter}' không hợp lệ.", "score", letter))

        if letter == "P":
            return out   # môn chỉ xét Đạt/Không đạt: không quy đổi

        if s10 is not None:
            exp4, exp_letter = expected_grade(s10)
            if s4 is not None and s4 in VALID_SCORE4 and abs(s4 - exp4) > 1e-6:
                out.append(RuleResult("R49", "Quy đổi điểm 10 → 4", ERROR, False,
                                      f"{tag}: {s10} (hệ 10) phải tương ứng {exp4} (hệ 4), văn bản ghi {s4}.",
                                      "score", s4, exp4))
            if letter is not None and letter in VALID_LETTERS and letter != exp_letter:
                out.append(RuleResult("R4A", "Quy đổi điểm 10 → chữ", ERROR, False,
                                      f"{tag}: {s10} (hệ 10) phải là '{exp_letter}', văn bản ghi '{letter}'.",
                                      "score", letter, exp_letter))
        elif s4 is not None and letter is not None and letter in VALID_LETTERS:
            exp_letters = {g[2] for g in GRADE_SCALE if abs(g[1] - s4) < 1e-6}
            if letter not in exp_letters:
                out.append(RuleResult("R4B", "Quy đổi điểm 4 → chữ", ERROR, False,
                                      f"{tag}: {s4} (hệ 4) không tương ứng điểm chữ '{letter}'.",
                                      "score", letter, sorted(exp_letters)))
        return out

    # -----------------------------------------------------------------
    # R5x – Chất lượng nhận dạng (human-in-the-loop)
    # -----------------------------------------------------------------
    def _check_ocr_confidence(self, ctx):
        low = []
        for d in ctx.detections:
            conf = d.get("ocr_confidence")
            if conf is None or d.get("class_name") not in CRITICAL_OCR_FIELDS:
                continue
            if conf < self.ocr_conf_threshold:
                low.append(f"{d['class_name']}='{d.get('text', '')}' ({conf:.0%})")
        if not low:
            return []
        preview = "; ".join(low[:8]) + (" ..." if len(low) > 8 else "")
        return [RuleResult("R50", "Độ tin cậy OCR trường trọng yếu", WARNING, False,
                           f"{len(low)} trường có độ tin cậy < {self.ocr_conf_threshold:.0%}, "
                           f"cần cán bộ đối chiếu bản gốc: {preview}", "ocr", len(low))]

    # -----------------------------------------------------------------
    # Tổng hợp
    # -----------------------------------------------------------------
    def _summarize(self, rules, ctx, form_id, source):
        failed_err = [r for r in rules if not r.passed and r.severity == ERROR]
        failed_warn = [r for r in rules if not r.passed and r.severity == WARNING]
        scored = [r for r in rules if r.severity in (ERROR, WARNING)]

        if failed_err:
            status = STATUS_INVALID
        elif failed_warn:
            status = STATUS_WARNING
        else:
            status = STATUS_VALID

        # Điểm tuân thủ: ERROR trọng số 2, WARNING trọng số 1
        total_w = sum(2 if r.severity == ERROR else 1 for r in scored) or 1
        pass_w = sum(2 if r.severity == ERROR else 1 for r in scored if r.passed)

        status_text = {
            STATUS_VALID: "Hồ sơ hợp lệ – đủ điều kiện tiếp nhận.",
            STATUS_WARNING: "Hồ sơ có điểm nghi vấn – cần cán bộ kiểm tra thủ công.",
            STATUS_INVALID: "Hồ sơ không hợp lệ – vi phạm quy chế/thiếu thành phần bắt buộc.",
        }[status]

        return {
            "school": SCHOOL_NAME,
            "reference_date": self.reference_date.isoformat(),
            "form_type": form_id,
            "form_type_name": FORM_REQUIREMENTS.get(form_id, UNKNOWN_FORM)["name"],
            "form_type_source": source,
            "status": status,
            "status_text": status_text,
            "compliance_score": round(100.0 * pass_w / total_w, 1),
            "summary": {
                "total_rules": len(rules),
                "passed": sum(1 for r in rules if r.passed),
                "errors": len(failed_err),
                "warnings": len(failed_warn),
            },
            "violations": [r.to_dict() for r in failed_err + failed_warn],
            "rules": [r.to_dict() for r in rules],
            "normalized": ctx.normalized,
        }


def format_report(report):
    """Chuỗi văn bản dễ đọc để in ra console / log."""
    icon = {STATUS_VALID: "✅", STATUS_WARNING: "⚠️", STATUS_INVALID: "❌"}[report["status"]]
    lines = [
        f"{icon} [{report['status']}] {report['form_type_name']} ({report['form_type']}, nguồn: {report['form_type_source']})",
        f"   {report['status_text']}  |  Điểm tuân thủ: {report['compliance_score']}%",
        f"   Luật: {report['summary']['total_rules']}  |  Đạt: {report['summary']['passed']}  |  "
        f"Lỗi: {report['summary']['errors']}  |  Cảnh báo: {report['summary']['warnings']}",
    ]
    for v in report["violations"]:
        lines.append(f"   - {v['rule_id']} [{v['severity']}] {v['message']}")
    return "\n".join(lines)


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    demo = {
        "document_file": "bao_luu_sample_001.png",
        "entities": {"student_name": "Nguyễn Bách Tùng", "student_id": "23110166",
                     "email": "23110166@student.hcmute.edu.vn", "date": "ngày 20 tháng 08 năm 2026"},
        "courses_table": [],
        "verification": {"signature_count": 1, "stamp_count": 0},
        "raw_detections": [],
    }
    print(format_report(AcademicRuleEngine(reference_date=date(2026, 10, 26)).validate(demo)))
