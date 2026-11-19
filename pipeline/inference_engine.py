# -*- coding: utf-8 -*-
"""
End-to-End Inference Engine for HCMUTE Academic Documents
(Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh)

Input Image -> YOLOv11 Layout Detection -> Text Crops -> VietOCR (fine-tuned) Recognition
            -> Structured JSON -> Rule Engine (kiểm tra tính hợp lệ nghiệp vụ)
"""

import os
import sys
import json
import re
import difflib
import unicodedata
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from pipeline.ontology import CLASS_NAMES, CLASS_MAP, CLASS_COLORS
from pipeline.rule_engine import AcademicRuleEngine, format_report
from pipeline.mock_data import COURSES_POOL

# Trọng số mặc định (tải về từ Kaggle – không commit lên GitHub do > 100MB)
DOMAIN_ADAPTED_V2_YOLO_WEIGHTS = os.path.join(
    BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2", "yolo11_domain_adapted_best.pt"
)
DOMAIN_ADAPTED_V2_ONNX_WEIGHTS = os.path.join(
    BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2", "yolo11_domain_adapted_best.onnx"
)
DOMAIN_ADAPTED_YOLO_WEIGHTS = os.path.join(
    BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts", "yolo11_domain_adapted_best.pt"
)
SYNTHETIC_YOLO_WEIGHTS = os.path.join(
    BASE_DIR, "HCMUTE_DocAI_YOLOv11_VietOCR_Artifacts", "runs", "detect",
    "hcmute_yolo11_layout", "weights", "best.pt"
)
if os.path.exists(DOMAIN_ADAPTED_V2_YOLO_WEIGHTS):
    DEFAULT_YOLO_WEIGHTS = DOMAIN_ADAPTED_V2_YOLO_WEIGHTS
elif os.path.exists(DOMAIN_ADAPTED_YOLO_WEIGHTS):
    DEFAULT_YOLO_WEIGHTS = DOMAIN_ADAPTED_YOLO_WEIGHTS
else:
    DEFAULT_YOLO_WEIGHTS = SYNTHETIC_YOLO_WEIGHTS
DEFAULT_OCR_WEIGHTS = os.path.join(
    BASE_DIR, "HCMUTE_VietOCR_Finetuned_Artifacts", "vietocr_hcmute_finetuned.pth"
)

# Cấu hình dự phòng offline cho VietOCR vgg_transformer (tránh lỗi khi không có mạng tới vocr.vn)
DEFAULT_OFFLINE_VIETOCR_CONFIG = {
    'vocab': 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ',
    'device': 'cpu',
    'seq_modeling': 'transformer',
    'transformer': {
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'dim_feedforward': 2048,
        'max_seq_length': 1024,
        'pos_dropout': 0.1,
        'trans_dropout': 0.1
    },
    'optimizer': {'max_lr': 0.0003, 'min_lr': 1e-05},
    'trainer': {'batch_size': 32, 'print_every': 200, 'valid_every': 1000, 'iters': 20000, 'export': './weights/transformerocr.pth', 'checkpoint': './checkpoint/transformerocr.pth', 'log': './train.log', 'eval_batches': 500},
    'dataset': {'name': 'data', 'data_root': './img/', 'train_annotation': 'train_line_annotation.txt', 'valid_annotation': 'val_line_annotation.txt', 'image_height': 32, 'image_min_width': 32, 'image_max_width': 512},
    'dataloader': {'num_workers': 3, 'pin_memory': True},
    'quiet': False,
    'backbone': 'vgg19_bn',
    'cnn': {'pretrained': False, 'ss': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]], 'ks': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]], 'hidden': 256},
    'predictor': {'beamsearch': False}
}

ENTITY_CLASSES = ["student_name", "student_id", "class_name", "faculty", "major", "dob", "pob",
                  "phone", "email", "date", "reason", "degree_info"]
TABLE_CLASSES = ["course_code", "course_name", "credits", "score"]

STATUS_COLORS = {"VALID": (40, 167, 69), "WARNING": (230, 150, 0), "INVALID": (220, 53, 69)}


def _build_known_courses():
    """Tổng hợp danh mục môn học chuẩn từ COURSES_POOL, CURRICULUM_DATABASE và các học phần Anh văn chuẩn."""
    pool = []
    seen = set()

    def _add(code, name, credits):
        c_up = str(code).strip().upper()
        n_clean = str(name).strip()
        key = (c_up, n_clean.lower())
        if c_up and key not in seen:
            seen.add(key)
            pool.append({"code": c_up, "name": n_clean, "credits": credits})

    for c in COURSES_POOL:
        _add(c["code"], c["name"], c.get("credits"))

    try:
        from pipeline.curriculum_data import CURRICULUM_DATABASE
        for program in CURRICULUM_DATABASE.values():
            for _, _, courses in program.get("semesters", []):
                for code, name, credits in courses:
                    _add(code, name, credits)
    except Exception:
        pass

    # Bổ sung nhóm học phần Anh văn chuẩn thường gặp trên đơn T07 Chuyển điểm NN
    for code, name, cred in [
        ("ENGL130137", "Anh văn 1", 3),
        ("ENGL230237", "Anh văn 2", 3),
        ("ENGL330337", "Anh văn 3", 3),
        ("ENGL430437", "Anh văn 4", 3),
        ("ENGL530537", "Anh văn 5", 3),
        ("EHQT230437", "Anh văn 4", 3),
        ("EHQT330537", "Anh văn 5", 3),
    ]:
        _add(code, name, cred)

    return pool


ALL_KNOWN_COURSES = _build_known_courses()
_CODE_DIGIT_MAP = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "L": "1",
    "S": "5", "B": "8", "E": "8", "G": "6", "A": "4", "Z": "2",
})


def _normalize_course_code_ocr(raw_code):
    """Chuẩn hóa mã môn học từ OCR, tách riêng phần mã khỏi tên môn bị dính nếu có."""
    s = re.sub(r"[^A-Za-z0-9]", "", str(raw_code or "")).upper()
    m = re.match(r"^([A-Z]{2,5})([A-Z0-9]{6})", s)
    if m:
        prefix, tail = m.groups()
        # Chỉ sửa chữ thành số ở 6 ký tự số cuối
        return prefix + tail.translate(_CODE_DIGIT_MAP)
    if len(s) >= 8 and s[:4].isalpha():
        return s[:4] + s[4:10].translate(_CODE_DIGIT_MAP)
    return s


def _is_ocr_hallucination(txt):
    """Nhận diện chuỗi ảo giác lặp từ của VietOCR trên vùng ảnh trống hoặc nhiễu kẻ bảng."""
    if not txt:
        return True
    s = str(txt).strip().lower()
    words = s.split()
    if len(words) >= 8 and len(set(words)) <= max(2, int(len(words) * 0.45)):
        return True
    if len(words) >= 13:
        return True
    return False


def ground_course(code_raw, name_raw):
    """
    Chuẩn hóa và hoàn thiện mã môn học / tên môn học dựa trên Knowledge Base đầy đủ
    của Trường ĐH Sư phạm Kỹ thuật TP.HCM (HCMUTE). Khắc phục hiện tượng OCR bị nhầm ký tự,
    dính tên môn vào mã môn, hoặc xuống dòng trong các ô bảng hẹp.
    """
    raw_c_str = str(code_raw or "").strip()
    raw_n_str = str(name_raw or "").strip()

    if _is_ocr_hallucination(raw_c_str):
        raw_c_str = ""
    if _is_ocr_hallucination(raw_n_str):
        raw_n_str = ""
    if not raw_c_str and not raw_n_str:
        return "", "", None, 0.0

    # Nếu mã môn bị dính kèm tên môn (vd 'PHED110513GIÁODỤCTHỂCHẤT1-4'), tách phần mã ra
    m_glued = re.match(r"^([A-Za-z]{2,5}\s*[A-Za-z0-9]{6})\s*(.*)$", raw_c_str)
    if m_glued and len(re.sub(r"[^A-Za-z0-9]", "", m_glued.group(1))) >= 8:
        cand_code = _normalize_course_code_ocr(m_glued.group(1))
        if not raw_n_str and m_glued.group(2).strip():
            raw_n_str = m_glued.group(2).strip()
        code_clean = cand_code
    else:
        code_clean = _normalize_course_code_ocr(raw_c_str)

    name_clean = re.sub(r"\s*\*\s*$", "", raw_n_str).lower().strip()
    if len(code_clean) < 4 and len(name_clean) < 4:
        return code_clean or raw_c_str, raw_n_str, None, 0.0

    best_c, best_score = None, 0.0
    for c in ALL_KNOWN_COURSES:
        c_code = c["code"]
        c_name = c["name"]
        c_name_low = re.sub(r"\s*\*\s*$", "", c_name).lower().strip()

        c_sim = 0.0
        if code_clean and len(code_clean) >= 5:
            c_sim = difflib.SequenceMatcher(None, code_clean, c_code).ratio()
            if code_clean == c_code:
                c_sim = 1.0
            elif len(code_clean) >= 8 and code_clean[-6:] == c_code[-6:]:
                c_sim = max(c_sim, 0.93)
            elif c_code.startswith(code_clean) and len(code_clean) >= 6:
                c_sim = max(c_sim, 0.92)

        n_sim = 0.0
        if name_clean and len(name_clean) >= 5:
            n_sim = difflib.SequenceMatcher(None, name_clean, c_name_low).ratio()
            if c_name_low == name_clean:
                n_sim = 1.0
            elif len(name_clean) >= 8 and min(len(name_clean), len(c_name_low)) >= 0.65 * max(len(name_clean), len(c_name_low)):
                if c_name_low.startswith(name_clean) or name_clean.startswith(c_name_low):
                    n_sim = max(n_sim, 0.86)

        # Kết hợp đồng thời độ tương đồng mã + tên để phân biệt môn cùng mã tiền tố (vd Anh văn 1 vs 2)
        if c_sim >= 0.75 and n_sim >= 0.55:
            score = 0.55 * c_sim + 0.45 * n_sim + 0.10
        else:
            score = max(c_sim, n_sim)

        if score > best_score:
            best_score = score
            best_c = c

    if best_score >= 0.72 and best_c:
        return best_c["code"], best_c["name"], best_c.get("credits"), min(1.0, best_score)
    return code_clean or raw_c_str, raw_n_str, None, best_score


def clean_academic_score(score_raw):
    """
    Chuẩn hóa điểm số học phần theo thang điểm 10 của quy chế đào tạo tín chỉ HCMUTE.
    Khắc phục các lỗi nhầm lẫn ký tự OCR (như 6A -> 6.4, 8.6.0 -> 8.6, 60 -> 6.0, 64 -> 6.4),
    đồng thời lọc bỏ chuỗi chữ rác không phải điểm số.
    """
    if not score_raw:
        return ""
    raw_str = str(score_raw).strip()
    # Nếu chứa nhiều từ tiếng Việt (ảo giác OCR trên ô trống), trả về rỗng
    if len(raw_str.split()) >= 2 and not re.search(r"\d", raw_str):
        return ""
    if re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", raw_str, re.IGNORECASE):
        if not re.search(r"\d", raw_str):
            return ""
    # Thay thế các ký tự chữ cái thường bị nhận nhầm từ chữ số
    s = re.sub(r"[A-Za-z]", lambda m: {"A": "4", "a": "4", "O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "s": "5", "B": "8", "b": "8", "G": "6"}.get(m.group(0), ""), raw_str)
    m = re.search(r"(\d+(?:[.,]\d+)?)", s)
    if not m:
        return ""
    val_str = m.group(1).replace(",", ".")
    try:
        val = float(val_str)
        if 10.0 < val <= 100.0:
            val = val / 10.0
            return f"{val:.1f}"
        elif val <= 10.0:
            return f"{val:.1f}" if ("." in val_str or val > 0) else val_str
    except Exception:
        pass
    return val_str


class DocumentLayoutAndOCR:
    """
    End-to-End Pipeline:
    Input Image -> YOLOv11 Layout Detection -> Text Crops -> VietOCR Recognition
                -> Structured JSON -> Rule Engine validation
    """

    def __init__(
        self,
        yolo_model_path=None,
        ocr_model_name="vgg_transformer",
        ocr_weights_path=None,
        device="cpu",
        conf_threshold=0.35,
        rule_engine=None,
        imgsz=1280,
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.yolo_model = None
        self.ocr_predictor = None
        self.ocr_engine_type = None
        self.ocr_weights_used = None
        self.rule_engine = rule_engine or AcademicRuleEngine()

        # 1. Load YOLO Model (ưu tiên trọng số fine-tuned của đề tài)
        yolo_model_path = yolo_model_path or DEFAULT_YOLO_WEIGHTS
        self.yolo_model_path_used = yolo_model_path
        if YOLO is None:
            print("⚠️ Chưa cài ultralytics – bỏ qua bước Layout Detection.")
        elif os.path.exists(yolo_model_path):
            print(f"📦 Đang tải mô hình YOLO Layout từ: {yolo_model_path}")
            self.yolo_model = YOLO(yolo_model_path, task="detect")
        else:
            print("⚠️ Không tìm thấy trọng số fine-tuned, đang khởi tạo YOLO11n mặc định...")
            self.yolo_model = YOLO("yolo11n.pt", task="detect")

        # 2. Load VietOCR Predictor (ưu tiên trọng số fine-tuned – RQ1: EM 51.67% → 69.78%)
        ocr_weights_path = ocr_weights_path or DEFAULT_OCR_WEIGHTS
        try:
            from vietocr.tool.predictor import Predictor
            from vietocr.tool.config import Cfg

            import copy
            cfg = Cfg(copy.deepcopy(DEFAULT_OFFLINE_VIETOCR_CONFIG))
            cfg['device'] = self.device
            cfg['predictor']['beamsearch'] = False
            if ocr_weights_path and os.path.exists(ocr_weights_path):
                # Predictor nạp trọng số từ cfg['weights'] (không phải 'prediction_chk')
                cfg['weights'] = ocr_weights_path
                cfg['cnn']['pretrained'] = False
                self.ocr_weights_used = ocr_weights_path
            else:
                print("⚠️ Không tìm thấy trọng số VietOCR fine-tuned – dùng trọng số pretrained gốc.")
                self.ocr_weights_used = cfg.get('weights')
            self.ocr_predictor = Predictor(cfg)
            self.ocr_engine_type = "vietocr"
            print(f"✅ Đã tải VietOCR ({ocr_model_name}) trên {device} | weights: {self.ocr_weights_used}")
        except Exception as e:
            print(f"⚠️ Không thể khởi tạo VietOCR ({e}), thử fallback sang EasyOCR...")
            try:
                import easyocr
                self.ocr_predictor = easyocr.Reader(['vi', 'en'], gpu=(device != "cpu"))
                self.ocr_engine_type = "easyocr"
                print("✅ Đã khởi tạo EasyOCR fallback!")
            except Exception as e_easy:
                print(f"⚠️ Không thể khởi tạo EasyOCR ({e_easy}). OCR sẽ tạm thời disabled.")
                self.ocr_engine_type = None

    def _ocr_crop(self, crop_pil):
        """Runs OCR on a single PIL image crop. Returns (text, confidence or None)."""
        if crop_pil is None or crop_pil.size[0] < 4 or crop_pil.size[1] < 4:
            return "", None

        if self.ocr_engine_type == "vietocr":
            try:
                text, prob = self.ocr_predictor.predict(crop_pil, return_prob=True)
                return text.strip(), float(prob)
            except Exception:
                return "", None
        elif self.ocr_engine_type == "easyocr":
            try:
                crop_np = np.array(crop_pil)
                results = self.ocr_predictor.readtext(crop_np, detail=1)
                if not results:
                    return "", None
                text = " ".join(r[1] for r in results).strip()
                conf = float(np.mean([r[2] for r in results]))
                return text, conf
            except Exception:
                return "", None
        return "", None

    def _ocr_batch(self, crops_pil, batch_size=32):
        """Runs OCR on a list of PIL crops using batch inference when supported."""
        if not crops_pil:
            return []
        if self.ocr_engine_type == "vietocr" and hasattr(self.ocr_predictor, "predict_batch"):
            valid_indices = [
                i for i, img in enumerate(crops_pil)
                if img is not None and img.size[0] >= 4 and img.size[1] >= 4
            ]
            out = [("", None) for _ in crops_pil]
            if not valid_indices:
                return out
            try:
                for start in range(0, len(valid_indices), batch_size):
                    idx_chunk = valid_indices[start:start + batch_size]
                    imgs_chunk = [crops_pil[i] for i in idx_chunk]
                    texts, probs = self.ocr_predictor.predict_batch(imgs_chunk, return_prob=True)
                    for idx, txt, pr in zip(idx_chunk, texts, probs):
                        out[idx] = (str(txt).strip(), float(pr) if pr is not None else None)
                return out
            except Exception:
                pass
        return [self._ocr_crop(img) for img in crops_pil]

    @staticmethod
    def _group_table_rows(table_records):
        """
        Gom các box bảng (mã môn, tên môn, TC, điểm) thành từng dòng theo tâm trục y,
        sau đó sắp xếp trái → phải. Tránh lỗi box cùng dòng có y0 lệch nhau làm đảo thứ tự,
        và giữ đủ 3 cột điểm (hệ 10, hệ 4, chữ) thay vì ghi đè.
        """
        if not table_records:
            return []
        heights = [r["box"][3] - r["box"][1] for r in table_records]
        tol = max(4.0, 0.6 * float(np.median(heights)))

        items = sorted(table_records, key=lambda r: (r["box"][1] + r["box"][3]) / 2.0)
        rows, current, current_yc = [], [], None
        for r in items:
            yc = (r["box"][1] + r["box"][3]) / 2.0
            if current and abs(yc - current_yc) > tol:
                rows.append(current)
                current = []
            current.append(r)
            current_yc = float(np.mean([(c["box"][1] + c["box"][3]) / 2.0 for c in current]))
        if current:
            rows.append(current)

        raw_rows = []
        for row in rows:
            row.sort(key=lambda r: r["box"][0])
            entry, scores, confs = {}, [], []
            for r in row:
                if r["ocr_confidence"] is not None:
                    confs.append(r["ocr_confidence"])
                if r["class_name"] == "score":
                    if r["text"]:
                        scores.append(r["text"])
                elif r["class_name"] not in entry:
                    entry[r["class_name"]] = r["text"]
                else:
                    entry[r["class_name"]] += " " + r["text"]
            if scores:
                if len(scores) == 1:
                    s_clean = clean_academic_score(scores[0])
                    if s_clean:
                        entry["scores"] = [s_clean]
                        entry["score"] = s_clean
                else:
                    entry["scores"] = scores
                    entry["score"] = " | ".join(scores)
            if not entry:
                continue
            r_box = [
                min(r["box"][0] for r in row), min(r["box"][1] for r in row),
                max(r["box"][2] for r in row), max(r["box"][3] for r in row),
            ]
            raw_rows.append((entry, r_box, confs))

        # Gộp dòng tên môn học bị xuống dòng (chỉ có course_name, không có mã/tín chỉ/điểm) vào dòng liền trên
        merged_rows = []
        for entry, r_box, confs in raw_rows:
            has_only_name = (
                bool(entry.get("course_name"))
                and not entry.get("course_code")
                and not entry.get("credits")
                and not entry.get("scores")
            )
            if has_only_name and merged_rows:
                prev_entry, prev_box, prev_confs = merged_rows[-1]
                y_gap = r_box[1] - prev_box[3]
                if y_gap <= 45 and prev_entry.get("course_name"):
                    prev_entry["course_name"] = f"{prev_entry['course_name']} {entry['course_name']}".strip()
                    prev_box[0] = min(prev_box[0], r_box[0])
                    prev_box[2] = max(prev_box[2], r_box[2])
                    prev_box[3] = max(prev_box[3], r_box[3])
                    prev_confs.extend(confs)
                    continue
            merged_rows.append((entry, r_box, confs))

        table = []
        for entry, r_box, confs in merged_rows:
            raw_code = str(entry.get("course_code", "") or "").strip()
            raw_name = str(entry.get("course_name", "") or "").strip()
            # Loại bỏ các dòng rác chỉ bắt nhầm điểm/số ở đoạn văn (không có cả mã môn lẫn tên môn)
            if not raw_code and not raw_name:
                continue

            # Grounding cho các bảng không có khung kẻ lưới (vd T08 Đơn xin học môn thay thế)
            can_code, can_name, can_cred, sim = ground_course(raw_code, raw_name)
            if can_code and (sim >= 0.72 or re.match(r"^[A-Z]{2,5}\d{5,6}$", can_code)):
                entry["course_code"] = can_code
            if can_name:
                entry["course_name"] = can_name
            if not entry.get("credits") and can_cred:
                entry["credits"] = str(can_cred)

            entry["row_box"] = r_box
            if confs:
                entry["min_ocr_confidence"] = round(min(confs), 3)
            table.append(entry)
        return table

    def process_document(self, image_input, save_visualization_path=None, form_type=None, validate=True):
        """
        Processes a full academic document page and extracts structured data.
        Returns a dictionary containing entities, course table, verification flags,
        raw detections and (optionally) the rule-engine validation report.

        image_input: đường dẫn file ảnh (str) hoặc đối tượng PIL.Image.Image.
        form_type: mã loại biểu mẫu (vd 'T05_BAO_LUU') hoặc từ khóa; None = tự suy luận.
        """
        if isinstance(image_input, Image.Image):
            img_pil = image_input.convert("RGB")
            doc_name = getattr(image_input, "filename", "document.png")
            pass_to_yolo = img_pil
        else:
            if not os.path.exists(str(image_input)):
                raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_input}")
            img_pil = Image.open(image_input).convert("RGB")
            doc_name = os.path.basename(str(image_input))
            pass_to_yolo = str(image_input)

        w_img, h_img = img_pil.size

        structured_result = {
            "document_file": doc_name,
            "image_size": {"width": w_img, "height": h_img},
            "models": {
                "layout_engine": "ONNX Runtime (.onnx)" if str(getattr(self, "yolo_model_path_used", "")).endswith(".onnx") else "PyTorch Native (.pt)",
                "layout_weights": getattr(self, "yolo_model_path_used", DEFAULT_YOLO_WEIGHTS),
                "ocr_engine": self.ocr_engine_type,
                "ocr_weights": self.ocr_weights_used
            },
            "entities": {},
            "courses_table": [],
            "verification": {
                "has_signature": False,
                "signature_count": 0,
                "has_stamp": False,
                "stamp_count": 0
            },
            "raw_detections": []
        }

        # 1. Xử lý nhanh chuyên biệt cho Bảng điểm cổng thông tin đào tạo (T10_BANG_DIEM / ảnh dài portal)
        is_tall_portal = (h_img >= int(w_img * 1.75))
        portal_extracted = False
        if (form_type == "T10_BANG_DIEM" or is_tall_portal) and cv2 is not None:
            detected_header_form = self._detect_form_type_from_header(img_pil, structured_result)
            if self._extract_transcript_portal(img_pil, structured_result):
                form_type = "T10_BANG_DIEM"
                portal_extracted = True
            elif not form_type or form_type == "UNKNOWN":
                if detected_header_form != "UNKNOWN":
                    form_type = detected_header_form

        # 2. YOLOv11 Layout Detection & VietOCR từng vùng (bỏ qua nếu đã trích xuất sạch từ portal T10)
        if not portal_extracted and self.yolo_model is not None:
            self._run_detection_and_ocr(pass_to_yolo, img_pil, structured_result)

        # 3. Nhận diện sớm loại biểu mẫu từ tiêu đề (giúp chọn đúng chiến lược trích xuất bảng & chữ ký)
        if not form_type or form_type == "UNKNOWN":
            detected_header_form = self._detect_form_type_from_header(img_pil, structured_result)
            if detected_header_form != "UNKNOWN":
                form_type = detected_header_form
            elif self.rule_engine is not None:
                fn_hit = self.rule_engine._match_keywords(doc_name)
                if fn_hit:
                    form_type = fn_hit

        # 4. Dự phòng chế độ portal nếu nhận diện từ tiêu đề là T10 nhưng ảnh không vượt ngưỡng tỷ lệ dọc
        if not portal_extracted and form_type == "T10_BANG_DIEM" and cv2 is not None:
            if self._extract_transcript_portal(img_pil, structured_result):
                portal_extracted = True

        # Các biểu mẫu không có bảng học phần (tránh số liệu đoạn văn bị bắt nhầm thành dòng môn học)
        NON_TABLE_FORMS = {"T02_HOC_2CTDT", "T03_CHUYEN_HE", "T06_BAN_SAO_SO_GOC"}
        if form_type in NON_TABLE_FORMS:
            structured_result["courses_table"] = []
            structured_result["raw_detections"] = [
                d for d in structured_result.get("raw_detections", [])
                if d.get("class_name") not in TABLE_CLASSES
            ]

        # 5. Xác định vị trí bảng & trích xuất bổ sung (chỉ chạy OCR lưới bảng khi YOLO chưa bắt đủ dòng/điểm)
        table_bottom_y = None
        if form_type != "T10_BANG_DIEM" and form_type not in NON_TABLE_FORMS and cv2 is not None:
            yolo_courses = structured_result.get("courses_table", [])
            valid_yolo = [
                c for c in yolo_courses
                if re.match(r"^[A-Z]{2,5}\d{5,6}$", str(c.get("course_code", "")))
                or (len(str(c.get("course_name", "")).strip()) >= 5 and str(c.get("score", "")).strip())
            ]
            # Nếu YOLO đã bắt sạch toàn bộ các dòng học phần (đủ mã môn chuẩn), chỉ lấy tọa độ đáy bảng để quét chữ ký
            need_grid_rows = (len(valid_yolo) == 0)
            if not need_grid_rows and form_type in ("T01_DIEM_I", "T09_MIEN_THI"):
                has_codes = sum(1 for c in valid_yolo if re.match(r"^[A-Z]{2,5}\d{5,6}$", str(c.get("course_code", ""))))
                if has_codes == 0:
                    need_grid_rows = True
            if not need_grid_rows and form_type == "T07_CHUYEN_DIEM_NN":
                if any(not str(c.get("score", "")).strip() for c in valid_yolo):
                    need_grid_rows = True
            table_bottom_y = self._extract_table_grid_fallback(
                img_pil, structured_result, form_type=form_type, detect_rows=need_grid_rows
            )

        # 6. Bổ sung trường đặc thù còn thiếu trên T03 Chuyển hệ và định danh phần đầu đơn (T01/T09/camera)
        if form_type == "T03_CHUYEN_HE" and not structured_result["entities"].get("reason"):
            self._extract_t03_reason_fallback(img_pil, structured_result)
        if form_type != "T10_BANG_DIEM":
            ents_cur = structured_result["entities"]
            if not ents_cur.get("student_name") or not ents_cur.get("student_id"):
                self._extract_identity_upper_fallback(img_pil, structured_result)

        # 7. Trích xuất ngày lập đơn chính thức ở chân trang (tránh nhầm với ngày thi / ngày cấp bằng ở thân đơn)
        if form_type != "T10_BANG_DIEM":
            self._extract_document_date_footer(img_pil, structured_result)

        # 8. Quét nhận diện chữ ký phù hợp theo từng loại biểu mẫu (bỏ qua trên T10 Bảng điểm portal)
        if cv2 is not None:
            self._extract_signatures_footer(img_pil, structured_result, table_bottom_y=table_bottom_y, form_type=form_type)

        # 9. Tự động suy luận lại loại biểu mẫu nếu bước đầu chưa xác định được
        if not form_type or form_type == "UNKNOWN":
            form_type = self._detect_form_type_from_header(img_pil, structured_result)

        # 10. Rule Engine: kiểm tra tính hợp lệ nghiệp vụ
        if validate and self.rule_engine is not None:
            structured_result["validation"] = self.rule_engine.validate(structured_result, form_type=form_type)

        if save_visualization_path:
            self._save_visualization(img_pil, structured_result, save_visualization_path)

        return structured_result

    def _run_detection_and_ocr(self, yolo_input, img_pil, structured_result):
        w_img, h_img = img_pil.size
        # Dùng threshold linh hoạt: quét từ conf 0.20 để gom candidate chữ ký và SĐT trên ảnh camera
        scan_conf = min(0.20, self.conf_threshold)
        yolo_results = self.yolo_model(yolo_input, conf=scan_conf, imgsz=self.imgsz, verbose=False)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        classes = yolo_results.boxes.cls.cpu().numpy().astype(int)
        confidences = yolo_results.boxes.conf.cpu().numpy()

        # Sort detections top-to-bottom for reading order
        detections = []
        for box, cls_idx, score in zip(boxes, classes, confidences):
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
            detections.append({"box": box, "class_name": cls_name, "score": float(score)})
        detections.sort(key=lambda d: (d["box"][1], d["box"][0]))

        table_records = []
        ver = structured_result["verification"]
        pending_ocr = []

        for d in detections:
            cls_name, score = d["class_name"], d["score"]
            x0, y0, x1, y1 = map(int, d["box"])

            # Lọc ngưỡng tin cậy theo từng loại đối tượng (chữ ký cho phép threshold thấp hơn để bắt đủ nét bút)
            if cls_name != "signature" and score < self.conf_threshold:
                continue

            # Check Stamps & Signatures (không cần OCR)
            if cls_name in ("stamp", "signature"):
                if cls_name == "stamp":
                    # Kiểm tra màu đỏ trong crop để loại bỏ 100% false positive trên khối text đen trắng
                    if score < 0.30 or not self._is_valid_stamp(img_pil, (x0, y0, x1, y1)):
                        continue
                elif cls_name == "signature":
                    # Kiểm tra hình dạng, vị trí chân trang và độ tin cậy của chữ ký
                    if not self._is_valid_signature((x0, y0, x1, y1), score, h_img=h_img):
                        continue

                key = "stamp" if cls_name == "stamp" else "signature"
                ver[f"has_{key}"] = True
                ver[f"{key}_count"] += 1
                structured_result["raw_detections"].append({
                    "class_name": cls_name,
                    "confidence": round(score, 3),
                    "box": [x0, y0, x1, y1],
                    "text": "[MỘC DẤU ĐỎ]" if key == "stamp" else "[CHỮ KÝ]"
                })
                continue

            # Text Entities: Crop with padding và đưa vào hàng đợi OCR batch
            # Với số điện thoại, mở rộng biên trái (+25px) để không bị xén mất tiền tố 08/09
            pad_l = 25 if cls_name == "phone" else 3
            pad_r = 5 if cls_name == "phone" else 3
            x0_eff = max(0, x0 - pad_l)
            crop = img_pil.crop((x0_eff, max(0, y0 - 3),
                                 min(w_img, x1 + pad_r), min(h_img, y1 + 3)))
            pending_ocr.append((cls_name, score, x0, y0, x1, y1, x0_eff, crop))

        batch_ocr_results = self._ocr_batch([item[7] for item in pending_ocr])

        for (cls_name, score, x0, y0, x1, y1, x0_eff, _), (rec_text, ocr_conf) in zip(pending_ocr, batch_ocr_results):
            d_record = {
                "class_name": cls_name,
                "confidence": round(score, 3),
                "ocr_confidence": round(ocr_conf, 3) if ocr_conf is not None else None,
                "box": [x0_eff if cls_name == "phone" else x0, y0, x1, y1],
                "text": rec_text
            }
            structured_result["raw_detections"].append(d_record)

            # Map to semantic fields
            if cls_name in ENTITY_CLASSES and rec_text:
                if cls_name == "student_name":
                    rec_text = re.sub(r"^(?:Họ và tên sinh viên|Họ và tên|Họ tên|Tôi tên|Em tên|Tên|Lên|sinh viên|Linu)\s*[:.-]?\s*", "", rec_text, flags=re.IGNORECASE)
                    rec_text = re.split(r"(?:Mã\s*SV|MSSV|SĐT|Điện thoại|Sinh ngày|Lớp)", rec_text, flags=re.IGNORECASE)[0].strip()
                    rec_text = re.sub(r"\s*(?:Liệu|Đội|Điện thoại|MSSV)$", "", rec_text, flags=re.IGNORECASE).strip()
                    rec_text = re.sub(r"[\s.,;:()/\d|]+$", "", rec_text).strip()
                    rec_text = re.sub(r"\s+[Il]$", "", rec_text).strip()
                    # Khử hiện tượng mực mặt sau giấy (bleed-through) hoặc rác chữ không phải tên người
                    low_name = rec_text.lower()
                    if any(bt in low_name for bt in ["chủ nghĩa xã hội", "triết học", "kinh tế chính trị", "tư tưởng hồ chí minh", "điệm trị", "thuật cấp"]):
                        rec_text = ""
                elif cls_name == "student_id":
                    rec_text = re.sub(r"^(?:Mã\s*SV|MSSV|MSV|Mã số sinh viên)\s*[:.-]?\s*", "", rec_text, flags=re.IGNORECASE).strip()
                    digits_only = re.sub(r"\D", "", rec_text)
                    # Loại trừ số điện thoại 10 chữ số (bắt đầu bằng 0) bị YOLO gán nhầm nhãn student_id
                    if re.match(r"^0\d{9}$", digits_only):
                        if "phone" not in structured_result["entities"]:
                            structured_result["entities"]["phone"] = digits_only
                        rec_text = ""
                    else:
                        # MSSV HCMUTE chuẩn gồm 8 chữ số (bắt đầu bằng 1x hoặc 2x)
                        m_mssv = re.search(r"\b(2\d{7}|1\d{7})\b", rec_text)
                        if not m_mssv:
                            m_mssv = re.search(r"\b(\d{8})\b", rec_text)
                        if m_mssv and not m_mssv.group(1).startswith("0"):
                            rec_text = m_mssv.group(1)
                        elif re.match(r"^[12]\d{7}$", digits_only):
                            rec_text = digits_only
                        else:
                            rec_text = re.split(r"[^\dA-Za-z]", rec_text)[0].strip()
                            if re.match(r"^0\d{7,9}$", rec_text):
                                rec_text = ""
                elif cls_name == "phone":
                    rec_text = re.sub(r"^(?:Điện thoại|SĐT|Phone|Tel)\s*[:.-]?\s*", "", rec_text, flags=re.IGNORECASE).strip()
                    m_p = re.search(r"0\d{9,10}", rec_text)
                    if m_p:
                        rec_text = m_p.group(0)
                    else:
                        digits = re.sub(r"\D", "", rec_text)
                        if len(digits) == 9 and not digits.startswith("0"):
                            rec_text = "0" + digits
                        elif len(digits) == 10 and digits.startswith("0"):
                            rec_text = digits
                        elif digits:
                            rec_text = digits
                elif cls_name == "class_name":
                    rec_text = re.sub(r"^(Lớp|Class)\s*[:.-]?\s*", "", rec_text, flags=re.IGNORECASE).strip()
                elif cls_name == "faculty":
                    rec_text = re.sub(r"^(Khoa|Khoa\s*\(TT\))\s*[:.-]?\s*", "", rec_text, flags=re.IGNORECASE).strip()
                elif cls_name == "email":
                    # Chuẩn hóa OCR email: OCR thường nhận nhầm @ thành 0 (vd 251455350student... -> @student)
                    rec_text = re.sub(r"(\d+)0(student\.hcmute\.edu\.vn)", r"\1@\2", rec_text)
                    rec_text = re.sub(r"^(?:Email|E-mail|Mail)\s*[:.-]?\s*", "", rec_text, flags=re.IGNORECASE).strip()
                    m_em = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", rec_text)
                    if m_em:
                        rec_text = m_em.group(0)
                    elif "hcmute.edu.vn" in rec_text:
                        rec_text = rec_text.split()[-1]
                    else:
                        rec_text = ""
                elif cls_name == "major":
                    rec_text = re.sub(r"^(?:Ngành|Chuyên ngành|Major)\s*[:.-]?\s*", "", rec_text, flags=re.IGNORECASE).strip()
                    # Khử các từ rác in ở chân trang bị nhận nhầm thành ngành học
                    if any(bad in rec_text.lower() for bad in ["conform", "quyết định", "đào tạo", "yêu cầu"]):
                        rec_text = ""
                elif cls_name == "date":
                    # Chỉ chấp nhận chuỗi chứa ngày tháng năm hợp lệ (năm từ 1990 đến 2030)
                    m_d = re.search(r"(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})", rec_text)
                    if m_d:
                        fd, fm, fy = m_d.groups()
                        if 1990 <= int(fy) <= 2030 and 1 <= int(fm) <= 12 and 1 <= int(fd) <= 31:
                            rec_text = f"{int(fd):02d}/{int(fm):02d}/{fy}"
                        else:
                            rec_text = ""
                    else:
                        m_d = re.search(r"(?:ngày|ngay|ngly|ngáy)\s*(\d{1,2})\s*(?:tháng|thang)\s*(\d{1,2})\s*(?:năm|nam)\s*(\d{4})", rec_text, re.IGNORECASE)
                        if m_d:
                            fd, fm, fy = m_d.groups()
                            if 1990 <= int(fy) <= 2030 and 1 <= int(fm) <= 12 and 1 <= int(fd) <= 31:
                                rec_text = f"{int(fd):02d}/{int(fm):02d}/{fy}"
                            else:
                                rec_text = ""
                        else:
                            rec_text = ""

                if not rec_text:
                    continue

                ents = structured_result["entities"]
                # Các trường đơn trị (Scalar Entity Fields): Ưu tiên giữ giá trị chuẩn nhất, không gom thành list
                SCALAR_FIELDS = {"student_name", "student_id", "class_name", "faculty", "major", "dob", "pob", "phone", "email", "date"}

                if cls_name == "student_id" and "student_id" in ents:
                    # Ưu tiên giữ MSSV chuẩn [12]xxxxxxx làm student_id chính nếu giá trị cũ chưa chuẩn
                    cur_id = str(ents["student_id"])
                    if not re.match(r"^[12]\d{7}$", cur_id) and re.match(r"^[12]\d{7}$", rec_text):
                        ents["student_id"] = rec_text
                    elif cur_id != rec_text and "old_student_id" not in ents:
                        ents["old_student_id"] = rec_text
                elif cls_name == "class_name" and "class_name" in ents:
                    # Nếu phát hiện thêm lớp (thường là Lớp cũ trên đơn T05 Bảo lưu / T02 Chuyển điểm)
                    if ents["class_name"] != rec_text and "old_class_name" not in ents:
                        ents["old_class_name"] = rec_text
                elif cls_name in SCALAR_FIELDS:
                    if cls_name not in ents:
                        ents[cls_name] = rec_text
                    else:
                        # Với student_name và major: ưu tiên box nằm ở nửa trên văn bản (Identity header y < 0.45 * H)
                        if cls_name in ("student_name", "major") and y0 < int(h_img * 0.45):
                            ents[cls_name] = rec_text
                        elif cls_name == "email" and "@" in rec_text and "@" not in ents[cls_name]:
                            ents[cls_name] = rec_text
                        elif cls_name == "date" and y0 > int(h_img * 0.65):
                            ents[cls_name] = rec_text
                else:
                    if cls_name not in ents:
                        ents[cls_name] = rec_text
                    elif isinstance(ents[cls_name], list):
                        if rec_text not in ents[cls_name]:
                            ents[cls_name].append(rec_text)
                    else:
                        if rec_text != ents[cls_name]:
                            ents[cls_name] = [ents[cls_name], rec_text]

            # Course Table fields
            if cls_name in TABLE_CLASSES:
                table_records.append(d_record)

        structured_result["courses_table"] = self._group_table_rows(table_records)

    def _is_valid_stamp(self, img_pil, box, min_red_ratio=0.02):
        """
        Kiểm tra mộc đỏ hành chính thật thông qua dải màu đỏ/đỏ cánh sen trong không gian HSV.
        Loại bỏ 100% các bounding box nhận nhầm trên các khối văn bản đen trắng ở chân trang.
        """
        if cv2 is None:
            return True
        try:
            x0, y0, x1, y1 = map(int, box)
            w_img, h_img = img_pil.size
            crop = img_pil.crop((max(0, x0), max(0, y0), min(w_img, x1), min(h_img, y1)))
            crop_np = np.array(crop)
            if crop_np.shape[0] < 10 or crop_np.shape[1] < 10:
                return False
            hsv = cv2.cvtColor(crop_np, cv2.COLOR_RGB2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([12, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([165, 40, 40]), np.array([180, 255, 255]))
            red_mask = mask1 | mask2
            red_ratio = np.sum(red_mask > 0) / float(red_mask.shape[0] * red_mask.shape[1])
            return red_ratio >= min_red_ratio
        except Exception:
            return True

    def _is_valid_signature(self, box, score, min_conf=0.25, h_img=None):
        """
        Kiểm tra tính hợp lệ của chữ ký:
        - Tỷ lệ khung hình và độ tin cậy tối thiểu.
        - Bỏ qua phần tiêu đề trang (y < 0.28 * H) để tránh false positive trên quốc hiệu/tiêu đề,
          nhưng chấp nhận vùng y >= 0.30 * H vì các mẫu đơn ngắn (T02, T03, T06, T07, T08, T09)
          có vùng ký tên nằm từ 34% đến 65% chiều cao trang.
        """
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        if bh <= 0 or bw <= 0:
            return False
        ar = bw / float(bh)
        if ar < 0.5 or ar > 6.5:
            return False
        if h_img is not None and y0 < int(h_img * 0.28):
            return False
        return score >= min_conf

    def _segment_text_phrases(self, crop_pil, gap_w=16, min_w=14, min_h=8, max_h=90, remove_lines=True, dark_thresh=None):
        """
        Tách các cụm chữ (word/phrase boxes) theo chiều ngang để tránh gửi dải ảnh quá rộng
        nhiều khoảng trắng khiến VietOCR ảo giác sinh chữ lặp.
        Trả về danh sách (x0, y0, x1, y1, text, conf).
        """
        if cv2 is None:
            return []
        w_c, h_c = crop_pil.size
        gray = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2GRAY)
        if dark_thresh is not None:
            bw = (gray < int(dark_thresh)).astype(np.uint8) * 255
        else:
            # Ngưỡng Gaussian (25, -14) loại bỏ nhiễu nền giấy camera/scan xám mà vẫn giữ nét chữ mảnh;
            # kết hợp ngưỡng mực đậm (< 85) để không làm rỗng ruột các tiêu đề in đậm cỡ lớn.
            bw_gauss = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, -14)
            bw_dark = (gray < 85).astype(np.uint8) * 255
            bw = cv2.bitwise_or(bw_gauss, bw_dark)
        if remove_lines:
            hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w_c // 8), 1))
            vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(24, h_c // 6)))
            bw = cv2.subtract(bw, cv2.dilate(cv2.erode(bw, hk), hk))
            bw = cv2.subtract(bw, cv2.dilate(cv2.erode(bw, vk), vk))

        dk = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_w, 2))
        dil = cv2.dilate(bw, dk)
        cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands, sub_crops = [], []
        for c in cnts:
            x, y, bw_c, bh_c = cv2.boundingRect(c)
            if bw_c >= min_w and min_h <= bh_c <= max_h:
                if int(np.sum(bw[y:y + bh_c, x:x + bw_c] > 0)) < 10:
                    continue
                sub = crop_pil.crop((max(0, x - 3), max(0, y - 2), min(w_c, x + bw_c + 3), min(h_c, y + bh_c + 2)))
                cands.append((x, y, bw_c, bh_c))
                sub_crops.append(sub)

        ocr_outs = self._ocr_batch(sub_crops) if sub_crops else []
        segs = []
        for (x, y, bw_c, bh_c), (txt, conf) in zip(cands, ocr_outs):
            if txt and not _is_ocr_hallucination(txt):
                segs.append((x, y, x + bw_c, y + bh_c, txt, float(conf) if conf is not None else 0.8))
        segs.sort(key=lambda s: (round(s[1] / 14.0), s[0]))
        return segs

    def _extract_document_date_footer(self, img_pil, structured_result):
        """
        Trích xuất ngày lập đơn chính thức ở phần ký tên ('Tp. Hồ Chí Minh, ngày ... tháng ... năm ...').
        Ưu tiên thay thế nếu trước đó YOLO nhầm ngày cấp bằng / ngày cấp chứng chỉ ở phần thân trên.
        """
        w_img, h_img = img_pil.size
        existing_date = structured_result["entities"].get("date")
        date_pat = r"(?:ngày|ngay|ngly|ngáy)\s*(\d{1,2})\s*(?:tháng|thang)\s*(\d{1,2})\s*(?:năm|nam)\s*(\d{4})"

        def _parse_date_str(txt):
            s = str(txt or "").strip()
            m1 = re.search(r"(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})", s)
            if m1:
                fd, fm, fy = m1.groups()
                if 1990 <= int(fy) <= 2030 and 1 <= int(fm) <= 12 and 1 <= int(fd) <= 31:
                    return f"{int(fd):02d}/{int(fm):02d}/{fy}"
            m2 = re.search(date_pat, s, re.IGNORECASE)
            if m2:
                fd, fm, fy = m2.groups()
                if 1990 <= int(fy) <= 2030 and 1 <= int(fm) <= 12 and 1 <= int(fd) <= 31:
                    return f"{int(fd):02d}/{int(fm):02d}/{fy}"
            return None

        # Kiểm tra xem date hiện tại có đến từ vùng thân dưới (y >= 0.40 * H) không
        has_footer_date = False
        parsed_existing = _parse_date_str(existing_date)
        if parsed_existing:
            for d in structured_result.get("raw_detections", []):
                if d.get("class_name") == "date" and d["box"][1] >= int(h_img * 0.40):
                    p_det = _parse_date_str(d.get("text"))
                    if p_det:
                        structured_result["entities"]["date"] = p_det
                        has_footer_date = True
                        break
        if has_footer_date:
            return

        if cv2 is not None:
            # Quét các cụm chữ ở vùng lập đơn (38% - 88% chiều cao, nửa phải trang)
            ry0, ry1 = int(h_img * 0.36), int(h_img * 0.90)
            rx0, rx1 = int(w_img * 0.30), int(w_img * 0.98)
            sub_crop = img_pil.crop((rx0, ry0, rx1, ry1))
            segs = self._segment_text_phrases(sub_crop, gap_w=24, min_w=80, min_h=10, max_h=65)
            # Gom các cụm gần cùng dòng y
            lines_by_y = {}
            for x0, y0, x1, y1, txt, _ in segs:
                yk = round((y0 + y1) / 2.0 / 18.0)
                lines_by_y.setdefault(yk, []).append((x0, y0, x1, y1, txt))
            for yk in sorted(lines_by_y.keys()):
                row_items = sorted(lines_by_y[yk], key=lambda t: t[0])
                full_line = " ".join(t[4] for t in row_items)
                m_fdate = re.search(
                    r"(?:ngày|ngay|ngly|ngáy)\s*(\d{1,2})\s*(?:tháng|thang)\s*(\d{1,2})\s*(?:năm|nam)\s*(\d{4})",
                    full_line, re.IGNORECASE,
                )
                if m_fdate:
                    fd, fm, fy = m_fdate.groups()
                    if 1990 <= int(fy) <= 2030 and 1 <= int(fm) <= 12 and 1 <= int(fd) <= 31:
                        date_str = f"{int(fd):02d}/{int(fm):02d}/{fy}"
                        structured_result["entities"]["date"] = date_str
                        fx0_abs = rx0 + min(t[0] for t in row_items)
                        fy0_abs = ry0 + min(t[1] for t in row_items)
                        fx1_abs = rx0 + max(t[2] for t in row_items)
                        fy1_abs = ry0 + max(t[3] for t in row_items)
                        structured_result["raw_detections"].append({
                            "class_name": "date",
                            "confidence": 0.90,
                            "box": [fx0_abs, fy0_abs, fx1_abs, fy1_abs],
                            "text": date_str
                        })
                        return

        # Fallback quét dải hẹp 70% - 76% nếu chưa bắt được
        fx0, fx1 = int(w_img * 0.45), int(w_img * 0.98)
        for y_pct in np.arange(0.70, 0.76, 0.012):
            fy0 = int(h_img * y_pct)
            fy1 = int(h_img * min(0.85, y_pct + 0.025))
            slice_crop = img_pil.crop((fx0, fy0, fx1, fy1))
            f_text, _ = self._ocr_crop(slice_crop)
            m_fdate = re.search(r"(?:ngày|ngay|ngly|ngáy)\s*(\d{1,2})\s*(?:tháng|thang)\s*(\d{1,2})\s*(?:năm|nam)\s*(\d{4})", f_text, re.IGNORECASE)
            if m_fdate:
                fd, fm, fy = m_fdate.groups()
                if 1990 <= int(fy) <= 2030 and 1 <= int(fm) <= 12 and 1 <= int(fd) <= 31:
                    date_str = f"{int(fd):02d}/{int(fm):02d}/{fy}"
                    structured_result["entities"]["date"] = date_str
                    structured_result["raw_detections"].append({
                        "class_name": "date", "confidence": 0.90,
                        "box": [fx0, fy0, fx1, fy1], "text": date_str
                    })
                    break

    def _extract_signatures_footer(self, img_pil, structured_result, table_bottom_y=None, form_type=None):
        """
        Quét các cụm chữ ký viết tay bằng phân tích nét mực viết tay (Handwritten Ink Blobs).
        Tự động điều chỉnh vùng quét theo loại biểu mẫu (vì T02/T03/T06/T07/T08/T09 có vùng ký ở 33%-68% trang,
        còn T01/T04/T05 có vùng ký ở chân trang >= 68%, riêng T10 Bảng điểm portal không có chữ ký tay).
        """
        # Bảng điểm cổng thông tin đào tạo (T10) không yêu cầu/không có chữ ký tay
        if form_type == "T10_BANG_DIEM":
            structured_result["raw_detections"] = [
                d for d in structured_result.get("raw_detections", [])
                if d.get("class_name") != "signature"
            ]
            structured_result["verification"]["signature_count"] = 0
            structured_result["verification"]["has_signature"] = False
            return

        if cv2 is None:
            return
        w_img, h_img = img_pil.size

        # Loại bỏ chữ ký giả nằm đè lên các ô bảng / trường văn bản đã nhận diện
        non_sig_boxes = [
            d["box"] for d in structured_result.get("raw_detections", [])
            if d.get("class_name") not in ("signature", "stamp")
        ]

        def _overlaps_text_box(b):
            bx0, by0, bx1, by1 = b
            cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
            for tb in non_sig_boxes:
                if tb[0] - 4 <= cx <= tb[2] + 4 and tb[1] - 4 <= cy <= tb[3] + 4:
                    return True
            return False

        structured_result["raw_detections"] = [
            d for d in structured_result.get("raw_detections", [])
            if d.get("class_name") != "signature" or not _overlaps_text_box(d["box"])
        ]

        high_sig_forms = {"T02_HOC_2CTDT", "T03_CHUYEN_HE", "T06_BAN_SAO_SO_GOC", "T07_CHUYEN_DIEM_NN", "T08_THAY_THE_MON", "T09_MIEN_THI"}
        if form_type in high_sig_forms:
            base_y = int(h_img * 0.33)
            if table_bottom_y is not None:
                start_y = max(base_y, table_bottom_y + int(h_img * 0.02))
            else:
                start_y = base_y
        else:
            start_y = max(int(h_img * 0.68), (table_bottom_y + int(h_img * 0.06)) if table_bottom_y else int(h_img * 0.68))

        end_y = int(h_img * 0.94)
        if start_y >= end_y - 30:
            sig_count = len([d for d in structured_result["raw_detections"] if d["class_name"] == "signature"])
            structured_result["verification"]["signature_count"] = sig_count
            structured_result["verification"]["has_signature"] = (sig_count > 0)
            return

        margin_x = max(20, int(w_img * 0.04))
        footer_np = np.array(img_pil)[start_y:end_y, margin_x:w_img - margin_x]
        gray = cv2.cvtColor(footer_np, cv2.COLOR_RGB2GRAY)
        bw = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

        # Loại bỏ các đường kẻ ngang dài (dòng phân cách / dòng chấm)
        h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_k)
        bw_clean = cv2.subtract(bw, lines)

        # Phát hiện thêm mực xanh/tím của chữ ký bút bi (ngay cả khi đè lên chữ in mờ hoặc mộc)
        hsv = cv2.cvtColor(footer_np, cv2.COLOR_RGB2HSV)
        blue_ink = cv2.inRange(hsv, np.array([95, 45, 35]), np.array([140, 255, 235]))

        # Gom các nét bút thành cụm chữ ký theo kích thước tương đối theo độ phân giải ảnh
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 7))
        clustered = cv2.dilate(bw_clean | blue_ink, k)

        contours, _ = cv2.findContours(clustered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        existing_sig_boxes = [d["box"] for d in structured_result["raw_detections"] if d["class_name"] == "signature"]

        scale = w_img / 2480.0
        min_w_c, max_w_c = max(25, int(45 * scale)), max(180, int(340 * scale))
        min_h_c, max_h_c = max(14, int(22 * scale)), max(90, int(160 * scale))

        for c in contours:
            x, y, bw_c, bh_c = cv2.boundingRect(c)
            if min_w_c <= bw_c <= max_w_c and min_h_c <= bh_c <= max_h_c and 0.7 <= (bw_c / float(bh_c)) <= 5.5:
                sub_gray = gray[y:y + bh_c, x:x + bw_c]
                sub_blue = blue_ink[y:y + bh_c, x:x + bw_c]
                ink_ratio = float(np.mean(sub_gray < 155))
                blue_ratio = float(np.mean(sub_blue > 0))
                # Chữ ký thật: có mực xanh bút bi HOẶC dạng khối nét bút (khác dòng chữ in mảnh ngang)
                is_handwritten = (blue_ratio >= 0.03) or (0.06 <= ink_ratio <= 0.45 and bh_c >= max(18, int(28 * scale)))
                if is_handwritten:
                    abs_box = [x + margin_x, y + start_y, x + margin_x + bw_c, y + start_y + bh_c]
                    if _overlaps_text_box(abs_box):
                        continue
                    # Bỏ qua các nhãn tiêu đề ngắn in trên dòng chữ ký nếu không có mực xanh và nằm quá hẹp
                    if blue_ratio < 0.02 and bw_c > int(220 * scale) and bh_c < int(42 * scale):
                        continue
                    overlap = False
                    for eb in existing_sig_boxes:
                        if abs(abs_box[0] - eb[0]) < int(45 * scale) and abs(abs_box[1] - eb[1]) < int(40 * scale):
                            overlap = True
                            break
                    if not overlap:
                        existing_sig_boxes.append(abs_box)
                        structured_result["raw_detections"].append({
                            "class_name": "signature",
                            "confidence": 0.85,
                            "box": abs_box,
                            "text": "[CHỮ KÝ]"
                        })

        sig_count = len([d for d in structured_result["raw_detections"] if d["class_name"] == "signature"])
        structured_result["verification"]["signature_count"] = sig_count
        structured_result["verification"]["has_signature"] = (sig_count > 0)

    def _detect_form_type_from_header(self, img_pil, structured_result=None):
        """
        Quét OCR phân đoạn cụm chữ ở vùng tiêu đề (3% - 28% phía trên trang) để tự động nhận diện
        chính xác mã loại biểu mẫu hành chính (T01 - T10) và tên trường đào tạo,
        tránh bị ảnh hưởng bởi nội dung trường 'reason' ở phần thân đơn.
        """
        w, h = img_pil.size
        # Thử phát hiện vùng tiêu đề chuẩn (dùng crop rộng hơn chút cho ảnh camera có lề trên)
        y_top, y_bot = int(h * 0.02), min(int(h * 0.28), 950)
        header_crop = img_pil.crop((int(w * 0.04), y_top, int(w * 0.96), y_bot))

        extracted_lines = []
        if cv2 is not None:
            segs = self._segment_text_phrases(header_crop, gap_w=18, min_w=16, min_h=8, max_h=85)
            lines_by_y = {}
            for sx0, sy0, sx1, sy1, txt, _ in segs:
                yk = round((sy0 + sy1) / 2.0 / 16.0)
                lines_by_y.setdefault(yk, []).append((sx0, txt))
            for yk in sorted(lines_by_y.keys()):
                row_items = sorted(lines_by_y[yk], key=lambda t: t[0])
                extracted_lines.append(" ".join(t[1] for t in row_items))

            # Hỗ trợ đọc thêm banner chữ trắng trên nền xanh đậm (cổng thông tin bảng điểm)
            top_banner = img_pil.crop((0, 0, w, min(int(h * 0.08), 110)))
            tb_np = np.array(top_banner)
            if tb_np.shape[0] >= 12:
                mean_bgr_diff = float(np.mean(tb_np[:, :, 2].astype(np.int16) - tb_np[:, :, 0].astype(np.int16)))
                if mean_bgr_diff > 15.0 or float(np.mean(tb_np)) < 130.0:
                    gray_tb = cv2.cvtColor(tb_np, cv2.COLOR_RGB2GRAY)
                    _, white_mask = cv2.threshold(gray_tb, 175, 255, cv2.THRESH_BINARY)
                    dil_w = cv2.dilate(white_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (16, 3)))
                    cnts_w, _ = cv2.findContours(dil_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cw in cnts_w:
                        bx, by, bw_c, bh_c = cv2.boundingRect(cw)
                        if bw_c >= 60 and 8 <= bh_c <= 60:
                            sub_inv = Image.fromarray(255 - gray_tb[max(0, by - 2):by + bh_c + 2, max(0, bx - 3):bx + bw_c + 3]).convert("RGB")
                            w_txt, _ = self._ocr_crop(sub_inv)
                            if w_txt:
                                extracted_lines.insert(0, w_txt)

        def _plain(s):
            norm = unicodedata.normalize("NFD", str(s)).lower()
            return "".join(c for c in norm if unicodedata.category(c) != "Mn").replace("đ", "d").replace("Đ", "D")

        title_text = " ".join(extracted_lines)
        title_plain = _plain(title_text)

        # Trích xuất nguyên văn tên cơ sở đào tạo chỉ từ các dòng thực sự chứa tên trường (không gộp tên SV)
        school_raw = None
        for l in extracted_lines:
            l_plain = _plain(l)
            if any(k in l_plain for k in ("su pham ky thuat", "cong nghe ky thuat", "spkt", "hcmute")):
                cleaned_s = re.sub(r"^(?:Kính gửi|Phòng Đào tạo)\s*[:.-]?\s*", "", l.strip(), flags=re.IGNORECASE).strip()
                # Cắt bỏ phần dư phía trước nếu có chứa "Trường Đại học..."
                m_univ = re.search(r"(Trường\s+Đại\s+học\s+.*)", cleaned_s, flags=re.IGNORECASE)
                school_raw = m_univ.group(1).strip() if m_univ else cleaned_s
                break
        if not school_raw:
            if "su pham ky thuat" in title_plain or "spkt" in title_plain:
                school_raw = "TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT TP. HỒ CHÍ MINH"
            elif "cong nghe ky thuat" in title_plain:
                school_raw = "TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT THÀNH PHỐ HỒ CHÍ MINH"

        if structured_result and school_raw:
            school_raw = re.sub(r"\bTP\.?\s*(?:HCIV|HCN|HCII|HCNN|HCI)\b", "TP.HCM", school_raw, flags=re.IGNORECASE)
            if "cong nghe ky thuat" in _plain(school_raw) and "tp.hcm" in _plain(school_raw):
                school_raw = "TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT TP.HCM"
            structured_result["entities"]["school_name"] = school_raw

        def _classify_from_plain(p_txt):
            # Ưu tiên cụm từ đặc trưng của tiêu đề từng mẫu (T05 phải đứng trước T03 vì phụ đề T05 có chữ 'chuyển hệ đào tạo')
            if "ket qua hoc tap" in p_txt or ("xem diem" in p_txt and "chuong trinh" in p_txt) or "bang diem" in p_txt:
                return "T10_BANG_DIEM"
            if "mon thay the" in p_txt or "hoc thay the" in p_txt or "thay the" in p_txt:
                return "T08_THAY_THE_MON"
            if "ngoai ngu" in p_txt and ("chuyen diem" in p_txt or "dau ra" in p_txt):
                return "T07_CHUYEN_DIEM_NN"
            if "bao luu" in p_txt or "bao l?u" in p_txt or ("chuyen truong" in p_txt and "nganh 2" in p_txt):
                return "T05_BAO_LUU"
            if "chuyen trinh do" in p_txt or (
                ("chuyen he" in p_txt or "trinh do dao tao" in p_txt)
                and "chuyen truong" not in p_txt
                and "nganh 2" not in p_txt
            ):
                return "T03_CHUYEN_HE"
            if "hai chuong trinh" in p_txt or "2 ctdt" in p_txt or "cung luc hai" in p_txt:
                return "T02_HOC_2CTDT"
            if (
                re.search(r"\bdiem\s+[i1]\b", p_txt)
                or "chua hoan tat" in p_txt
                or "nhan va tra diem" in p_txt
                or "nhan diem i" in p_txt
            ):
                return "T01_DIEM_I"
            if "chuyen diem" in p_txt or ("chuyen" in p_txt and "bang diem" not in p_txt and "chuyen mon" not in p_txt and ("mon hoc" in p_txt or "hoc phan" in p_txt)):
                return "T04_CHUYEN_DIEM"
            if "ban sao" in p_txt or "so goc" in p_txt:
                return "T06_BAN_SAO_SO_GOC"
            if "mien thi" in p_txt:
                return "T09_MIEN_THI"
            return "UNKNOWN"

        # Ưu tiên phân loại hoàn toàn từ vùng tiêu đề trên cùng trước khi nhìn xuống thân bài
        hit = _classify_from_plain(title_plain)
        if hit != "UNKNOWN":
            return hit

        # Trên ảnh camera bị dịch khung xuống dưới, mở rộng quét thêm dải 24%-36% chiều cao trang
        if cv2 is not None and h > 0:
            low_hdr_crop = img_pil.crop((int(w * 0.04), int(h * 0.24), int(w * 0.96), int(h * 0.36)))
            low_segs = self._segment_text_phrases(low_hdr_crop, gap_w=18, min_w=16, min_h=8, max_h=85)
            low_plain = _plain(" ".join(s[4] for s in low_segs))
            hit = _classify_from_plain(low_plain)
            if hit != "UNKNOWN":
                return hit

        # Fallback an toàn qua các detection khác (LOẠI TRỪ trường 'reason' để tránh từ khóa giả trong lý do làm lệch loại mẫu)
        if structured_result:
            non_reason_text = " ".join(
                d.get("text", "") for d in structured_result.get("raw_detections", [])
                if d.get("class_name") not in ("reason", "student_name")
            )
            hit = _classify_from_plain(_plain(non_reason_text))
            if hit != "UNKNOWN":
                return hit

            doc_file = str(structured_result.get("document_file", ""))
            if self.rule_engine is not None and doc_file:
                fn_hit = self.rule_engine._match_keywords(doc_file)
                if fn_hit:
                    return fn_hit

        return "UNKNOWN"

    def _extract_t03_reason_fallback(self, img_pil, structured_result):
        """
        Trên mẫu T03_CHUYEN_HE (Đơn xin xét chuyển trình độ đào tạo), mẫu không ghi nhãn 'Lý do:'
        mà ghi 'Bị buộc thôi học theo quyết định số ...' / 'Xin được xét chuyển sang hệ ...'.
        """
        if cv2 is None:
            return
        w_img, h_img = img_pil.size
        body_y0, body_y1 = int(h_img * 0.22), int(h_img * 0.56)
        body_x0, body_x1 = int(w_img * 0.06), int(w_img * 0.95)
        body_crop = img_pil.crop((body_x0, body_y0, body_x1, body_y1))
        segs = self._segment_text_phrases(body_crop, gap_w=20, min_w=40, min_h=10, max_h=65)
        if not segs:
            segs = self._segment_text_phrases(body_crop, gap_w=20, min_w=40, min_h=10, max_h=65, dark_thresh=150)
        lines_by_y = {}
        for sx0, sy0, sx1, sy1, txt, _ in segs:
            yk = round((sy0 + sy1) / 2.0 / 16.0)
            lines_by_y.setdefault(yk, []).append((sx0, sy0, sx1, sy1, txt))
        matched_lines = []
        first_box = None
        for yk in sorted(lines_by_y.keys()):
            items = sorted(lines_by_y[yk], key=lambda t: t[0])
            line_txt = " ".join(t[4] for t in items)
            plain_l = (
                "".join(c for c in unicodedata.normalize("NFD", line_txt).lower() if unicodedata.category(c) != "Mn")
                .replace("đ", "d")
                .replace("Đ", "D")
            )
            if any(k in plain_l for k in ("thoi hoc", "quyet dinh", "chuyen sang he", "cao dang", "dai hoc", "he chinh quy", "trinh do")):
                matched_lines.append(line_txt.strip())
                if first_box is None:
                    first_box = [
                        body_x0 + min(t[0] for t in items),
                        body_y0 + min(t[1] for t in items),
                        body_x0 + max(t[2] for t in items),
                        body_y0 + max(t[3] for t in items),
                    ]
        if matched_lines:
            reason_val = "; ".join(matched_lines[:2])
            structured_result["entities"]["reason"] = reason_val
            if first_box:
                structured_result["raw_detections"].append({
                    "class_name": "reason", "confidence": 0.85, "box": first_box, "text": reason_val
                })

    def _extract_identity_upper_fallback(self, img_pil, structured_result):
        """
        Trích xuất dự phòng Họ tên và MSSV ở nửa trên đơn (0.08H..0.48H) khi YOLOv11
        bỏ sót hộp student_name hoặc student_id trên các mẫu T01/T09 hoặc ảnh chụp camera.
        """
        if cv2 is None or self.ocr_predictor is None:
            return
        w_img, h_img = img_pil.size
        ents = structured_result["entities"]
        uy0, uy1 = int(h_img * 0.08), int(h_img * 0.48)
        ux0, ux1 = int(w_img * 0.04), int(w_img * 0.96)
        upper_crop = img_pil.crop((ux0, uy0, ux1, uy1))
        segs = self._segment_text_phrases(upper_crop, gap_w=16, min_w=16, min_h=8, max_h=70)
        if not segs:
            return
        lines_by_y = {}
        for sx0, sy0, sx1, sy1, txt, _ in segs:
            yk = round((sy0 + sy1) / 2.0 / 16.0)
            lines_by_y.setdefault(yk, []).append((sx0, sy0, sx1, sy1, txt))

        has_valid_id = bool(re.match(r"^[12]\d{7}$", str(ents.get("student_id", ""))))
        for yk in sorted(lines_by_y.keys()):
            items = sorted(lines_by_y[yk], key=lambda t: t[0])
            line_txt = " ".join(t[4] for t in items)
            if not ents.get("student_name"):
                m_nm = re.search(
                    r"(?:H[oọộc]\s*(?:v[àa]\s*)?t[eêi]n(?:\s*s[iị]nh\s*v[iị][eêệ]n|\s*v[iị][eêệ]n)?|H[oọ]\s*tinh\s*v[iị][eêệ]n|T[oô]i\s*t[eê]n(?:\s*l[àa])?|Em\s*t[eê]n(?:\s*l[àa])?)\s*[:.,-]?\s*([A-ZÀ-Ỹa-zà-ỹĐđ\s]{5,48})",
                    line_txt,
                    flags=re.IGNORECASE,
                )
                if m_nm:
                    cand = re.split(r"(?:Mã\s*(?:số\s*)?(?:SV|sinh\s*viên)|MSSV|MSV|SĐT|Điện\s*thoại|Sinh\s*ngày|Ngày\s*sinh|Lớp)", m_nm.group(1), flags=re.IGNORECASE)[0]
                    cand = re.sub(r"^(?:s[iị]nh\s*v[iị][eêệ]n|v[iị][eêệ]n)\s*[:.,-]?\s*", "", cand, flags=re.IGNORECASE).strip()
                    cand = re.sub(r"[\s.,;:()/\d|]+$", "", cand).strip()
                    cand = re.sub(r"\s+[Il]$", "", cand).strip()
                    if len(cand.split()) >= 2 and "sinh viện" not in cand.lower() and "sinh viên" not in cand.lower():
                        ents["student_name"] = cand
                        structured_result["raw_detections"].append({
                            "class_name": "student_name",
                            "confidence": 0.84,
                            "box": [ux0 + items[0][0], uy0 + items[0][1], ux0 + items[-1][2], uy0 + items[-1][3]],
                            "text": cand,
                        })
            if not has_valid_id:
                m_id = re.search(r"(?:Mã\s*(?:số\s*)?(?:SV|sinh\s*viên)|MSSV|MSV)\s*[:.,-]?\s*([A-Za-z0-9]{7,10})", line_txt, flags=re.IGNORECASE)
                if m_id:
                    cand_id = re.sub(r"\D", "", m_id.group(1).translate(_CODE_DIGIT_MAP))
                else:
                    m_b = re.search(r"\b([12]\d{7})\b", line_txt)
                    cand_id = m_b.group(1) if m_b else ""
                if re.match(r"^[12]\d{7}$", cand_id):
                    ents["student_id"] = cand_id
                    has_valid_id = True
                    structured_result["raw_detections"].append({
                        "class_name": "student_id",
                        "confidence": 0.84,
                        "box": [ux0 + items[0][0], uy0 + items[0][1], ux0 + items[-1][2], uy0 + items[-1][3]],
                        "text": cand_id,
                    })

    def _crop_tight_ink(self, cell_pil):
        """Cắt sát vùng chữ mực bên trong ô để loại bỏ khoảng trắng và đường kẻ bảng trước khi đưa vào VietOCR."""
        np_cell = np.array(cell_pil)
        if np_cell.size == 0 or np_cell.shape[0] < 10 or np_cell.shape[1] < 6:
            return None, False
        gray = cv2.cvtColor(np_cell, cv2.COLOR_RGB2GRAY) if np_cell.ndim == 3 else np_cell
        dark = (gray < 170).astype(np.uint8) * 255
        # Loại bỏ các đường kẻ bảng ngang/dọc dài nằm sót ở mép ô để tránh VietOCR ảo giác trên vạch kẻ trống
        if cv2 is not None:
            w_c, h_c = cell_pil.size
            if w_c >= 20:
                hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, int(w_c * 0.72)), 1))
                dark = cv2.subtract(dark, cv2.dilate(cv2.erode(dark, hk), hk))
            if h_c >= 16:
                vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, int(h_c * 0.72))))
                dark = cv2.subtract(dark, cv2.dilate(cv2.erode(dark, vk), vk))
        if int(np.sum(dark > 0)) < 10:
            return None, False
        ys, xs = np.where(dark > 0)
        x0, x1 = max(0, int(xs.min()) - 4), min(cell_pil.width, int(xs.max()) + 5)
        y0, y1 = max(0, int(ys.min()) - 3), min(cell_pil.height, int(ys.max()) + 4)
        if (x1 - x0) < 6 or (y1 - y0) < 10:
            return None, False
        return cell_pil.crop((x0, y0, x1, y1)), True

    def _extract_transcript_portal(self, img_pil, structured_result):
        """
        Trích xuất chính xác bảng điểm Cổng thông tin đào tạo (T10_BANG_DIEM), hỗ trợ cả ảnh
        dài nhiều học kỳ (như transcript_sample_*.png và ảnh thật NguyenBachTung_Academic_Transcript.jpg).
        """
        if cv2 is None or self.ocr_predictor is None:
            return False

        w_img, h_img = img_pil.size
        img_np = np.array(img_pil)
        gray_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        hsv_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        def _plain(s):
            n = unicodedata.normalize("NFD", str(s)).lower()
            return "".join(c for c in n if unicodedata.category(c) != "Mn").replace("đ", "d").replace("Đ", "D")

        # 1. Tìm thanh tiêu đề cột màu xanh đậm ("STT | Mã môn học | Tên môn học | Số TC | Điểm hệ 10 | Điểm hệ 4 | Điểm chữ")
        search_top = int(h_img * 0.05)
        search_bot = min(int(h_img * 0.35), 1000)
        sub_bgr_blue = img_np[search_top:search_bot, :, 2].astype(np.int16) - img_np[search_top:search_bot, :, 0].astype(np.int16)
        sub_bright = np.mean(img_np[search_top:search_bot], axis=2)
        navy_rows = np.mean((sub_bgr_blue > 18) & (sub_bright < 135), axis=1) > 0.55

        hdr_y0, hdr_y1 = None, None
        in_hdr, hs = False, 0
        for yi, is_n in enumerate(navy_rows):
            if is_n and not in_hdr:
                in_hdr, hs = True, yi
            elif not is_n and in_hdr:
                in_hdr = False
                if 16 <= (yi - hs) <= 85:
                    hdr_y0, hdr_y1 = search_top + hs, search_top + yi
                    break
        if in_hdr and hdr_y0 is None and 16 <= (len(navy_rows) - hs) <= 85:
            hdr_y0, hdr_y1 = search_top + hs, search_top + len(navy_rows)

        # Kiểm tra chắc chắn đây là bảng điểm portal
        if hdr_y0 is None:
            top_segs = self._segment_text_phrases(img_pil.crop((0, 0, w_img, min(int(h_img * 0.25), 750))), gap_w=14)
            top_plain = _plain(" ".join(s[4] for s in top_segs))
            if "ma mon" not in top_plain and "ket qua hoc tap" not in top_plain and "diem he 10" not in top_plain:
                return False
            # Ước lượng vị trí header từ cụm "Mã môn học" / "Tên môn học"
            hdr_cands = [s for s in top_segs if any(k in _plain(s[4]) for k in ("ma mon", "ten mon", "diem he 10"))]
            if not hdr_cands:
                return False
            hdr_y0 = max(0, min(s[1] for s in hdr_cands) - 6)
            hdr_y1 = min(h_img, max(s[3] for s in hdr_cands) + 6)

        # 2. Trích xuất thông tin hành chính phía trên thanh header bảng
        ents = structured_result["entities"]
        top_meta_crop = img_pil.crop((0, 0, w_img, hdr_y0))
        # Kết hợp Adaptive Threshold + Dark Threshold (150) để không bị viền xám của ô Dropdown làm mất tên CTĐT/Ngành
        meta_segs = self._segment_text_phrases(top_meta_crop, gap_w=14, min_w=12, min_h=8, max_h=80)
        dark_segs = self._segment_text_phrases(top_meta_crop, gap_w=14, min_w=12, min_h=8, max_h=60, remove_lines=True, dark_thresh=150)
        lines_meta = {}
        for sx0, sy0, sx1, sy1, txt, _ in (meta_segs + dark_segs):
            yk = round((sy0 + sy1) / 2.0 / 16.0)
            lines_meta.setdefault(yk, []).append((sx0, sy0, sx1, sy1, txt))

        for yk in sorted(lines_meta.keys()):
            # Khử trùng lặp cụm gần nhau khi gộp 2 ngưỡng
            dedup_items = []
            for it in sorted(lines_meta[yk], key=lambda t: t[0]):
                if not dedup_items or abs(it[0] - dedup_items[-1][0]) > 25:
                    dedup_items.append(it)
                elif len(it[4]) > len(dedup_items[-1][4]):
                    dedup_items[-1] = it
            items = dedup_items
            line_txt = " ".join(t[4] for t in items)
            line_p = _plain(line_txt)

            # Xem điểm sinh viên[:] Họ Tên (MSSV)
            m_sv = re.search(r"(?:sinh\s*viên\s*[:.-]?|:)\s*([^()]{4,45}?)\s*\(\s*([^()]{5,14})\s*\)", line_txt, flags=re.IGNORECASE)
            if m_sv and ("sinh vien" in line_p or "xem" in line_p or re.search(r"\d{5}", m_sv.group(2))):
                cand_name = re.sub(r"^(?:Sinh viên|Xem điểm sinh viên|O)\s*[:.-]?\s*", "", m_sv.group(1), flags=re.IGNORECASE).strip()
                cand_id = re.sub(r"[^A-Za-z0-9]", "", m_sv.group(2)).translate(_CODE_DIGIT_MAP)
                if len(cand_name.split()) >= 2 and not ents.get("student_name"):
                    ents["student_name"] = cand_name
                    structured_result["raw_detections"].append({
                        "class_name": "student_name", "confidence": 0.90,
                        "box": [items[0][0], items[0][1], items[0][2], items[0][3]], "text": cand_name
                    })
                if re.match(r"^\d{7,9}$", cand_id) and not ents.get("student_id"):
                    ents["student_id"] = cand_id
                    structured_result["raw_detections"].append({
                        "class_name": "student_id", "confidence": 0.90,
                        "box": [items[0][0], items[0][1], items[0][2], items[0][3]], "text": cand_id
                    })

            # Chương trình đào tạo (vd: 23110CLC_AI_Trí tuệ nhân tạo hoặc 23110CLC.ALTrí tuệ nhân tạo)
            if re.search(r"\b2\d{4}[A-Za-z]{0,5}[_.\s]", line_txt) or any(mk in line_p for mk in ("tri tue nhan tao", "cong nghe thong tin", "co khi che tao", "ky thuat du lieu")):
                if "xem diem" not in line_p and not ents.get("major"):
                    clean_deg = re.sub(r"\s+", " ", line_txt).strip()
                    clean_deg = re.sub(r"^(2\d{2})\s+(\d{2})", r"\1\2", clean_deg)
                    clean_deg = re.sub(
                        r"^(2\d{4}[A-Za-z]{0,5})[._\s]+(A[IL1]|KHMT|KTMT|KTPM|CNTT|HTTT|ATTT|CNKT|[A-Z]{2})[._\s]*(?=[A-ZĐÀ-Ỹ])",
                        lambda m: f"{m.group(1)}_{'AI' if m.group(2) in ('AL', 'A1') else m.group(2)}_",
                        clean_deg,
                    )
                    clean_deg = re.sub(r"\.(?=[A-Z]{2}_)", "_", clean_deg)
                    # Loại bỏ số ký tự icon mũi tên dropdown ở cuối nếu bị dính vào
                    clean_deg = re.sub(r"\s+[0-9]$", "", clean_deg).strip()
                    ents["degree_info"] = clean_deg
                    if "_" in clean_deg:
                        ents["major"] = clean_deg.split("_")[-1].strip()
                    else:
                        ents["major"] = re.sub(r"^2\d{4}[A-Za-z0-9]{0,6}\s*", "", clean_deg).strip() or clean_deg
                    m_cls = re.match(r"^(2\d{4}[A-Za-z0-9]{0,5})", clean_deg)
                    if m_cls and not ents.get("class_name"):
                        ents["class_name"] = m_cls.group(1)

        # Fallback phóng to 2x dải ngay trên thanh header (0.02W..0.48W, hdr_y0-68..hdr_y0-4) để đọc rõ Tên & MSSV trong ngoặc
        if (not ents.get("student_name") or not ents.get("student_id")) and hdr_y0 >= 40:
            sv_y0, sv_y1 = max(0, hdr_y0 - 68), max(10, hdr_y0 - 2)
            sv_x0, sv_x1 = int(w_img * 0.02), int(w_img * 0.48)
            sv_strip = img_pil.crop((sv_x0, sv_y0, sv_x1, sv_y1))
            sv_up = sv_strip.resize((sv_strip.width * 2, sv_strip.height * 2), Image.Resampling.BICUBIC)
            up_segs = self._segment_text_phrases(sv_up, gap_w=24, min_w=18, min_h=12, max_h=110)
            up_lines = {}
            for sx0, sy0, sx1, sy1, txt, _ in up_segs:
                yk = round((sy0 + sy1) / 2.0 / 28.0)
                up_lines.setdefault(yk, []).append((sx0, sy0, sx1, sy1, txt))
            for yk in sorted(up_lines.keys()):
                items = sorted(up_lines[yk], key=lambda t: t[0])
                l_txt = " ".join(t[4] for t in items)
                l_p = _plain(l_txt)
                if "xem diem" in l_p or "sinh vien" in l_p or "(" in l_txt:
                    m_sv2 = re.search(r"(?:sinh\s*viên\s*[:.-]?|:)\s*([^()]{4,45}?)\s*\(\s*([^()]{5,14})\s*\)", l_txt, flags=re.IGNORECASE)
                    if m_sv2:
                        c_name = re.sub(r"^(?:Sinh viên|Xem điểm sinh viên|O)\s*[:.-]?\s*", "", m_sv2.group(1), flags=re.IGNORECASE).strip()
                        c_id = re.sub(r"[^A-Za-z0-9]", "", m_sv2.group(2)).translate(_CODE_DIGIT_MAP)
                        if len(c_name.split()) >= 2 and not ents.get("student_name"):
                            ents["student_name"] = c_name
                            structured_result["raw_detections"].append({
                                "class_name": "student_name", "confidence": 0.88,
                                "box": [sv_x0 + items[0][0] // 2, sv_y0 + items[0][1] // 2, sv_x0 + items[-1][2] // 2, sv_y0 + items[-1][3] // 2],
                                "text": c_name,
                            })
                        if re.match(r"^\d{7,9}$", c_id) and not ents.get("student_id"):
                            ents["student_id"] = c_id
                            structured_result["raw_detections"].append({
                                "class_name": "student_id", "confidence": 0.88,
                                "box": [sv_x0 + items[0][0] // 2, sv_y0 + items[0][1] // 2, sv_x0 + items[-1][2] // 2, sv_y0 + items[-1][3] // 2],
                                "text": c_id,
                            })
                        elif ents.get("student_name") and not ents.get("student_id"):
                            # Nếu phần (MSSV) trong dòng vẫn bị mờ dấu, phóng to riêng đoạn ngoặc cuối để OCR lại
                            paren_box = items[-1]
                            px0 = sv_x0 + max(0, paren_box[0] // 2 - 4)
                            py0_p = sv_y0 + max(0, paren_box[1] // 2 - 3)
                            px1 = sv_x0 + min(sv_strip.width, paren_box[2] // 2 + 4)
                            py1_p = sv_y0 + min(sv_strip.height, paren_box[3] // 2 + 3)
                            p_crop = img_pil.crop((px0, py0_p, px1, py1_p))
                            p_up = p_crop.resize((max(40, p_crop.width * 3), max(28, p_crop.height * 3)), Image.Resampling.BICUBIC)
                            p_txt, _ = self._ocr_crop(p_up)
                            m_pid = re.search(r"([12]\d{7}|\d{8})", p_txt.translate(_CODE_DIGIT_MAP))
                            if m_pid:
                                ents["student_id"] = m_pid.group(1)
                                structured_result["raw_detections"].append({
                                    "class_name": "student_id", "confidence": 0.86,
                                    "box": [px0, py0_p, px1, py1_p], "text": m_pid.group(1),
                                })

        # 3. Xác định tọa độ X từng cột từ thanh header (chữ trắng trên nền xanh đậm)
        hdr_strip = img_np[hdr_y0:hdr_y1, :]
        hdr_gray = cv2.cvtColor(hdr_strip, cv2.COLOR_RGB2GRAY)
        is_dark_hdr = float(np.mean(hdr_gray)) < 145.0
        if is_dark_hdr:
            _, hdr_mask = cv2.threshold(hdr_gray, 170, 255, cv2.THRESH_BINARY)
        else:
            hdr_mask = cv2.adaptiveThreshold(~hdr_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        hdr_dil = cv2.dilate(hdr_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 4)))
        hdr_cnts, _ = cv2.findContours(hdr_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hdr_cands, hdr_crops = [], []
        for c in hdr_cnts:
            hx, hy, hw, hh = cv2.boundingRect(c)
            if hw >= 14 and hh >= 8:
                sub_h = hdr_gray[max(0, hy - 2):min(hdr_strip.shape[0], hy + hh + 2), max(0, hx - 2):min(w_img, hx + hw + 2)]
                if is_dark_hdr:
                    sub_h = 255 - sub_h
                hdr_cands.append((hx, hx + hw))
                hdr_crops.append(Image.fromarray(sub_h).convert("RGB"))
        hdr_outs = self._ocr_batch(hdr_crops) if hdr_crops else []
        hdr_cols = [(hx0, hx1, txt_h) for (hx0, hx1), (txt_h, _) in zip(hdr_cands, hdr_outs) if txt_h]
        hdr_cols.sort(key=lambda item: item[0])

        # Ánh xạ tâm X của từng cột chuẩn trên cổng thông tin
        col_centers = {}
        for hx0, hx1, htxt in hdr_cols:
            hp = _plain(htxt)
            cx = (hx0 + hx1) / 2.0
            if "stt" in hp:
                col_centers["stt"] = (hx0, hx1, cx)
            elif "ma" in hp and ("mon" in hp or "mh" in hp or hx0 < w_img * 0.22):
                col_centers["code"] = (hx0, hx1, cx)
            elif "ten" in hp:
                col_centers["name"] = (hx0, hx1, cx)
            elif "tc" in hp or "tin" in hp or "tic" in hp or "so tn" in hp or hp == "tn":
                col_centers["credits"] = (hx0, hx1, cx)
            elif "10" in hp:
                col_centers["s10"] = (hx0, hx1, cx)
            elif "4" in hp and ("he" in hp or "diem" in hp):
                col_centers["s4"] = (hx0, hx1, cx)
            elif "chu" in hp or "chui" in hp:
                col_centers["letter"] = (hx0, hx1, cx)
            elif "ket" in hp or "qua" in hp:
                col_centers["result"] = (hx0, hx1, cx)

        # Tính ranh giới X của từng cột dựa trên khe hở giữa mép ngoài tiêu đề liên tiếp (tránh chạm sang icon Kết quả)
        x_code_0 = int(col_centers["stt"][1] + 2) if "stt" in col_centers else int(w_img * 0.068)
        x_code_0 = min(x_code_0, int(col_centers["code"][0] - 8)) if "code" in col_centers else x_code_0
        x_code_1 = int(col_centers["code"][1] + 18) if "code" in col_centers else int(w_img * 0.185)
        x_code_1 = max(x_code_1, int(w_img * 0.175))
        x_name_0 = x_code_1 - 4
        if "credits" in col_centers:
            x_cred_0 = int(col_centers["credits"][0] - 12)
            x_cred_1 = int((col_centers["credits"][1] + col_centers["s10"][0]) / 2.0) if "s10" in col_centers else int(col_centers["credits"][1] + 12)
        else:
            x_cred_0 = int(w_img * 0.455)
            x_cred_1 = int((int(w_img * 0.525) + col_centers["s10"][0]) / 2.0) if "s10" in col_centers else int(w_img * 0.525)
        x_name_1 = x_cred_0 - 2

        if "s10" in col_centers and "s4" in col_centers:
            x_s10_0 = max(x_cred_1 + 2, int(col_centers["s10"][0] - 8))
            x_s10_1 = int((col_centers["s10"][1] + col_centers["s4"][0]) / 2.0)
            x_s4_0 = x_s10_1 + 1
            if "letter" in col_centers:
                x_s4_1 = int((col_centers["s4"][1] + col_centers["letter"][0]) / 2.0)
                x_let_0 = x_s4_1 + 1
                if "result" in col_centers:
                    x_let_1 = int(col_centers["result"][0] - 5)
                else:
                    x_let_1 = min(w_img, int(col_centers["letter"][1] + 8))
            elif "result" in col_centers and (col_centers["result"][0] - col_centers["s4"][1]) >= 45:
                # Cột Điểm chữ luôn nằm trọn vẹn trong khoảng giữa 'Điểm hệ 4' và 'Kết quả'
                x_s4_1 = int(col_centers["s4"][1] + 6)
                x_let_0 = x_s4_1 + 2
                x_let_1 = int(col_centers["result"][0] - 5)
            else:
                x_s4_1 = int(col_centers["s4"][1] + 8)
                x_let_0 = x_s4_1 + 2
                x_let_1 = min(w_img, x_let_0 + int(w_img * 0.065))
        else:
            x_s10_0, x_s10_1 = int(w_img * 0.535), int(w_img * 0.635)
            x_s4_0, x_s4_1 = int(w_img * 0.635), int(w_img * 0.720)
            x_let_0, x_let_1 = int(w_img * 0.720), int(w_img * 0.805)

        # 4. Tìm các dải dòng học phần (nền trắng, không phải thanh màu Năm học/Học kỳ/Tổng kết)
        scan_y0 = hdr_y1 + 2
        row_dark = (gray_full[scan_y0:, int(w_img * 0.03):int(w_img * 0.79)] < 140)
        row_sat = hsv_full[scan_y0:, int(w_img * 0.03):int(w_img * 0.79), 1]
        row_bg_med = np.median(gray_full[scan_y0:, int(w_img * 0.03):int(w_img * 0.79)], axis=1)
        row_sat_med = np.median(row_sat, axis=1)
        row_counts = np.sum(row_dark, axis=1)

        in_row, rs = False, 0
        row_bands = []
        for i in range(len(row_counts)):
            is_white_bg = (row_bg_med[i] >= 225) and (row_sat_med[i] <= 24)
            has_text = row_counts[i] >= 8 and is_white_bg
            if has_text and not in_row:
                in_row, rs = True, i
            elif not has_text and in_row:
                in_row = False
                if 7 <= (i - rs) <= 44:
                    row_bands.append((scan_y0 + rs, scan_y0 + i))

        def _pad_small(pil_im):
            if pil_im is None:
                return None
            if pil_im.width < 64 or pil_im.height < 28:
                return ImageOps.expand(pil_im, border=4, fill=(255, 255, 255))
            return pil_im

        # Giai đoạn 4A: Crop toàn bộ ô Mã + Tên để chạy batch OCR nhanh gấp ~10 lần
        row_crops_meta = []
        code_name_crops = []
        for ry0, ry1 in row_bands:
            py0, py1 = max(0, ry0 - 4), min(h_img, ry1 + 4)
            code_pil, has_code_ink = self._crop_tight_ink(img_pil.crop((x_code_0, py0, x_code_1, py1)))
            name_pil, has_name_ink = self._crop_tight_ink(img_pil.crop((x_name_0, py0, x_name_1, py1)))
            if not has_code_ink and not has_name_ink:
                continue
            c_idx = len(code_name_crops) if has_code_ink else None
            if has_code_ink:
                code_name_crops.append(code_pil)
            n_idx = len(code_name_crops) if has_name_ink else None
            if has_name_ink:
                code_name_crops.append(name_pil)
            row_crops_meta.append((py0, py1, c_idx, n_idx))

        cn_results = self._ocr_batch(code_name_crops) if code_name_crops else []

        # Giai đoạn 4B: Kiểm tra & nối ghép nếu mã+tên bị lệch ranh giới cột, đồng thời chuẩn bị batch TC + Điểm
        matched_rows = []
        merged_crops, merged_indices = [], []
        for py0, py1, c_idx, n_idx in row_crops_meta:
            code_raw, code_conf = cn_results[c_idx] if c_idx is not None else ("", None)
            name_raw, _ = cn_results[n_idx] if n_idx is not None else ("", None)
            plain_check = _plain(f"{code_raw} {name_raw}")
            if any(skip in plain_check for skip in ("nam hoc", "hoc ky", "so tin chi", "diem tb", "diem ren", "xem diem")):
                continue

            code_norm = _normalize_course_code_ocr(code_raw)
            can_code, can_name, can_cred, sim = ground_course(code_norm, name_raw)
            row_pos = len(matched_rows)
            matched_rows.append([py0, py1, code_conf, can_code, can_name, can_cred, sim])
            if not re.match(r"^[A-Z]{2,5}\d{5,6}$", can_code) and sim < 0.72:
                merged_pil, has_m = self._crop_tight_ink(img_pil.crop((x_code_0, py0, x_name_1, py1)))
                if has_m:
                    merged_indices.append(row_pos)
                    merged_crops.append(merged_pil)

        if merged_crops:
            m_outs = self._ocr_batch(merged_crops)
            for row_pos, (m_txt, _) in zip(merged_indices, m_outs):
                m_split = re.match(r"^([A-Za-z]{2,5}\s*[A-Za-z0-9]{5,7})\s+(.*)$", m_txt.strip())
                if m_split:
                    matched_rows[row_pos][3:7] = list(ground_course(m_split.group(1), m_split.group(2)))

        valid_rows = [
            mr for mr in matched_rows
            if re.match(r"^[A-Z]{2,5}\d{5,6}$", mr[3]) or mr[6] >= 0.72
        ]

        # Giai đoạn 4C: Batch OCR toàn bộ cột Số TC, Điểm hệ 10, Điểm hệ 4, Điểm chữ cho các dòng hợp lệ
        grade_crops = []
        grade_indices = []
        for py0, py1, code_conf, can_code, can_name, can_cred, sim in valid_rows:
            idxs = [None, None, None, None]
            for j, (gx0, gx1) in enumerate(((x_cred_0, x_cred_1), (x_s10_0, x_s10_1), (x_s4_0, x_s4_1), (x_let_0, x_let_1))):
                g_pil, has_g = self._crop_tight_ink(img_pil.crop((gx0, py0, gx1, py1)))
                if has_g:
                    idxs[j] = len(grade_crops)
                    grade_crops.append(_pad_small(g_pil))
            grade_indices.append(idxs)

        grade_outs = self._ocr_batch(grade_crops) if grade_crops else []

        extracted_courses = []
        for (py0, py1, code_conf, can_code, can_name, can_cred, _), (i_cr, i_s10, i_s4, i_let) in zip(valid_rows, grade_indices):
            cred_text = ""
            if i_cr is not None:
                c_raw, _ = grade_outs[i_cr]
                m_c = re.search(r"\d+", c_raw.translate(_CODE_DIGIT_MAP))
                if m_c:
                    cred_text = m_c.group(0)
            if not cred_text and can_cred is not None:
                cred_text = str(can_cred)

            s10_text = ""
            if i_s10 is not None:
                s10_raw, _ = grade_outs[i_s10]
                m_10 = re.search(r"(\d{1,2}(?:[.,]\d)?)", s10_raw.replace(",", ".").translate(_CODE_DIGIT_MAP))
                if m_10:
                    s10_text = clean_academic_score(m_10.group(1))

            s4_text = ""
            if i_s4 is not None:
                s4_raw, _ = grade_outs[i_s4]
                m_4 = re.search(r"([0-4](?:[.,]\d)?)", s4_raw.replace(",", ".").translate(_CODE_DIGIT_MAP))
                if m_4:
                    try:
                        v4 = float(m_4.group(1))
                        if 0.0 <= v4 <= 4.0:
                            s4_text = f"{v4:.1f}"
                    except Exception:
                        pass

            let_text = ""
            if i_let is not None:
                let_raw, _ = grade_outs[i_let]
                # Chuẩn hóa điểm chữ từ OCR (vd AA4 -> A+, AT -> A+, IB -> B, (P -> P, B4 -> B+)
                # Không ép '0'/'O' -> 'D' vì icon dấu tích ở cột Kết quả rất hay bị OCR đọc thành '0'/'O'
                l_clean = re.sub(r"[^A-Za-z0-9+]", "", let_raw).upper()
                l_clean = re.sub(r"^(A|B|C|D)[T4]$", r"\1+", l_clean)
                l_clean = l_clean.replace("1", "")
                if l_clean in ("AA+", "AA", "A+A"):
                    l_clean = "A+"
                elif l_clean.startswith("I") and len(l_clean) == 2:
                    l_clean = l_clean[1]
                m_l = re.match(r"^([ABCDF]\+?|[P])$", l_clean)
                if m_l:
                    let_text = m_l.group(1)

            row_scores = [x for x in (s10_text, s4_text, let_text) if x]
            row_entry = {
                "course_code": can_code,
                "course_name": can_name,
                "credits": cred_text,
                "score": " | ".join(row_scores) if row_scores else "",
                "scores": row_scores,
                "row_box": [x_code_0, py0, x_let_1, py1],
            }
            if code_conf is not None:
                row_entry["min_ocr_confidence"] = round(code_conf, 3)
            extracted_courses.append(row_entry)

        if not extracted_courses:
            return False

        # Thay thế hoàn toàn raw_detections bảng cũ bằng các ô đã căn chỉnh chính xác
        structured_result["raw_detections"] = [
            d for d in structured_result.get("raw_detections", [])
            if d.get("class_name") not in TABLE_CLASSES and d.get("class_name") != "signature"
        ]
        for row in extracted_courses:
            rx0, ry0, _, ry1 = row["row_box"]
            structured_result["raw_detections"].append({
                "class_name": "course_code", "confidence": 0.92,
                "box": [x_code_0, ry0, x_code_1, ry1], "text": row["course_code"]
            })
            structured_result["raw_detections"].append({
                "class_name": "course_name", "confidence": 0.92,
                "box": [x_name_0, ry0, x_name_1, ry1], "text": row["course_name"]
            })
            if row.get("credits"):
                structured_result["raw_detections"].append({
                    "class_name": "credits", "confidence": 0.92,
                    "box": [x_cred_0, ry0, x_cred_1, ry1], "text": row["credits"]
                })
            if row.get("score"):
                structured_result["raw_detections"].append({
                    "class_name": "score", "confidence": 0.92,
                    "box": [x_s10_0, ry0, x_let_1, ry1], "text": row["score"]
                })
        structured_result["courses_table"] = extracted_courses
        structured_result["verification"]["signature_count"] = 0
        structured_result["verification"]["has_signature"] = False
        return True

    def _extract_table_grid_fallback(self, img_pil, structured_result, form_type=None, detect_rows=True):
        """
        Kỹ thuật Hybrid Document AI: Khi YOLOv11 bỏ sót các ô bảng điểm do kích thước
        quá nhỏ (Downsampling Loss), sử dụng OpenCV Morphology để tự động phát hiện khung lưới
        bảng (Table Grid), tự động ánh xạ cột từ dòng tiêu đề và cắt sát vùng mực từng ô.
        """
        if cv2 is None or self.ocr_predictor is None:
            return None

        img_np = np.array(img_pil)
        h_img, w_img = img_np.shape[:2]
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        bw = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

        # 1. Phát hiện đường ngang và dọc của bảng
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(w_img / 30)), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(h_img / 35))))
        horiz = cv2.dilate(cv2.erode(bw, h_kernel), h_kernel)
        vert = cv2.dilate(cv2.erode(bw, v_kernel), v_kernel)
        table_mask = horiz + vert

        # Mặt nạ chữ đã khử hoàn toàn viền bảng
        bw_nolines = cv2.subtract(bw, horiz)
        bw_nolines = cv2.subtract(bw_nolines, vert)

        # 2. Tìm bounding box của bảng
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t_boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > w_img * 0.4 and cv2.boundingRect(c)[3] > h_img * 0.08]
        if not t_boxes:
            return None

        tx, ty, tw, th = sorted(t_boxes, key=lambda b: b[2] * b[3], reverse=True)[0]
        if not detect_rows:
            return ty + th
        table_crop_bw = bw[ty:ty + th, tx:tx + tw]

        # 3. Phân tách các đường phân chia dòng ngang (Horizontal row lines)
        h_k_sub = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(tw / 25)), 1))
        sub_horiz = cv2.dilate(cv2.erode(table_crop_bw, h_k_sub), h_k_sub)
        sub_horiz_dil = cv2.dilate(sub_horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        row_proj = np.sum(sub_horiz_dil > 0, axis=1)
        row_thresh = max(80, int(tw * 0.16))
        row_peaks = [i for i in range(1, len(row_proj) - 1) if row_proj[i] > row_thresh and row_proj[i] >= row_proj[i - 1] and row_proj[i] >= row_proj[i + 1]]
        row_lines = []
        for p in row_peaks:
            if not row_lines or p - row_lines[-1] >= 18:
                row_lines.append(p)

        # 4. Phân tách các đường phân chia cột dọc (có bù sai số nghiêng nhẹ từ ảnh scan/camera)
        v_k_sub = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, int(th / 30))))
        sub_vert = cv2.dilate(cv2.erode(table_crop_bw, v_k_sub), v_k_sub)
        sub_vert_dil = cv2.dilate(sub_vert, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
        col_proj = np.sum(sub_vert_dil > 0, axis=0)
        col_thresh = max(35, int(th * 0.10))
        col_peaks = [i for i in range(1, len(col_proj) - 1) if col_proj[i] > col_thresh and col_proj[i] >= col_proj[i - 1] and col_proj[i] >= col_proj[i + 1]]
        col_lines = []
        for p in col_peaks:
            if not col_lines or p - col_lines[-1] > 20:
                col_lines.append(p)

        if not col_lines:
            return None
        if col_lines[0] > 40:
            col_lines.insert(0, 0)
        if tw - col_lines[-1] > 50:
            col_lines.append(tw)

        if len(row_lines) < 2 or len(col_lines) < 3:
            return None

        # Tính độ nghiêng dốc (tilt slope) của bảng để bù tọa độ y theo từng cột
        ref_y = row_lines[2] if len(row_lines) > 2 else row_lines[0]
        left_x = col_lines[1] + 20 if len(col_lines) > 1 else 20
        right_x = col_lines[-2] - 20 if len(col_lines) > 1 else tw - 20
        left_win = np.sum(sub_horiz[max(0, ref_y - 12):min(th, ref_y + 12), max(0, left_x - 15):min(tw, left_x + 15)], axis=1)
        right_win = np.sum(sub_horiz[max(0, ref_y - 12):min(th, ref_y + 12), max(0, right_x - 15):min(tw, right_x + 15)], axis=1)
        y_left_peak = (int(np.argmax(left_win)) - 12 + ref_y) if len(left_win) > 0 else ref_y
        y_right_peak = (int(np.argmax(right_win)) - 12 + ref_y) if len(right_win) > 0 else ref_y
        slope = (y_right_peak - y_left_peak) / float(max(1, right_x - left_x))

        def get_dy(col_idx):
            if col_idx is None or col_idx + 1 >= len(col_lines):
                return 0
            c_center = (col_lines[col_idx] + col_lines[col_idx + 1]) / 2.0
            return int(round(slope * (c_center - left_x)))

        def _plain(s):
            n = unicodedata.normalize("NFD", str(s)).lower()
            return "".join(c for c in n if unicodedata.category(c) != "Mn").replace("đ", "d").replace("Đ", "D")

        # 5. Tự động đọc dòng tiêu đề của bảng để ánh xạ đúng cột (tránh đảo ngược Mã môn <-> Tên môn)
        detected_form = form_type or structured_result.get("validation", {}).get("form_type") or ""
        if not detected_form or detected_form == "UNKNOWN":
            detected_form = self._detect_form_type_from_header(img_pil, structured_result)

        col_code_idx = None
        col_name_idx = None
        col_cred_idx = None
        col_score_idx = None
        combined_code_cred_col = None

        hdr_texts_by_row = []
        hdr_batch_crops, hdr_batch_map = [], []
        for hr in range(min(2, len(row_lines) - 1)):
            ry0_h, ry1_h = row_lines[hr], row_lines[hr + 1]
            row_h_txts = ["" for _ in range(len(col_lines) - 1)]
            for ci in range(len(col_lines) - 1):
                dy_h = get_dy(ci)
                c_pil = img_pil.crop((
                    max(0, tx + col_lines[ci] + 7),
                    max(0, ty + ry0_h + dy_h + 5),
                    min(w_img, tx + col_lines[ci + 1] - 7),
                    min(h_img, ty + ry1_h + dy_h - 5),
                ))
                t_pil, has_ink = self._crop_tight_ink(c_pil)
                if has_ink:
                    hdr_batch_map.append((hr, ci))
                    hdr_batch_crops.append(t_pil)
            hdr_texts_by_row.append(row_h_txts)
        if hdr_batch_crops:
            h_outs = self._ocr_batch(hdr_batch_crops)
            for (hr, ci), (h_txt, _) in zip(hdr_batch_map, h_outs):
                hdr_texts_by_row[hr][ci] = h_txt

        # Kiểm tra cả dòng 0 và dòng 1 để tìm cột Mã / Tên / TC / Điểm
        for r_txts in hdr_texts_by_row[:2]:
            for ci, h_txt in enumerate(r_txts):
                hp = _plain(h_txt)
                if not hp:
                    continue
                if "stt" in hp and ci == 0:
                    continue
                if ("ma" in hp and ("mon" in hp or "mh" in hp or "hoc phan" in hp)) and col_code_idx is None:
                    col_code_idx = ci
                    if "tin" in hp or "tc" in hp:
                        combined_code_cred_col = ci
                elif ("ten" in hp and ("mon" in hp or "mh" in hp or "hoc phan" in hp)) and col_name_idx is None:
                    col_name_idx = ci
                elif ("tc" in hp or "tin chi" in hp or "so tin" in hp) and col_cred_idx is None:
                    col_cred_idx = ci
                elif "diem" in hp and col_score_idx is None:
                    col_score_idx = ci

        # Xác định start_row_idx (nhảy qua tiêu đề 1 dòng hoặc tiêu đề 2 dòng gộp)
        start_row_idx = 1
        if len(hdr_texts_by_row) >= 2:
            r1_plain = " ".join(_plain(t) for t in hdr_texts_by_row[1])
            if any(kw in r1_plain for kw in ("ma ", "ten ", "tin chi", "diem", "xac nhan", "dong y", "hoc phan", "(cu)")):
                start_row_idx = 2

        # Fallback ánh xạ cột theo đặc thù loại biểu mẫu nếu OCR header bị mờ
        n_cols = len(col_lines) - 1
        is_diem_i = ("DIEM_I" in detected_form) or n_cols >= 9 and not any(k in detected_form for k in ("CHUYEN_DIEM", "BAO_LUU"))
        if "MIEN_THI" in detected_form and n_cols == 3:
            col_name_idx = 1
            col_code_idx = 2
            combined_code_cred_col = 2
            col_cred_idx = None
            col_score_idx = None
        elif any(kw in detected_form for kw in ("CHUYEN_DIEM", "T04", "T05", "T07", "BAO_LUU")):
            if col_name_idx is None:
                col_name_idx = 1
            if col_code_idx is None:
                col_code_idx = 2
            if col_cred_idx is None and n_cols >= 4:
                col_cred_idx = 3
            if col_score_idx is None and n_cols >= 5:
                col_score_idx = 4
        elif is_diem_i:
            col_code_idx = col_code_idx if col_code_idx is not None else 1
            col_name_idx = col_name_idx if col_name_idx is not None else 2
            col_cred_idx = col_cred_idx if col_cred_idx is not None else (3 if n_cols >= 4 else None)
            col_score_idx = min(8, n_cols - 2) if n_cols >= 6 else None
            start_row_idx = max(start_row_idx, 2) if len(row_lines) > 5 else start_row_idx
        else:
            if col_code_idx is None:
                col_code_idx = 1
            if col_name_idx is None:
                col_name_idx = 2 if n_cols >= 3 else None

        # Bảo vệ chống chọn trùng cột giữa code và name
        if col_code_idx == col_name_idx:
            col_name_idx = 1
            col_code_idx = 2 if n_cols >= 3 else 1

        # Giai đoạn A: Thu thập toàn bộ crop các ô (Mã, Tên, TC, Điểm) hợp lệ để chạy Batch OCR nhanh
        pending_rows = []
        cell_batch_crops = []
        for r_idx in range(start_row_idx, len(row_lines) - 1):
            ry0, ry1 = row_lines[r_idx], row_lines[r_idx + 1]
            if ry1 - ry0 < 18:
                continue

            # Kiểm tra mật độ mực sạch (đã trừ đường kẻ ngang/dọc + bù nghiêng) trong cột Tên & Mã
            has_any_cell_content = False
            for check_ci in (col_name_idx, col_code_idx):
                if check_ci is None or check_ci + 1 >= len(col_lines):
                    continue
                dy_c = get_dy(check_ci)
                cx0 = max(0, tx + col_lines[check_ci] + 7)
                cx1 = min(w_img, tx + col_lines[check_ci + 1] - 7)
                cy0 = max(0, ty + ry0 + dy_c + 7)
                cy1 = min(h_img, ty + ry1 + dy_c - 7)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                cell_bw = bw_nolines[cy0:cy1, cx0:cx1]
                # Yêu cầu có thành phần nét chữ thật (loại bỏ nhiễu hạt chấm đơn lẻ)
                ink_pct = float(np.mean(cell_bw > 0) * 100.0)
                if ink_pct >= 0.45:
                    has_any_cell_content = True
                    break

            if not has_any_cell_content:
                # Hàng trống hoàn toàn -> dừng trích xuất ngay để không đọc hàng trống bên dưới
                break

            row_idxs = {"code": None, "name": None, "cred": None, "score": None}
            col_specs = [
                ("code", col_code_idx, 8, 7),
                ("name", col_name_idx, 8, 7),
                ("cred", col_cred_idx, 7, 7),
                ("score", col_score_idx, 7, 7),
            ]
            for key, ci, px_pad, py_pad in col_specs:
                if key == "cred" and combined_code_cred_col == col_code_idx:
                    continue
                if ci is not None and ci + 1 < len(col_lines):
                    dy_ci = get_dy(ci)
                    c_raw = img_pil.crop((
                        max(0, tx + col_lines[ci] + px_pad),
                        max(0, ty + ry0 + dy_ci + py_pad),
                        min(w_img, tx + col_lines[ci + 1] - px_pad),
                        min(h_img, ty + ry1 + dy_ci - py_pad),
                    ))
                    c_tight, has_ink = self._crop_tight_ink(c_raw)
                    if has_ink:
                        if c_tight.width < 56 or c_tight.height < 26:
                            c_tight = ImageOps.expand(c_tight, border=4, fill=(255, 255, 255))
                        row_idxs[key] = len(cell_batch_crops)
                        cell_batch_crops.append(c_tight)
            pending_rows.append((ry0, ry1, row_idxs))

        cell_batch_outs = self._ocr_batch(cell_batch_crops) if cell_batch_crops else []

        extracted_courses = []
        for ry0, ry1, row_idxs in pending_rows:
            code_text, name_text, score_text, cred_text = "", "", "", ""

            if row_idxs["code"] is not None:
                c_ocr, _ = cell_batch_outs[row_idxs["code"]]
                if combined_code_cred_col == col_code_idx:
                    # Mẫu T09 Miễn thi gộp "Mã MH - Số TC" trong cùng 1 ô (vd "MATH141701 - 3 TC")
                    m_cc = re.search(r"([A-Za-z]{2,5}[A-Za-z0-9]{5,7})", c_ocr)
                    code_text = _normalize_course_code_ocr(m_cc.group(1)) if m_cc else _normalize_course_code_ocr(c_ocr)
                    m_tc = re.search(r"[-/]\s*(\d{1,2})\s*(?:TC|tín)?", c_ocr, re.IGNORECASE)
                    if m_tc:
                        cred_text = m_tc.group(1)
                else:
                    code_text = _normalize_course_code_ocr(c_ocr)

            if row_idxs["name"] is not None:
                name_text, _ = cell_batch_outs[row_idxs["name"]]

            if not cred_text and row_idxs["cred"] is not None:
                cred_raw, _ = cell_batch_outs[row_idxs["cred"]]
                m_cred = re.search(r"\d+", cred_raw.translate(_CODE_DIGIT_MAP))
                cred_text = m_cred.group(0) if m_cred else ""

            if row_idxs["score"] is not None:
                score_raw, _ = cell_batch_outs[row_idxs["score"]]
                score_text = clean_academic_score(score_raw)

            name_plain = _plain(name_text)
            code_plain = _plain(code_text)
            if any(hdr_kw in name_plain for hdr_kw in ("ten mon", "ten hoc phan")) or "mamh" in code_plain:
                continue

            # Khử trường hợp hiếm hoi cột mã và tên bị đảo do mẫu tùy biến: nếu code_text không phải mã môn nhưng name_text lại khớp mã môn
            if not re.match(r"^[A-Z]{2,5}\d{5,6}$", code_text) and re.match(r"^[A-Z]{2,5}\d{5,6}$", _normalize_course_code_ocr(name_text)):
                code_text, name_text = _normalize_course_code_ocr(name_text), code_text

            # Grounding với Knowledge Base đầy đủ của HCMUTE
            canonical_code, canonical_name, canonical_cred, sim = ground_course(code_text, name_text)
            # Chỉ giữ dòng thực sự là môn học (hoặc khớp Knowledge Base, hoặc có mã môn hợp lệ)
            is_valid_code = bool(re.match(r"^[A-Z]{2,5}\d{5,6}$", canonical_code))
            if sim < 0.72 and not is_valid_code:
                # Tránh nhận rác ký tự ngắn từ nhiễu ở dòng trống
                continue

            code_text = canonical_code
            if canonical_name:
                name_text = canonical_name
            if not cred_text and canonical_cred:
                cred_text = str(canonical_cred)

            if len(code_text) >= 3 or len(name_text) >= 4:
                row_record = {
                    "course_code": code_text,
                    "course_name": name_text,
                    "score": score_text,
                    "scores": [score_text] if score_text else [],
                    "credits": cred_text,
                    "row_box": [tx, ty + ry0, tx + tw, ty + ry1]
                }
                extracted_courses.append(row_record)

        if extracted_courses:
            def is_inside_table(box):
                bx0, by0, bx1, by1 = box
                cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
                return (tx - 5 <= cx <= tx + tw + 5) and (ty - 5 <= cy <= ty + th + 5)

            if "date" in structured_result.get("entities", {}):
                for d in structured_result.get("raw_detections", []):
                    if d.get("class_name") == "date" and is_inside_table(d["box"]):
                        if structured_result["entities"]["date"] == d.get("text"):
                            del structured_result["entities"]["date"]
                            break

            structured_result["raw_detections"] = [
                d for d in structured_result.get("raw_detections", [])
                if d.get("class_name") not in TABLE_CLASSES and not (is_inside_table(d["box"]) and d.get("class_name") in ("date", "text"))
            ]
            for row in extracted_courses:
                r_box = row["row_box"]
                ry0, ry1 = r_box[1], r_box[3]
                if row.get("course_code") and col_code_idx is not None and col_code_idx + 1 < len(col_lines):
                    structured_result["raw_detections"].append({
                        "class_name": "course_code", "confidence": 0.90,
                        "box": [tx + col_lines[col_code_idx], ry0, tx + col_lines[col_code_idx + 1], ry1],
                        "text": row["course_code"]
                    })
                if row.get("course_name") and col_name_idx is not None and col_name_idx + 1 < len(col_lines):
                    structured_result["raw_detections"].append({
                        "class_name": "course_name", "confidence": 0.90,
                        "box": [tx + col_lines[col_name_idx], ry0, tx + col_lines[col_name_idx + 1], ry1],
                        "text": row["course_name"]
                    })
                if row.get("credits") and col_cred_idx is not None and col_cred_idx + 1 < len(col_lines):
                    structured_result["raw_detections"].append({
                        "class_name": "credits", "confidence": 0.90,
                        "box": [tx + col_lines[col_cred_idx], ry0, tx + col_lines[col_cred_idx + 1], ry1],
                        "text": row["credits"]
                    })
                if row.get("score") and col_score_idx is not None and col_score_idx + 1 < len(col_lines):
                    structured_result["raw_detections"].append({
                        "class_name": "score", "confidence": 0.90,
                        "box": [tx + col_lines[col_score_idx], ry0, tx + col_lines[col_score_idx + 1], ry1],
                        "text": row["score"]
                    })
            structured_result["courses_table"] = extracted_courses
        return ty + th

    def create_visualization(self, img_pil, structured_result, show_labels=True, show_banner=True):
        """Tạo ảnh PIL trực quan hóa bounding boxes, nhãn trường và banner thẩm định."""
        vis_img = img_pil.copy()
        draw = ImageDraw.Draw(vis_img)
        try:
            font_small = ImageFont.truetype("arial.ttf", 13)
            font_big = ImageFont.truetype("arialbd.ttf", 26)
        except Exception:
            font_small = font_big = ImageFont.load_default()

        for item in structured_result.get("raw_detections", []):
            bx0, by0, bx1, by1 = item["box"]
            c_name = item["class_name"]
            txt = item.get("text", "")
            conf = item.get("confidence", 0)
            color = CLASS_COLORS.get(CLASS_MAP.get(c_name, 0), (0, 120, 255))
            draw.rectangle([bx0, by0, bx1, by1], outline=color, width=2)
            if show_labels:
                if c_name in TABLE_CLASSES:
                    label_display = c_name
                elif c_name in ("stamp", "signature"):
                    label_display = f"{c_name} {conf:.2f}"
                else:
                    label_display = f"{c_name} {conf:.2f}"

                tag_y = max(2, by0 - 15) if by0 >= 17 else by1 + 2
                tag_w = len(label_display) * 7 + 6
                draw.rectangle([bx0, tag_y, bx0 + tag_w, tag_y + 14], fill=color)
                draw.text((bx0 + 3, tag_y), label_display, fill=(255, 255, 255), font=font_small)

        report = structured_result.get("validation")
        if report and show_banner:
            color = STATUS_COLORS.get(report.get("status", "WARNING"), (0, 0, 0))
            banner = (f"{report.get('status', '')} | {report.get('form_type', '')} | "
                      f"Compliance: {report.get('compliance_score', 0)}% | "
                      f"Lỗi: {report.get('summary', {}).get('errors', 0)} | "
                      f"Cảnh báo: {report.get('summary', {}).get('warnings', 0)}")
            draw.rectangle([0, 0, vis_img.size[0], 48], fill=color)
            draw.text((12, 10), banner, fill=(255, 255, 255), font=font_big)

        return vis_img

    def _save_visualization(self, img_pil, structured_result, save_visualization_path):
        vis_img = self.create_visualization(img_pil, structured_result)
        parent_dir = os.path.dirname(save_visualization_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        vis_img.save(save_visualization_path)
        print(f"🖼️ Đã lưu ảnh trực quan hóa: {save_visualization_path}")


def main():
    parser = argparse.ArgumentParser(description="HCMUTE DocAI – Nhận dạng, trích xuất & thẩm định đơn học vụ")
    parser.add_argument("image", nargs="?", help="Đường dẫn ảnh tài liệu (scan/camera)")
    parser.add_argument("--form-type", default=None, help="Loại mẫu, vd T05_BAO_LUU (mặc định: tự suy luận)")
    parser.add_argument("--yolo", default=None, help="Trọng số YOLO (mặc định: best.pt fine-tuned)")
    parser.add_argument("--ocr-weights", default=None, help="Trọng số VietOCR (mặc định: bản fine-tuned)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--vis", default=None, help="Lưu ảnh trực quan hóa")
    parser.add_argument("--out", default=None, help="Lưu kết quả JSON")
    args = parser.parse_args()

    engine = DocumentLayoutAndOCR(yolo_model_path=args.yolo, ocr_weights_path=args.ocr_weights,
                                  device=args.device, conf_threshold=args.conf)
    if not args.image:
        print("Inference Engine đã sẵn sàng nhận ảnh! (truyền đường dẫn ảnh để chạy)")
        return

    result = engine.process_document(args.image, save_visualization_path=args.vis, form_type=args.form_type)
    print(f"\n📄 {result['document_file']}: {len(result['raw_detections'])} vùng, "
          f"{len(result['courses_table'])} dòng học phần")
    for k, v in result["entities"].items():
        print(f"   {k}: {v}")
    if "validation" in result:
        print("\n" + format_report(result["validation"]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 Đã lưu JSON: {args.out}")


if __name__ == "__main__":
    main()
