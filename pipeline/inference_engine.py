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
import argparse
from PIL import Image, ImageDraw, ImageFont
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

# Trọng số mặc định (tải về từ Kaggle – không commit lên GitHub do > 100MB)
DOMAIN_ADAPTED_YOLO_WEIGHTS = os.path.join(
    BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts", "yolo11_domain_adapted_best.pt"
)
SYNTHETIC_YOLO_WEIGHTS = os.path.join(
    BASE_DIR, "HCMUTE_DocAI_YOLOv11_VietOCR_Artifacts", "runs", "detect",
    "hcmute_yolo11_layout", "weights", "best.pt"
)
DEFAULT_YOLO_WEIGHTS = (
    DOMAIN_ADAPTED_YOLO_WEIGHTS if os.path.exists(DOMAIN_ADAPTED_YOLO_WEIGHTS)
    else SYNTHETIC_YOLO_WEIGHTS
)
DEFAULT_OCR_WEIGHTS = os.path.join(
    BASE_DIR, "HCMUTE_VietOCR_Finetuned_Artifacts", "vietocr_hcmute_finetuned.pth"
)

ENTITY_CLASSES = ["student_name", "student_id", "class_name", "faculty", "major", "dob", "pob",
                  "phone", "email", "date", "reason", "degree_info"]
TABLE_CLASSES = ["course_code", "course_name", "credits", "score"]

STATUS_COLORS = {"VALID": (40, 167, 69), "WARNING": (230, 150, 0), "INVALID": (220, 53, 69)}


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
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.yolo_model = None
        self.ocr_predictor = None
        self.ocr_engine_type = None
        self.ocr_weights_used = None
        self.rule_engine = rule_engine or AcademicRuleEngine()

        # 1. Load YOLO Model (ưu tiên trọng số fine-tuned của đề tài)
        yolo_model_path = yolo_model_path or DEFAULT_YOLO_WEIGHTS
        if YOLO is None:
            print("⚠️ Chưa cài ultralytics – bỏ qua bước Layout Detection.")
        elif os.path.exists(yolo_model_path):
            print(f"📦 Đang tải mô hình YOLO Layout từ: {yolo_model_path}")
            self.yolo_model = YOLO(yolo_model_path)
        else:
            print("⚠️ Không tìm thấy trọng số fine-tuned, đang khởi tạo YOLO11n mặc định...")
            self.yolo_model = YOLO("yolo11n.pt")

        # 2. Load VietOCR Predictor (ưu tiên trọng số fine-tuned – RQ1: EM 51.67% → 69.78%)
        ocr_weights_path = ocr_weights_path or DEFAULT_OCR_WEIGHTS
        try:
            from vietocr.tool.predictor import Predictor
            from vietocr.tool.config import Cfg

            cfg = Cfg.load_config_from_name(ocr_model_name)
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

        table = []
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
                entry["scores"] = scores
                entry["score"] = " | ".join(scores)
            if not entry:
                continue
            entry["row_box"] = [min(r["box"][0] for r in row), min(r["box"][1] for r in row),
                                max(r["box"][2] for r in row), max(r["box"][3] for r in row)]
            if confs:
                entry["min_ocr_confidence"] = round(min(confs), 3)
            table.append(entry)
        return table

    def process_document(self, image_path, save_visualization_path=None, form_type=None, validate=True):
        """
        Processes a full academic document page and extracts structured data.
        Returns a dictionary containing entities, course table, verification flags,
        raw detections and (optionally) the rule-engine validation report.

        form_type: mã loại biểu mẫu (vd 'T05_BAO_LUU') hoặc từ khóa; None = tự suy luận.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")

        img_pil = Image.open(image_path).convert("RGB")
        w_img, h_img = img_pil.size

        structured_result = {
            "document_file": os.path.basename(image_path),
            "image_size": {"width": w_img, "height": h_img},
            "models": {"ocr_engine": self.ocr_engine_type, "ocr_weights": self.ocr_weights_used},
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

        if self.yolo_model is not None:
            self._run_detection_and_ocr(image_path, img_pil, structured_result)

        # Rule Engine: kiểm tra tính hợp lệ nghiệp vụ
        if validate and self.rule_engine is not None:
            structured_result["validation"] = self.rule_engine.validate(structured_result, form_type=form_type)

        if save_visualization_path:
            self._save_visualization(img_pil, structured_result, save_visualization_path)

        return structured_result

    def _run_detection_and_ocr(self, image_path, img_pil, structured_result):
        w_img, h_img = img_pil.size
        yolo_results = self.yolo_model(image_path, conf=self.conf_threshold, imgsz=1024, verbose=False)[0]
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

        for d in detections:
            cls_name, score = d["class_name"], d["score"]
            x0, y0, x1, y1 = map(int, d["box"])

            # Check Stamps & Signatures (không cần OCR)
            if cls_name in ("stamp", "signature"):
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

            # Text Entities: Crop with padding
            pad = 3
            crop = img_pil.crop((max(0, x0 - pad), max(0, y0 - pad),
                                 min(w_img, x1 + pad), min(h_img, y1 + pad)))
            rec_text, ocr_conf = self._ocr_crop(crop)

            d_record = {
                "class_name": cls_name,
                "confidence": round(score, 3),
                "ocr_confidence": round(ocr_conf, 3) if ocr_conf is not None else None,
                "box": [x0, y0, x1, y1],
                "text": rec_text
            }
            structured_result["raw_detections"].append(d_record)

            # Map to semantic fields
            if cls_name in ENTITY_CLASSES and rec_text:
                ents = structured_result["entities"]
                if cls_name not in ents:
                    ents[cls_name] = rec_text
                elif isinstance(ents[cls_name], list):
                    ents[cls_name].append(rec_text)
                else:
                    # If multiple (e.g. major 1, major 2 in Song bằng)
                    ents[cls_name] = [ents[cls_name], rec_text]

            # Course Table fields
            if cls_name in TABLE_CLASSES:
                table_records.append(d_record)

        structured_result["courses_table"] = self._group_table_rows(table_records)

    def _save_visualization(self, img_pil, structured_result, save_visualization_path):
        vis_img = img_pil.copy()
        draw = ImageDraw.Draw(vis_img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_big = ImageFont.truetype("arialbd.ttf", 28)
        except Exception:
            font = font_big = ImageFont.load_default()

        for item in structured_result["raw_detections"]:
            bx0, by0, bx1, by1 = item["box"]
            c_name = item["class_name"]
            txt = item["text"]
            color = CLASS_COLORS.get(CLASS_MAP.get(c_name, 0), (0, 120, 255))
            draw.rectangle([bx0, by0, bx1, by1], outline=color, width=3)
            label_display = f"{c_name}: {txt[:18]}" if txt else c_name
            draw.text((bx0, max(0, by0 - 18)), label_display, fill=color, font=font)

        report = structured_result.get("validation")
        if report:
            color = STATUS_COLORS.get(report["status"], (0, 0, 0))
            banner = (f"{report['status']} | {report['form_type']} | "
                      f"Compliance {report['compliance_score']}% | "
                      f"E:{report['summary']['errors']} W:{report['summary']['warnings']}")
            draw.rectangle([0, 0, vis_img.size[0], 44], fill=color)
            draw.text((12, 8), banner, fill=(255, 255, 255), font=font_big)

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
