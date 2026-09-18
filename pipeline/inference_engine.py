# -*- coding: utf-8 -*-
"""
End-to-End Inference Engine for HCMUTE Academic Documents
Combines YOLOv11 Layout Detection + VietOCR Text Recognition.
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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

class DocumentLayoutAndOCR:
    """
    End-to-End Pipeline:
    Input Image -> YOLOv11 Layout Detection -> Text Crops -> VietOCR Recognition -> Structured JSON
    """
    
    def __init__(
        self,
        yolo_model_path=None,
        ocr_model_name="vgg_transformer",
        device="cpu",
        conf_threshold=0.35
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.yolo_model = None
        self.ocr_predictor = None
        self.ocr_engine_type = None
        
        # 1. Load YOLO Model
        if yolo_model_path and os.path.exists(yolo_model_path):
            print(f"📦 Đang tải mô hình YOLO Layout từ: {yolo_model_path}")
            self.yolo_model = YOLO(yolo_model_path)
        elif YOLO is not None:
            # Fallback to standard pretrained yolo11n
            print("⚠️ Không tìm thấy trọng số fine-tuned, đang khởi tạo YOLO11n mặc định...")
            self.yolo_model = YOLO("yolo11n.pt")
            
        # 2. Load VietOCR Predictor
        try:
            from vietocr.tool.predictor import Predictor
            from vietocr.tool.config import Cfg
            
            cfg = Cfg.load_config_from_name(ocr_model_name)
            cfg['device'] = self.device
            cfg['prediction_chk'] = None
            self.ocr_predictor = Predictor(cfg)
            self.ocr_engine_type = "vietocr"
            print(f"✅ Đã tải mô hình VietOCR ({ocr_model_name}) trên {device}")
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
        """Runs OCR on a single PIL image crop."""
        if crop_pil is None or crop_pil.size[0] < 4 or crop_pil.size[1] < 4:
            return ""
            
        if self.ocr_engine_type == "vietocr":
            try:
                text = self.ocr_predictor.predict(crop_pil)
                return text.strip()
            except Exception as err:
                return ""
        elif self.ocr_engine_type == "easyocr":
            try:
                crop_np = np.array(crop_pil)
                results = self.ocr_predictor.readtext(crop_np, detail=0)
                return " ".join(results).strip()
            except Exception:
                return ""
        return ""

    def process_document(self, image_path, save_visualization_path=None):
        """
        Processes a full academic document page and extracts structured data.
        Returns a dictionary containing entities, verification flags, and raw detections.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")

        img_pil = Image.open(image_path).convert("RGB")
        w_img, h_img = img_pil.size
        img_np = np.array(img_pil)

        structured_result = {
            "document_file": os.path.basename(image_path),
            "image_size": {"width": w_img, "height": h_img},
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

        if self.yolo_model is None:
            return structured_result

        # Run YOLO Inference
        yolo_results = self.yolo_model(image_path, conf=self.conf_threshold, imgsz=1024, verbose=False)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        classes = yolo_results.boxes.cls.cpu().numpy().astype(int)
        confidences = yolo_results.boxes.conf.cpu().numpy()

        detected_crops = []

        # Sort detections top-to-bottom for reading order
        detections = []
        for box, cls_idx, score in zip(boxes, classes, confidences):
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
            detections.append({
                "box": box,
                "class_id": cls_idx,
                "class_name": cls_name,
                "score": float(score)
            })
        detections.sort(key=lambda d: (d["box"][1], d["box"][0]))

        current_course_row = {}

        for d in detections:
            box = d["box"]
            cls_name = d["class_name"]
            score = d["score"]

            x0, y0, x1, y1 = map(int, box)
            
            # Check Stamps & Signatures
            if cls_name == "stamp":
                structured_result["verification"]["has_stamp"] = True
                structured_result["verification"]["stamp_count"] += 1
                structured_result["raw_detections"].append({
                    "class_name": cls_name,
                    "confidence": round(score, 3),
                    "box": [x0, y0, x1, y1],
                    "text": "[MỘC DẤU ĐỎ]"
                })
                continue
            elif cls_name == "signature":
                structured_result["verification"]["has_signature"] = True
                structured_result["verification"]["signature_count"] += 1
                structured_result["raw_detections"].append({
                    "class_name": cls_name,
                    "confidence": round(score, 3),
                    "box": [x0, y0, x1, y1],
                    "text": "[CHỮ KÝ]"
                })
                continue

            # Text Entities: Crop with padding
            pad_x = 3
            pad_y = 3
            cx0 = max(0, x0 - pad_x)
            cy0 = max(0, y0 - pad_y)
            cx1 = min(w_img, x1 + pad_x)
            cy1 = min(h_img, y1 + pad_y)

            crop = img_pil.crop((cx0, cy0, cx1, cy1))
            rec_text = self._ocr_crop(crop)

            d_record = {
                "class_name": cls_name,
                "confidence": round(score, 3),
                "box": [x0, y0, x1, y1],
                "text": rec_text
            }
            structured_result["raw_detections"].append(d_record)
            detected_crops.append(d_record)

            # Map to semantic fields
            if cls_name in ["student_name", "student_id", "class_name", "faculty", "major", "dob", "phone", "email", "date", "reason"]:
                if cls_name not in structured_result["entities"]:
                    structured_result["entities"][cls_name] = rec_text
                else:
                    # If multiple (e.g. major 1, major 2 in Song bằng)
                    if isinstance(structured_result["entities"][cls_name], list):
                        structured_result["entities"][cls_name].append(rec_text)
                    else:
                        structured_result["entities"][cls_name] = [structured_result["entities"][cls_name], rec_text]

            # Course Table fields
            if cls_name in ["course_code", "course_name", "credits", "score"]:
                if cls_name == "course_code" and current_course_row:
                    structured_result["courses_table"].append(current_course_row)
                    current_course_row = {}
                current_course_row[cls_name] = rec_text

        if current_course_row:
            structured_result["courses_table"].append(current_course_row)

        # 3. Optional Visualization
        if save_visualization_path:
            vis_img = img_pil.copy()
            draw = ImageDraw.Draw(vis_img)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except Exception:
                font = ImageFont.load_default()

            for item in structured_result["raw_detections"]:
                bx0, by0, bx1, by1 = item["box"]
                c_name = item["class_name"]
                txt = item["text"]
                cls_id = CLASS_MAP.get(c_name, 0)
                color = CLASS_COLORS.get(cls_id, (0, 120, 255))

                draw.rectangle([bx0, by0, bx1, by1], outline=color, width=3)
                label_display = f"{c_name}: {txt[:18]}" if txt else c_name
                draw.text((bx0, max(0, by0 - 18)), label_display, fill=color, font=font)

            parent_dir = os.path.dirname(save_visualization_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            vis_img.save(save_visualization_path)
            print(f"🖼️ Đã lưu ảnh trực quan hóa: {save_visualization_path}")

        return structured_result


if __name__ == "__main__":
    print("🚀 Test Inference Engine...")
    engine = DocumentLayoutAndOCR(device="cpu")
    print("Inference Engine đã sẵn sàng nhận ảnh!")
