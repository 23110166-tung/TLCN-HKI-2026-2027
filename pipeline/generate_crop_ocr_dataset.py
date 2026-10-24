# -*- coding: utf-8 -*-
"""
Automated Text Crop Extraction Pipeline for VietOCR Fine-tuning (HCMUTE DocAI).
Extracts 3,000–5,000 high-quality text crops from:
  1. Real-world smartphone photos (dataset_yolo_real_world) - with perspective rectification & printer noise
  2. Printed academic vector PDFs (print_dataset_100) - high-resolution 300 DPI renders
  
Outputs VietOCR-standard annotations:
  - dataset_ocr/train_annotation.txt
  - dataset_ocr/val_annotation.txt
  - dataset_ocr/test_annotation.txt
  - dataset_ocr/vocab.txt
  - dataset_ocr/dataset_metadata.json
"""

import os
import sys
import re
import json
import random
import unicodedata
from pathlib import Path
import numpy as np
from PIL import Image

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
    import fitz # PyMuPDF
except ImportError:
    fitz = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline.ontology import CLASS_MAP, CLASS_NAMES

# Non-text classes that should not be used for OCR fine-tuning
NON_TEXT_CLASSES = {"stamp", "signature"}

def clean_ocr_text(text: str) -> str:
    """Normalizes Unicode (NFC), strips unwanted surrounding whitespace, collapses internal whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    # Replace multiple spaces/newlines/tabs with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_valid_text_crop(crop_img, text: str, min_w=14, min_h=10, min_len=1) -> bool:
    """Validates that a crop has sufficient image dimension and valid text label."""
    if crop_img is None:
        return False
    h, w = crop_img.shape[:2]
    if w < min_w or h < min_h:
        return False
    if len(text.strip()) < min_len:
        return False
    # Avoid crops with zero contrast (solid white or black)
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if len(crop_img.shape) == 3 else crop_img
    if np.std(gray) < 5.0:
        return False
    return True

class CropOCRExtractor:
    def __init__(
        self,
        base_dir=BASE_DIR,
        output_dir=None,
        padding=4,
        split_ratios=(0.80, 0.10, 0.10),
        random_seed=42
    ):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir) if output_dir else self.base_dir / "dataset_ocr"
        self.padding = padding
        self.split_ratios = split_ratios
        self.random_seed = random_seed
        random.seed(random_seed)
        
        self.crops_dir = self.output_dir / "crops"
        for s in ["train", "val", "test"]:
            (self.crops_dir / s).mkdir(parents=True, exist_ok=True)
            
        self.real_manifest_path = self.base_dir / "dataset_yolo_real_world" / "real_world_manifest.json"
        self.print_manifest_path = self.base_dir / "print_dataset_100" / "dataset_manifest.json"

    def load_manifests(self):
        """Loads both print_dataset_100 and real_world_manifest metadata."""
        if not self.print_manifest_path.exists():
            raise FileNotFoundError(f"Không tìm thấy manifest in: {self.print_manifest_path}")
        with open(self.print_manifest_path, "r", encoding="utf-8") as f:
            print_manifest = json.load(f)

        real_manifest = None
        if self.real_manifest_path.exists():
            with open(self.real_manifest_path, "r", encoding="utf-8") as f:
                real_manifest = json.load(f)

        return print_manifest, real_manifest

    def partition_documents(self, print_manifest):
        """
        Partitions documents by sample_id to prevent data leakage across splits.
        E.g. PRINT_001 crops will strictly belong to Train, not across Train and Val.
        """
        sample_ids = [s["sample_id"] for s in print_manifest["samples"]]
        # Shuffle deterministically
        shuffled = sample_ids.copy()
        rng = random.Random(self.random_seed)
        rng.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(n_total * self.split_ratios[0])
        n_val = int(n_total * self.split_ratios[1])
        
        train_ids = set(shuffled[:n_train])
        val_ids = set(shuffled[n_train:n_train + n_val])
        test_ids = set(shuffled[n_train + n_val:])
        
        print(f"📊 Phân chia tập dữ liệu (theo Document ID để chống rò rỉ dữ liệu):")
        print(f"   - Train : {len(train_ids)} tài liệu ({len(train_ids)/n_total*100:.1f}%)")
        print(f"   - Val   : {len(val_ids)} tài liệu ({len(val_ids)/n_total*100:.1f}%)")
        print(f"   - Test  : {len(test_ids)} tài liệu ({len(test_ids)/n_total*100:.1f}%)")
        
        doc_split_map = {}
        for sid in train_ids:
            doc_split_map[sid] = "train"
        for sid in val_ids:
            doc_split_map[sid] = "val"
        for sid in test_ids:
            doc_split_map[sid] = "test"
            
        return doc_split_map

    def extract_from_clean_pdfs(self, print_manifest, doc_split_map):
        """
        Renders clean vector PDFs from print_dataset_100 at 300 DPI,
        extracts text bounding boxes from recorded fields and saves OCR crops.
        """
        print("\n" + "=" * 80)
        print("📄 [BƯỚC 1/2] TRÍCH XUẤT TEXT CROPS TỪ VECTOR PDFS (print_dataset_100)...")
        print("=" * 80)
        
        crops_record = []
        dpi = 300
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        samples = print_manifest["samples"]
        for idx, s in enumerate(samples, 1):
            sample_id = s["sample_id"]
            split = doc_split_map.get(sample_id, "train")
            
            # Find PDF file
            pdf_path = self.base_dir / "print_dataset_100" / "vector_pdf_cache" / f"{sample_id}.pdf"
            if not pdf_path.exists():
                pdf_path = self.base_dir / "print_dataset_100" / s.get("pdf_file", "")
            if not pdf_path.exists():
                continue
                
            doc = fitz.open(str(pdf_path))
            page = doc[0]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                
            cur_h, cur_w = img_bgr.shape[:2]
            
            # 1. Extract from recorded fields
            for f_idx, field in enumerate(s.get("fields_recorded", []), 1):
                raw_text = field.get("text", "")
                text = clean_ocr_text(raw_text)
                cls_name = field.get("class_name", "text")
                
                if cls_name in NON_TEXT_CLASSES or not text:
                    continue
                    
                rects = page.search_for(raw_text.strip())
                if not rects and len(raw_text) > 20:
                    rects = page.search_for(raw_text[:15])
                    
                for r_idx, r in enumerate(rects):
                    bx0 = int(r.x0 * zoom) - self.padding
                    by0 = int(r.y0 * zoom) - self.padding
                    bx1 = int(r.x1 * zoom) + self.padding
                    by1 = int(r.y1 * zoom) + self.padding
                    
                    bx0 = max(0, min(cur_w - 1, bx0))
                    by0 = max(0, min(cur_h - 1, by0))
                    bx1 = max(bx0 + 2, min(cur_w, bx1))
                    by1 = max(by0 + 2, min(cur_h, by1))
                    
                    crop_patch = img_bgr[by0:by1, bx0:bx1]
                    if is_valid_text_crop(crop_patch, text):
                        crop_filename = f"crop_clean_{sample_id}_{f_idx:03d}_{r_idx}_{cls_name}.png"
                        crop_rel_path = f"crops/{split}/{crop_filename}"
                        crop_abs_path = self.crops_dir / split / crop_filename
                        
                        cv2.imwrite(str(crop_abs_path), crop_patch)
                        crops_record.append({
                            "rel_path": crop_rel_path,
                            "text": text,
                            "split": split,
                            "source": "clean_pdf",
                            "sample_id": sample_id,
                            "class_name": cls_name,
                            "width": crop_patch.shape[1],
                            "height": crop_patch.shape[0]
                        })
            doc.close()
            
            if idx % 20 == 0 or idx == len(samples):
                print(f"   [{idx:03d}/{len(samples):03d}] Đã xử lý {sample_id} ({s.get('document_type', '')[:22]}) -> Tích lũy {len(crops_record)} clean crops")
                
        print(f"✅ Hoàn thành Bước 1: Trích xuất {len(crops_record)} crops từ clean PDFs.")
        return crops_record

    def extract_from_real_camera_photos(self, real_manifest, print_manifest, doc_split_map):
        """
        Extracts crops from real-world smartphone photos in dataset_yolo_real_world.
        Uses matched sample annotations and offset alignment to precisely locate text boxes.
        """
        if not real_manifest or "samples" not in real_manifest:
            print("⚠️ Không có real_world_manifest.json, bỏ qua Bước 2.")
            return []
            
        print("\n" + "=" * 80)
        print("📸 [BƯỚC 2/2] TRÍCH XUẤT TEXT CROPS TỪ ẢNH CHỤP CAMERA THỰC TẾ (dataset_yolo_real_world)...")
        print("=" * 80)
        
        sample_map = {s["sample_id"]: s for s in print_manifest["samples"]}
        crops_record = []
        real_samples = real_manifest["samples"]
        
        for idx, item in enumerate(real_samples, 1):
            img_name = item["yolo_image"]
            matched_id = item["matched_sample_id"]
            split = doc_split_map.get(matched_id, "train")
            sample_obj = sample_map.get(matched_id, {})
            
            img_path = self.base_dir / "dataset_yolo_real_world" / "images" / img_name
            if not img_path.exists():
                continue
                
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            cur_h, cur_w = img_bgr.shape[:2]
            
            # Locate clean PDF for coordinate lookup
            pdf_path = self.base_dir / "print_dataset_100" / "vector_pdf_cache" / f"{matched_id}.pdf"
            if not pdf_path.exists():
                pdf_path = self.base_dir / "print_dataset_100" / sample_obj.get("pdf_file", "")
            if not pdf_path.exists():
                continue
                
            doc = fitz.open(str(pdf_path))
            page = doc[0]
            zoom = 300.0 / 72.0
            
            dx = item.get("offset_dx", 0)
            dy = item.get("offset_dy", 0)
            
            recorded_fields = sample_obj.get("fields_recorded", [])
            for f_idx, field in enumerate(recorded_fields, 1):
                raw_text = field.get("text", "")
                text = clean_ocr_text(raw_text)
                cls_name = field.get("class_name", "text")
                
                if cls_name in NON_TEXT_CLASSES or not text:
                    continue
                    
                rects = page.search_for(raw_text.strip())
                if not rects and len(raw_text) > 20:
                    rects = page.search_for(raw_text[:15])
                    
                for r_idx, r in enumerate(rects):
                    bx0 = int(r.x0 * zoom) + dx - self.padding
                    by0 = int(r.y0 * zoom) + dy - self.padding
                    bx1 = int(r.x1 * zoom) + dx + self.padding
                    by1 = int(r.y1 * zoom) + dy + self.padding
                    
                    bx0 = max(0, min(cur_w - 1, bx0))
                    by0 = max(0, min(cur_h - 1, by0))
                    bx1 = max(bx0 + 2, min(cur_w, bx1))
                    by1 = max(by0 + 2, min(cur_h, by1))
                    
                    crop_patch = img_bgr[by0:by1, bx0:bx1]
                    if is_valid_text_crop(crop_patch, text):
                        crop_filename = f"crop_real_{item['index']:03d}_{matched_id}_{f_idx:03d}_{r_idx}_{cls_name}.png"
                        crop_rel_path = f"crops/{split}/{crop_filename}"
                        crop_abs_path = self.crops_dir / split / crop_filename
                        
                        cv2.imwrite(str(crop_abs_path), crop_patch)
                        crops_record.append({
                            "rel_path": crop_rel_path,
                            "text": text,
                            "split": split,
                            "source": "real_camera",
                            "sample_id": matched_id,
                            "camera_file": item.get("camera_file", ""),
                            "class_name": cls_name,
                            "width": crop_patch.shape[1],
                            "height": crop_patch.shape[0]
                        })
            doc.close()
            
            if idx % 20 == 0 or idx == len(real_samples):
                print(f"   [{idx:03d}/{len(real_samples):03d}] Đã xử lý {img_name} -> Tích lũy {len(crops_record)} real-world crops")
                
        print(f"✅ Hoàn thành Bước 2: Trích xuất {len(crops_record)} crops từ ảnh thực tế.")
        return crops_record

    def generate_annotations_and_metadata(self, all_crops):
        """
        Writes VietOCR annotations files (train, val, test, all) and metadata JSON.
        """
        print("\n" + "=" * 80)
        print("💾 [BƯỚC 3/3] GHI FILE ANNOTATION CHUẨN VIETOCR & METADATA BỘ DỮ LIỆU...")
        print("=" * 80)
        
        split_records = {"train": [], "val": [], "test": []}
        all_lines = []
        vocab_chars = set()
        class_distribution = {}
        source_distribution = {}
        
        for c in all_crops:
            split = c["split"]
            rel_path = c["rel_path"]
            text = c["text"]
            line = f"{rel_path}\t{text}"
            
            split_records[split].append(line)
            all_lines.append(line)
            
            for ch in text:
                vocab_chars.add(ch)
                
            cls_name = c["class_name"]
            class_distribution[cls_name] = class_distribution.get(cls_name, 0) + 1
            
            src = c["source"]
            source_distribution[src] = source_distribution.get(src, 0) + 1
            
        # 1. Write annotation files
        for split in ["train", "val", "test"]:
            ann_path = self.output_dir / f"{split}_annotation.txt"
            with open(ann_path, "w", encoding="utf-8") as f:
                f.write("\n".join(split_records[split]) + "\n")
            print(f"📝 Đã ghi {ann_path.name:<22}: {len(split_records[split]):>5} dòng annotations")
            
        # Full annotation
        all_ann_path = self.output_dir / "all_annotation.txt"
        with open(all_ann_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_lines) + "\n")
        print(f"📝 Đã ghi {all_ann_path.name:<22}: {len(all_lines):>5} dòng annotations")
        
        # 2. Write vocabulary file
        sorted_vocab = sorted(list(vocab_chars))
        vocab_path = self.output_dir / "vocab.txt"
        with open(vocab_path, "w", encoding="utf-8") as f:
            f.write("".join(sorted_vocab))
        print(f"🔤 Đã ghi {vocab_path.name:<22}: {len(sorted_vocab):>5} ký tự độc nhất (kèm đầy đủ dấu tiếng Việt)")
        
        # 3. Write metadata JSON
        metadata = {
            "dataset_name": "HCMUTE Academic Document Text Crops Dataset for VietOCR Fine-tuning",
            "created_date": "2026-10-25",
            "total_crops": len(all_crops),
            "split_counts": {k: len(v) for k, v in split_records.items()},
            "source_distribution": source_distribution,
            "classes_distribution": class_distribution,
            "vocab_size": len(sorted_vocab),
            "vocabulary_preview": "".join(sorted_vocab[:80]) + "...",
            "vietocr_config_hint": {
                "data_root": str(self.output_dir.name),
                "train_annotation": "train_annotation.txt",
                "valid_annotation": "val_annotation.txt",
                "test_annotation": "test_annotation.txt",
                "image_height": 32,
                "image_min_width": 32,
                "image_max_width": 512
            }
        }
        meta_path = self.output_dir / "dataset_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"📊 Đã ghi {meta_path.name:<22}: Cấu hình & thống kê chi tiết")
        
        return metadata

    def run(self):
        start_time = sys.modules["time"].time()
        print("=" * 80)
        print("🚀 BẮT ĐẦU PIPELINE SINH BỘ DỮ LIỆU CROP OCR CHO VIETOCR FINE-TUNING")
        print(f"📂 Thư mục xuất dữ liệu: {self.output_dir}")
        print("=" * 80)
        
        print_manifest, real_manifest = self.load_manifests()
        doc_split_map = self.partition_documents(print_manifest)
        
        clean_crops = self.extract_from_clean_pdfs(print_manifest, doc_split_map)
        real_crops = self.extract_from_real_camera_photos(real_manifest, print_manifest, doc_split_map)
        
        all_crops = clean_crops + real_crops
        print(f"\n🎉 TỔNG SỐ TEXT CROPS HỢP LỆ THU THẬP ĐƯỢC: {len(all_crops)} CROPS!")
        
        metadata = self.generate_annotations_and_metadata(all_crops)
        
        elapsed = sys.modules["time"].time() - start_time
        print("\n" + "=" * 80)
        print(f"✨ HOÀN THÀNH TOÀN BỘ PIPELINE TRONG {elapsed:.1f} GIÂY!")
        print(f"📁 Dữ liệu sẵn sàng tại: {self.output_dir}")
        print("=" * 80)
        return metadata

if __name__ == "__main__":
    import time
    extractor = CropOCRExtractor()
    extractor.run()
