# -*- coding: utf-8 -*-
"""
Dataset Preparation & Advanced Augmentation Pipeline for Domain Adaptation (YOLOv11).
Tiểu luận chuyên ngành AI - Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh.

Giải quyết Câu hỏi nghiên cứu 2 (RQ2) & Mục 3.1.2 trong Đề cương:
- Phân chia 120 ảnh chụp camera thực tế thành Train / Val / Test strictly theo Document ID (Zero Data Leakage).
- Áp dụng các kỹ thuật Data Augmentation mô phỏng thực tế smartphone:
    1. Biến dạng phối cảnh & góc chụp (Random Perspective / Tilt / Rotation)
    2. Chiếu sáng không đồng đều & bóng đổ (Random Shadows / Contrast / Gamma)
    3. Nhiễu cảm biến & rung tay (Motion Blur / Defocus / GaussNoise)
- Trộn tập Synthetic nhỏ làm Anchor Replay để tránh hiện tượng Catastrophic Forgetting.
- Tự động sinh file cấu hình data.yaml và nén zip dataset sẵn sàng đưa lên Kaggle.
"""

import os
import sys
import cv2
import json
import random
import shutil
import zipfile
import numpy as np
from collections import defaultdict

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
    import albumentations as A
except ImportError:
    raise ImportError("Vui lòng cài đặt albumentations: pip install albumentations")

from pipeline.ontology import CLASS_NAMES, CLASS_MAP

REAL_DATASET_DIR = os.path.join(BASE_DIR, "dataset_yolo_real_world")
SYNTHETIC_DATASET_DIR = os.path.join(BASE_DIR, "dataset_yolo")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_yolo_domain_adaptation")
OUTPUT_ZIP = os.path.join(BASE_DIR, "dataset_yolo_domain_adaptation.zip")


# =====================================================================
# 1. BỘ TĂNG CƯỜNG DỮ LIỆU ALBUMENTATIONS CHO ẢNH CHỤP SMARTPHONE
# =====================================================================

def get_geometric_augmenter():
    """Mô phỏng góc chụp nghiêng của smartphone và độ lệch góc của camera."""
    return A.Compose([
        A.Perspective(scale=(0.02, 0.05), p=0.8),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=4, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255], p=0.7),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.25))


def get_photometric_augmenter():
    """Mô phỏng ánh sáng phòng không đều, đèn vàng/trắng và độ tương phản."""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.2, hue=0.04, p=0.6),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.25))


def get_sensor_blur_augmenter():
    """Mô phỏng rung tay khi bấm nút chụp, out nét nhẹ và nhiễu cảm biến ISO cao."""
    return A.Compose([
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 7), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.Defocus(radius=(1, 3), p=0.3),
        ], p=0.7),
        A.GaussNoise(var_limit=(10.0, 40.0), p=0.4),
        A.ImageCompression(quality_lower=70, quality_upper=95, p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.25))


def apply_realistic_arm_shadow(img_bgr):
    """
    Vẽ bóng đổ đa giác bán trong suốt (semi-transparent shadow) mô phỏng bóng
    cánh tay hoặc điện thoại đè lên mặt giấy khi chụp từ trên xuống.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    # Chọn 3 đến 5 đỉnh cho góc bóng đổ
    num_pts = random.randint(3, 5)
    pts = [[random.randint(0, w), random.randint(0, h)] for _ in range(num_pts)]
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1.0)

    # Làm mờ biên bóng đổ để tạo hiệu ứng penumbra tự nhiên
    k_size = int(max(w, h) * 0.06) | 1  # Đảm bảo số lẻ
    mask = cv2.GaussianBlur(mask, (k_size, k_size), 0)

    # Giảm độ sáng từ 30% đến 55%
    alpha = random.uniform(0.30, 0.55)
    shadow_map = 1.0 - (mask * alpha)
    shadowed = (img_bgr.astype(np.float32) * shadow_map[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)
    return shadowed


def clean_and_deduplicate_yolo_bboxes(lines):
    """
    Chuẩn hóa, lọc nhãn rác và khử trùng lặp (NMS / Deduplication) triệt để:
    1. Giới hạn tọa độ chặt chẽ trong [0.0, 1.0], loại bỏ box dị thường (width, height < 0.002).
    2. Ràng buộc không gian bảng biểu: Các ô bảng điểm (course_code, course_name, credits, score)
       chỉ nằm ở phần thân biểu mẫu (0.18 <= yc <= 0.94), loại bỏ nhãn giả do trùng số trong tiêu đề/ngày tháng.
    3. Giới hạn kích thước ô điểm / tín chỉ: credits (class 11) chỉ có 1 chữ số nên w <= 0.08, h <= 0.06;
       score (class 12) có w <= 0.12, h <= 0.06.
    4. Non-Maximum Suppression (NMS) khử nhãn lặp lại (IoU > 0.35 hoặc IoA > 0.65 cùng lớp).
    5. Khử nhiễu nhãn đơn lẻ (credits/score) bị đè lên các trường thông tin lớn (MSSV, SĐT, Tên, Ngày).
    """
    boxes = []
    for l in lines:
        parts = l.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])

        x0 = max(0.0, min(1.0, xc - w / 2.0))
        x1 = max(0.0, min(1.0, xc + w / 2.0))
        y0 = max(0.0, min(1.0, yc - h / 2.0))
        y1 = max(0.0, min(1.0, yc + h / 2.0))

        bw = x1 - x0
        bh = y1 - y0
        if bw < 0.002 or bh < 0.002:
            continue

        # Ràng buộc không gian cho ô bảng biểu
        if cls_id in [9, 10, 11, 12]:
            if yc < 0.18 or yc > 0.94:
                continue
            if cls_id == 11 and (bw > 0.08 or bh > 0.06):
                continue
            if cls_id == 12 and (bw > 0.12 or bh > 0.06):
                continue

        boxes.append((cls_id, x0, y0, x1, y1, bw, bh))

    # Khử trùng lặp (Deduplication)
    kept = []
    for b in boxes:
        cls_id, x0, y0, x1, y1, bw, bh = b
        area_b = bw * bh
        dup = False
        for k in kept:
            k_cls, kx0, ky0, kx1, ky1, kbw, kbh = k
            area_k = kbw * kbh
            ix0, iy0 = max(x0, kx0), max(y0, ky0)
            ix1, iy1 = min(x1, kx1), min(y1, ky1)
            iw = max(0.0, ix1 - ix0)
            ih = max(0.0, iy1 - iy0)
            inter = iw * ih
            if inter > 0:
                union = area_b + area_k - inter
                iou = inter / union if union > 0 else 0
                ioa_b = inter / area_b if area_b > 0 else 0
                # Trùng lặp cùng lớp
                if cls_id == k_cls and (iou > 0.35 or ioa_b > 0.65):
                    dup = True
                    break
                # Nhãn tín chỉ/điểm số ảo bị lẫn vào trường văn bản (MSSV, Phone, Name, Course)
                if cls_id in [11, 12] and k_cls in [0, 1, 2, 7, 8, 9, 10] and ioa_b > 0.60:
                    dup = True
                    break
        if not dup:
            kept.append(b)

    final_bboxes = []
    final_cats = []
    for cls_id, x0, y0, x1, y1, bw, bh in kept:
        xc = (x0 + x1) / 2.0
        yc = (y0 + y1) / 2.0
        final_bboxes.append([xc, yc, bw, bh])
        final_cats.append(cls_id)
    return final_bboxes, final_cats


def save_yolo_label(out_path, bboxes, category_ids):
    with open(out_path, "w", encoding="utf-8") as f:
        for cat_id, (xc, yc, w, h) in zip(category_ids, bboxes):
            f.write(f"{cat_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# =====================================================================
# 2. PHÂN CHIA DỮ LIỆU ZERO DATA LEAKAGE THEO DOCUMENT ID
# =====================================================================

def partition_manifest(manifest_path, seed=42):
    """
    Phân chia 120 ảnh theo 91 unique Document IDs để đảm bảo văn bản ở Test
    hoàn toàn chưa từng xuất hiện ở Train hoặc Val.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = manifest["samples"]
    doc_to_type = {}
    doc_to_samples = defaultdict(list)

    for s in samples:
        doc_id = s["matched_sample_id"]
        doc_to_type[doc_id] = s["document_type"]
        doc_to_samples[doc_id].append(s)

    type_to_docs = defaultdict(list)
    for doc_id, dtype in doc_to_type.items():
        type_to_docs[dtype].append(doc_id)

    random.seed(seed)
    train_docs, val_docs, test_docs = set(), set(), set()

    # PRINT_100 có tới 17 ảnh chụp đa góc -> gán cố định vào Train
    if "PRINT_100" in doc_to_type:
        train_docs.add("PRINT_100")

    for dtype, docs in sorted(type_to_docs.items()):
        docs_pool = [d for d in docs if d != "PRINT_100"]
        random.shuffle(docs_pool)
        n = len(docs_pool)

        # Phân bổ ~ 15% Test, ~ 15% Val, còn lại là Train
        n_test = max(1, int(round(n * 0.16)))
        n_val = max(1, int(round(n * 0.16)))
        n_train = max(1, n - n_val - n_test)

        test_d = docs_pool[:n_test]
        val_d = docs_pool[n_test:n_test + n_val]
        train_d = docs_pool[n_test + n_val:]

        test_docs.update(test_d)
        val_docs.update(val_d)
        train_docs.update(train_d)

    # Tổng hợp danh sách mẫu cho từng tập
    train_samples = [s for d in train_docs for s in doc_to_samples[d]]
    val_samples = [s for d in val_docs for s in doc_to_samples[d]]
    test_samples = [s for d in test_docs for s in doc_to_samples[d]]

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
        "train_docs": sorted(list(train_docs)),
        "val_docs": sorted(list(val_docs)),
        "test_docs": sorted(list(test_docs)),
    }


# =====================================================================
# 3. CHƯƠNG TRÌNH CHÍNH XỬ LÝ & ĐÓNG GÓI BỘ DỮ LIỆU
# =====================================================================

def prepare_domain_adaptation_dataset(
    real_dataset_dir=REAL_DATASET_DIR,
    synthetic_dataset_dir=SYNTHETIC_DATASET_DIR,
    output_dir=OUTPUT_DIR,
    num_synthetic_anchor=60,
    seed=42
):
    print("=" * 80)
    print("🚀 BẮT ĐẦU CHUẨN BỊ BỘ DỮ LIỆU DOMAIN ADAPTATION (YOLOV11 REAL-WORLD)")
    print("=" * 80)

    manifest_file = os.path.join(real_dataset_dir, "real_world_manifest.json")
    if not os.path.exists(manifest_file):
        raise FileNotFoundError(f"Không tìm thấy manifest tại: {manifest_file}")

    # Tạo cấu trúc thư mục YOLO chuẩn
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # 1. Phân chia tập dữ liệu
    print("📊 Đang phân chia 120 ảnh theo Document ID (Zero Data Leakage)...")
    splits = partition_manifest(manifest_file, seed=seed)

    print(f"   • Train : {len(splits['train']):<3} ảnh ({len(splits['train_docs'])} văn bản gốc)")
    print(f"   • Val   : {len(splits['val']):<3} ảnh ({len(splits['val_docs'])} văn bản gốc)")
    print(f"   • Test  : {len(splits['test']):<3} ảnh ({len(splits['test_docs'])} văn bản gốc)")

    # Khởi tạo các bộ tăng cường
    geo_aug = get_geometric_augmenter()
    photo_aug = get_photometric_augmenter()
    blur_aug = get_sensor_blur_augmenter()

    real_img_dir = os.path.join(real_dataset_dir, "images")
    real_lbl_dir = os.path.join(real_dataset_dir, "labels")

    counts = {"train": 0, "val": 0, "test": 0, "augmented": 0, "synthetic": 0}

    # 2. Xử lý tập Val & Test (LÀM SẠCH NHÃN & KHỬ TRÙNG LẶP PHANTOM)
    for split_name in ["val", "test"]:
        print(f"\n📂 Đang chuẩn bị tập {split_name.upper()} ({len(splits[split_name])} ảnh thực tế)...")
        for s in splits[split_name]:
            img_name = s["yolo_image"]
            lbl_name = s["yolo_label"]

            src_img = os.path.join(real_img_dir, img_name)
            src_lbl = os.path.join(real_lbl_dir, lbl_name)

            dst_img = os.path.join(output_dir, "images", split_name, img_name)
            dst_lbl = os.path.join(output_dir, "labels", split_name, lbl_name)

            shutil.copy2(src_img, dst_img)
            if os.path.exists(src_lbl):
                with open(src_lbl, "r", encoding="utf-8") as f_lbl:
                    raw_lines = f_lbl.readlines()
                c_bboxes, c_cats = clean_and_deduplicate_yolo_bboxes(raw_lines)
                save_yolo_label(dst_lbl, c_bboxes, c_cats)
            counts[split_name] += 1

    # 3. Xử lý tập Train (Ảnh gốc + 3 biến thể Augmentation)
    print(f"\n🎨 Đang tăng cường dữ liệu tập TRAIN ({len(splits['train'])} ảnh gốc)...")
    for idx, s in enumerate(splits["train"], 1):
        img_name = s["yolo_image"]
        lbl_name = s["yolo_label"]
        base_name = os.path.splitext(img_name)[0]

        src_img = os.path.join(real_img_dir, img_name)
        src_lbl = os.path.join(real_lbl_dir, lbl_name)

        if not os.path.exists(src_img) or not os.path.exists(src_lbl):
            continue

        img_bgr = cv2.imread(src_img)
        if img_bgr is None:
            continue

        with open(src_lbl, "r", encoding="utf-8") as f:
            lines = f.readlines()
        bboxes, cat_ids = clean_and_deduplicate_yolo_bboxes(lines)

        # Lưu ảnh gốc vào tập train
        dst_img_orig = os.path.join(output_dir, "images", "train", img_name)
        dst_lbl_orig = os.path.join(output_dir, "labels", "train", lbl_name)
        cv2.imwrite(dst_img_orig, img_bgr)
        save_yolo_label(dst_lbl_orig, bboxes, cat_ids)
        counts["train"] += 1

        # Aug 1: Biến dạng phối cảnh & góc chụp (Geometric)
        try:
            res_geo = geo_aug(image=img_bgr, bboxes=bboxes, category_ids=cat_ids)
            if len(res_geo["bboxes"]) > 0:
                cv2.imwrite(os.path.join(output_dir, "images", "train", f"{base_name}_aug_geo.jpg"), res_geo["image"])
                save_yolo_label(os.path.join(output_dir, "labels", "train", f"{base_name}_aug_geo.txt"),
                                res_geo["bboxes"], res_geo["category_ids"])
                counts["augmented"] += 1
                counts["train"] += 1
        except Exception:
            pass

        # Aug 2: Chiếu sáng không đồng đều + Bóng đổ cánh tay/điện thoại
        try:
            shadowed = apply_realistic_arm_shadow(img_bgr)
            res_light = photo_aug(image=shadowed, bboxes=bboxes, category_ids=cat_ids)
            if len(res_light["bboxes"]) > 0:
                cv2.imwrite(os.path.join(output_dir, "images", "train", f"{base_name}_aug_shadow.jpg"), res_light["image"])
                save_yolo_label(os.path.join(output_dir, "labels", "train", f"{base_name}_aug_shadow.txt"),
                                res_light["bboxes"], res_light["category_ids"])
                counts["augmented"] += 1
                counts["train"] += 1
        except Exception:
            pass

        # Aug 3: Rung tay, mờ nét & nhiễu cảm biến ISO cao
        try:
            res_blur = blur_aug(image=img_bgr, bboxes=bboxes, category_ids=cat_ids)
            if len(res_blur["bboxes"]) > 0:
                cv2.imwrite(os.path.join(output_dir, "images", "train", f"{base_name}_aug_blur.jpg"), res_blur["image"])
                save_yolo_label(os.path.join(output_dir, "labels", "train", f"{base_name}_aug_blur.txt"),
                                res_blur["bboxes"], res_blur["category_ids"])
                counts["augmented"] += 1
                counts["train"] += 1
        except Exception:
            pass

        if idx % 20 == 0 or idx == len(splits["train"]):
            print(f"   [{idx:03d}/{len(splits['train']):03d}] Đã sinh ảnh augmented...")

    # 4. Trộn thêm tập Synthetic Anchor Replay (nếu có)
    syn_train_img_dir = os.path.join(synthetic_dataset_dir, "images", "train")
    syn_train_lbl_dir = os.path.join(synthetic_dataset_dir, "labels", "train")
    if os.path.exists(syn_train_img_dir) and num_synthetic_anchor > 0:
        print(f"\n⚓ Đang bổ sung {num_synthetic_anchor} mẫu Synthetic Anchor Replay để chống quên tri thức cũ...")
        syn_imgs = [f for f in os.listdir(syn_train_img_dir) if f.lower().endswith((".png", ".jpg"))]
        random.shuffle(syn_imgs)
        selected_syn = syn_imgs[:num_synthetic_anchor]

        for s_img in selected_syn:
            base_s = os.path.splitext(s_img)[0]
            s_lbl = base_s + ".txt"
            src_s_img = os.path.join(syn_train_img_dir, s_img)
            src_s_lbl = os.path.join(syn_train_lbl_dir, s_lbl)

            if os.path.exists(src_s_img) and os.path.exists(src_s_lbl):
                dst_s_img = os.path.join(output_dir, "images", "train", f"SYN_{s_img}")
                dst_s_lbl = os.path.join(output_dir, "labels", "train", f"SYN_{s_lbl}")
                shutil.copy2(src_s_img, dst_s_img)
                shutil.copy2(src_s_lbl, dst_s_lbl)
                counts["synthetic"] += 1
                counts["train"] += 1

    # 5. Sinh file cấu hình data.yaml
    yaml_content = f"""# =====================================================================
# YOLOv11 Domain Adaptation Dataset – HCMUTE DocAI (HKI 2026-2027)
# Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh
# =====================================================================
path: {output_dir.replace(os.sep, '/')}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f_y:
        f_y.write(yaml_content)

    # 6. Lưu manifest phân chia để minh bạch khoa học
    manifest_split_path = os.path.join(output_dir, "manifest_split.json")
    with open(manifest_split_path, "w", encoding="utf-8") as f_ms:
        json.dump({
            "dataset_name": "HCMUTE YOLOv11 Domain Adaptation Dataset (Smartphone Camera)",
            "school": "Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh",
            "seed": seed,
            "counts": counts,
            "splits": {
                "train_docs_count": len(splits["train_docs"]),
                "val_docs_count": len(splits["val_docs"]),
                "test_docs_count": len(splits["test_docs"]),
                "train_docs": splits["train_docs"],
                "val_docs": splits["val_docs"],
                "test_docs": splits["test_docs"]
            }
        }, f_ms, ensure_ascii=False, indent=2)

    # 7. Tích hợp trọng số Pretrained Synthetic (Stage 2) vào dataset để Kaggle tự động tìm thấy
    synth_weights_src = os.path.join(
        BASE_DIR, "HCMUTE_DocAI_YOLOv11_VietOCR_Artifacts", "runs", "detect",
        "hcmute_yolo11_layout", "weights", "best.pt"
    )
    if os.path.exists(synth_weights_src):
        dst_weights = os.path.join(output_dir, "pretrained_synthetic_best.pt")
        print(f"📦 Đang tích hợp trọng số Synthetic Pretrained ({os.path.getsize(synth_weights_src)/(1024*1024):.2f} MB) vào dataset...")
        shutil.copy2(synth_weights_src, dst_weights)

    # 8. Đóng gói ZIP dataset
    print(f"\n📦 Đang nén toàn bộ dataset thành: {OUTPUT_ZIP} ...")
    total_files = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, output_dir)
                zipf.write(full_path, arcname=os.path.join("dataset_yolo_domain_adaptation", rel_path))
                total_files += 1

    zip_size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)

    print("=" * 80)
    print("🎉 HOÀN TẤT CHUẨN BỊ BỘ DỮ LIỆU DOMAIN ADAPTATION!")
    print(f"📁 Thư mục dataset : {output_dir}")
    print(f"📦 File Zip đóng gói: {OUTPUT_ZIP} ({zip_size_mb:.2f} MB)")
    print(f"   • Train images  : {counts['train']} (Gốc: {len(splits['train'])}, Aug: {counts['augmented']}, Synthetic: {counts['synthetic']})")
    print(f"   • Val images    : {counts['val']} (100% Real-World Camera)")
    print(f"   • Test images   : {counts['test']} (100% Real-World Camera, Zero Leakage)")
    print(f"⚙️ File cấu hình   : {yaml_path}")
    print("=" * 80)

    return output_dir, OUTPUT_ZIP


if __name__ == "__main__":
    prepare_domain_adaptation_dataset()
