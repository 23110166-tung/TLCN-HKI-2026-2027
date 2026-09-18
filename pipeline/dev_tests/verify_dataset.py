# -*- coding: utf-8 -*-
"""
Verification script for generated YOLOv11 dataset
"""
import os, sys, yaml
sys.stdout.reconfigure(encoding='utf-8')

dataset_root = r"dataset_yolo"
with open(os.path.join(dataset_root, "data.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

class_names = config["names"]
print("=" * 70)
print("📊 BÁO CÁO KIỂM TRA DATASET YOLOV11 - MẪU 1: ĐIỂM I")
print("=" * 70)

class_counts = {c: 0 for c in class_names.values()}
total_images = 0
total_labels = 0

for split in ["train", "val", "test"]:
    img_dir = os.path.join(dataset_root, "images", split)
    lbl_dir = os.path.join(dataset_root, "labels", split)
    vis_dir = os.path.join(dataset_root, "visualizations", split)
    
    imgs = sorted(os.listdir(img_dir)) if os.path.exists(img_dir) else []
    lbls = sorted(os.listdir(lbl_dir)) if os.path.exists(lbl_dir) else []
    viss = sorted(os.listdir(vis_dir)) if os.path.exists(vis_dir) else []
    
    print(f"\n📁 Split [{split.upper()}]:")
    print(f"   - Số ảnh scan (.png):       {len(imgs)}")
    print(f"   - Số file nhãn YOLO (.txt): {len(lbls)}")
    print(f"   - Số ảnh visualization:     {len(viss)}")
    
    total_images += len(imgs)
    total_labels += len(lbls)
    
    split_boxes = 0
    for lbl in lbls:
        with open(os.path.join(lbl_dir, lbl), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                cname = class_names[cls_id]
                class_counts[cname] += 1
                split_boxes += 1
                
    print(f"   - Tổng bounding boxes [{split}]: {split_boxes}")

print("\n" + "=" * 70)
print("🏷️ THỐNG KÊ CHI TIẾT SỐ LƯỢNG BOUNDING BOX THEO TỪNG CLASS:")
print("=" * 70)
for idx, (cname, cnt) in enumerate(class_counts.items()):
    if cnt > 0:
        print(f"   [{idx:02d}] {cname:<18}: {cnt:>5} boxes")

print("=" * 70)
print(f"✅ TỔNG CỘNG: {total_images} ảnh, {total_labels} file nhãn, {sum(class_counts.values())} bounding boxes.")
print("=" * 70)
