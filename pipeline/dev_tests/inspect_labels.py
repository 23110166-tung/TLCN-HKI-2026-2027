# -*- coding: utf-8 -*-
import fitz, sys
sys.stdout.reconfigure(encoding='utf-8')

doc = fitz.open(r'dataset_yolo/images/train/diem_i_sample_001.png')
print(f"Image size: {doc[0].rect}")

# Let's inspect where the labels are in diem_i_sample_001.txt
with open(r'dataset_yolo/labels/train/diem_i_sample_001.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for l in lines:
    parts = l.strip().split()
    cls_id = int(parts[0])
    if cls_id in [16, 17]: # signature or stamp
        print(f"Class {cls_id} (16=sig, 17=stamp): {parts}")
