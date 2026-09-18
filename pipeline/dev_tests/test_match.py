# -*- coding: utf-8 -*-
import os, sys, cv2, json, re, numpy as np
from PIL import Image, ImageOps
sys.stdout.reconfigure(encoding='utf-8')

img_dir = r'c:\Users\ztung\Downloads\TLCN\dataset_anh_chup_thuc_te'
manifest_path = r'print_dataset_100\dataset_manifest.json'

with open(manifest_path, 'r', encoding='utf-8') as f:
    mf = json.load(f)

mssv_map = {s['student_id']: s for s in mf['samples'] if 'student_id' in s}
name_map = {s['student_name']: s for s in mf['samples'] if 'student_name' in s}
print(f'Indexed {len(mssv_map)} samples in manifest')

test_files = ['IMG_2535.JPG', 'IMG_2536.JPG', 'IMG_2537.JPG', 'IMG_2538.JPG', 'IMG_2539.JPG', 'IMG_2540.JPG', 'IMG_2550.JPG']

import easyocr
reader = easyocr.Reader(['vi', 'en'], gpu=False)

for tf in test_files:
    fp = os.path.join(img_dir, tf)
    if os.path.exists(fp):
        im = Image.open(fp)
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        # Crop header and student info (y: 10% to 35%)
        crop_info = im.crop((int(w * 0.05), int(h * 0.10), int(w * 0.95), int(h * 0.40)))
        crop_np = np.array(crop_info)
        scale_r = 1000.0 / crop_np.shape[1]
        small_crop = cv2.resize(crop_np, (1000, int(crop_np.shape[0] * scale_r)))
        
        results = reader.readtext(small_crop, detail=0)
        text = " ".join(results)
        found_mssvs = re.findall(r'\b(2[0-6]\d{6})\b', text)
        
        matched_s = None
        if found_mssvs:
            for m in found_mssvs:
                if m in mssv_map:
                    matched_s = mssv_map[m]
                    break
                    
        if not matched_s:
            # Try matching student name
            for sname, s_obj in name_map.items():
                if sname in text:
                    matched_s = s_obj
                    break
        
        if matched_s:
            print(f'{tf} -> MATCHED: {matched_s["sample_id"]} | MSSV: {matched_s["student_id"]} | SV: {matched_s["student_name"]}')
        else:
            print(f'{tf} -> Not matched directly. Found MSSVs: {found_mssvs}')
