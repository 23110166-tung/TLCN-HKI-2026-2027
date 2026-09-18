# -*- coding: utf-8 -*-
"""
Generate 3 trial sample Academic Transcripts and copy to artifacts for user inspection.
"""

import os
import sys
import shutil
from selenium import webdriver
from selenium.webdriver.edge.options import Options

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline.transcript_generator import generate_transcript_sample_with_driver, EDGE_EXE

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    out_dir = r"C:\Users\ztung\.gemini\antigravity-ide\brain\210da422-759c-4365-bf9f-24500b1323b9"
    temp_dir = os.path.join(out_dir, "sample_transcripts")
    os.makedirs(temp_dir, exist_ok=True)
    
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.binary_location = EDGE_EXE
    
    driver = webdriver.Edge(options=options)
    
    samples = [
        ("Nguyễn Bạch Tùng", "23110088", "AI", "BANG_DIEM_SAMPLE_1_AI"),
        ("Trần Thị Mai Anh", "21115042", "CNTT", "BANG_DIEM_SAMPLE_2_CNTT"),
        ("Lê Hoàng Nam",     "22145108", "DDT",  "BANG_DIEM_SAMPLE_3_DDT")
    ]
    
    print("=" * 80)
    print("🚀 BẮT ĐẦU SINH 3 BẢNG ĐIỂM HỌC TẬP THỬ NGHIỆM")
    print("=" * 80)
    
    results = []
    try:
        for name, mssv, major, out_name in samples:
            res = generate_transcript_sample_with_driver(
                driver=driver,
                student_name=name,
                student_id=mssv,
                major_key=major,
                output_base_name=out_name,
                output_dir=temp_dir
            )
            # Copy visualization and raw image to root artifacts dir
            vis_dst = os.path.join(out_dir, f"{out_name}_annotated.png")
            raw_dst = os.path.join(out_dir, f"{out_name}.png")
            shutil.copy2(res["visualization"], vis_dst)
            shutil.copy2(res["image"], raw_dst)
            
            print(f"✅ Sinh thành công: {name:<20} | MSSV: {mssv} | Ngành: {major:<5} | BBoxes: {res['num_bboxes']} | Size: {res['dimensions']}")
            results.append((out_name, name, mssv, major, res['num_bboxes']))
            
    finally:
        driver.quit()
        
    print("=" * 80)
    print("🎉 HOÀN TẤT SINH 3 BẢNG ĐIỂM MẪU!")
    print("=" * 80)

if __name__ == "__main__":
    main()
