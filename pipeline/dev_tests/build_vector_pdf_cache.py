# -*- coding: utf-8 -*-
import os
import sys
import glob
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline.pdf_converter import PDFConverter

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    src_dir = os.path.join(BASE_DIR, "print_dataset_100", "docx_and_html_sources")
    cache_dir = os.path.join(BASE_DIR, "print_dataset_100", "vector_pdf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    docx_files = sorted(glob.glob(os.path.join(src_dir, "*.docx")))
    print(f"📄 Đang chuyển đổi {len(docx_files)} file docx sang vector PDF...")
    
    conv = PDFConverter()
    for f in docx_files:
        fname = os.path.basename(f)
        parts = fname.split("_")
        sample_id = f"{parts[0]}_{parts[1]}"
        out_pdf = os.path.join(cache_dir, f"{sample_id}.pdf")
        if not os.path.exists(out_pdf):
            conv.convert(f, out_pdf)
    conv.close()
    
    # Copy html pdfs
    for i in range(91, 101):
        sid = f"PRINT_{i:03d}"
        src_p = os.path.join(BASE_DIR, "print_dataset_100", "pdf_individual", f"{sid}.pdf")
        dst_p = os.path.join(cache_dir, f"{sid}.pdf")
        if os.path.exists(src_p) and not os.path.exists(dst_p):
            shutil.copyfile(src_p, dst_p)
            
    cached = glob.glob(os.path.join(cache_dir, "*.pdf"))
    print(f"✅ Đã tạo xong vector PDF cache: {len(cached)} file PDF chuẩn xác.")

if __name__ == "__main__":
    main()
