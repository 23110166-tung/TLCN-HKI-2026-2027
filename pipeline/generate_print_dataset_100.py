# -*- coding: utf-8 -*-
"""
Generator for Print-Ready Real-World Benchmark Dataset (100 Samples) - METHOD B.
Embeds authentic blue handwritten signatures and official red circular stamps
directly into all 100 PDF documents so they are 100% ready for color/laser printing.
"""

import os
import sys
import json
import time
import random
import shutil
import subprocess
import fitz # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline.mock_data import generate_mock_student, generate_random_name
from pipeline.docx_filler import fill_document
from pipeline.pdf_converter import PDFConverter
from pipeline.transcript_generator import generate_transcript_html, EDGE_EXE
from pipeline.stamp_generator import (
    create_handwritten_signature,
    create_circular_stamp
)
from pipeline.ontology import CLASS_MAP

TEMPLATES_9 = [
    {
        "type_id": "T01_DIEM_I",
        "name": "Đơn xin nhận và trả điểm I",
        "file": os.path.join(BASE_DIR, "sample", "Đơn xin nhận và trả điểm I.docx"),
        "cohorts": [2023, 2024, 2025],
        "desc": "Sinh viên xin nhận/trả điểm hoãn thi (5 chữ ký viết tay, 0 dấu mộc)"
    },
    {
        "type_id": "T02_HOC_2CTDT",
        "name": "Đơn đăng ký học cùng lúc hai CTĐT",
        "file": os.path.join(BASE_DIR, "sample", "TBaovaMaudangky_hoc2CTDT.docx"),
        "cohorts": [2023, 2024, 2025],
        "desc": "Sinh viên từ năm 2 đăng ký song bằng (Chữ ký SV + Mộc PĐT)"
    },
    {
        "type_id": "T03_CHUYEN_HE",
        "name": "Đơn xin chuyển hệ đào tạo",
        "file": os.path.join(BASE_DIR, "sample", "Don_xin_chuyenhedaotao.docx"),
        "cohorts": [2023, 2024, 2025],
        "desc": "Chuyển hệ đào tạo (Chữ ký SV + Mộc BGH + Mộc PĐT)"
    },
    {
        "type_id": "T04_CHUYEN_DIEM",
        "name": "Đơn xin chuyển điểm học phần",
        "file": os.path.join(BASE_DIR, "sample", "DonXinChuyenDiem_2018.docx"),
        "cohorts": [2024, 2025, 2026],
        "desc": "Chuyển điểm học phần tương đương (Chữ ký SV + Mộc PĐT)"
    },
    {
        "type_id": "T05_BAO_LUU",
        "name": "Đơn xin bảo lưu điểm môn học",
        "file": os.path.join(BASE_DIR, "sample", "Don xin bao luu.docx"),
        "cohorts": [2022, 2023, 2024, 2025],
        "desc": "Bảo lưu kết quả học tập (Chữ ký SV + Mộc PĐT)"
    },
    {
        "type_id": "T06_BAN_SAO_SO_GOC",
        "name": "Đơn xin cấp bản sao từ sổ gốc",
        "file": os.path.join(BASE_DIR, "sample", "BAN SAO TU SO GOC.docx"),
        "cohorts": [2020, 2021, 2022],
        "desc": "Cựu SV tốt nghiệp xin cấp bản sao (Chữ ký Người làm đơn + Mộc PĐT)"
    },
    {
        "type_id": "T07_CHUYEN_DIEM_NN",
        "name": "Đơn xin chuyển điểm Ngoại ngữ tốt nghiệp",
        "file": os.path.join(BASE_DIR, "sample", "BM10_QT_PĐT_QLĐ_LSX00_NHL15052020_DonXinChuyenDiemNgoaiNgu.docx"),
        "cohorts": [2022, 2023],
        "desc": "Chứng chỉ TOEIC/IELTS xét chuẩn đầu ra (Chữ ký SV + Mộc PĐT)"
    },
    {
        "type_id": "T08_THAY_THE_MON",
        "name": "Đơn xin học môn thay thế",
        "file": os.path.join(BASE_DIR, "sample", "BM12_QT_PĐT_XCBHTBTHCNTN_LSX02_NHL15052020_ThayThe.docx"),
        "cohorts": [2022, 2023],
        "desc": "Môn thay thế tốt nghiệp (3 Chữ ký + Mộc Khoa đè chữ ký Khoa + Mộc PĐT)"
    },
    {
        "type_id": "T09_MIEN_THI",
        "name": "Đơn xin miễn thi học phần",
        "file": os.path.join(BASE_DIR, "sample", "Don mien thi.docx"),
        "cohorts": [2024, 2025, 2026],
        "desc": "Miễn thi các học phần đã học (Chữ ký SV + Mộc PĐT)"
    }
]

def render_form_page_with_stamps_and_sigs(pdf_temp_path, t_id, dpi=300):
    doc = fitz.open(pdf_temp_path)
    page = doc[0] # Take first page
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_w, img_h = img_pil.size
    
    is_diem_i_form = ("DIEM_I" in t_id)
    is_thay_the_form = ("THAY_THE" in t_id)
    is_chuyen_he_form = ("CHUYEN_HE" in t_id)
    
    # 1. SPECIAL CASE: DIEM I (5 Handwritten Signatures, 0 Stamps)
    if is_diem_i_form:
        sig_targets = [
            ("Sinh viên", "student", 0.72, 0.88),
            ("GV chấm thi 1", "teacher1", 0.18, 0.88),
            ("GV chấm thi 2", "teacher2", 0.38, 0.88),
            ("Trưởng khoa", "dean", 0.58, 0.88),
            ("Trưởng phòng", "head_pdt", 0.80, 0.88)
        ]
        for term, s_type, fallback_x_ratio, fallback_y_ratio in sig_targets:
            anchors = [a for a in page.search_for(term) if a.y1 > page.rect.height * 0.4]
            sig_w = random.randint(140, 185)
            sig_h = random.randint(55, 75)
            sig_img = create_handwritten_signature(size=(sig_w, sig_h))
            
            if anchors:
                a = anchors[-1]
                a_cx = (a.x0 + a.x1) / 2.0 * zoom
                sx = int(a_cx - sig_w / 2.0 + random.randint(-5, 5))
                sy = int(a.y1 * zoom + random.randint(8, 20))
            else:
                sx = int(img_w * fallback_x_ratio - sig_w / 2.0)
                sy = int(img_h * fallback_y_ratio)
                
            sx = max(10, min(img_w - sig_w - 10, sx))
            sy = max(10, min(img_h - sig_h - 10, sy))
            img_pil.paste(sig_img, (sx, sy), sig_img)
            
    # 2. SPECIAL CASE: THAY THE (3 Signatures + 2 Stamps)
    elif is_thay_the_form:
        # 1. Student signature under Kính đơn
        kd_anchors = page.search_for("Kính đơn")
        if kd_anchors:
            a = kd_anchors[-1]
            sig_w = random.randint(145, 185)
            sig_h = random.randint(55, 75)
            sig_img = create_handwritten_signature(size=(sig_w, sig_h))
            sx = int(((a.x0 + a.x1)/2.0)*zoom - sig_w/2.0)
            sy = int(a.y1 * zoom + random.randint(8, 18))
            img_pil.paste(sig_img, (sx, sy), sig_img)
            
        # 2. Bo mon signature under (Ghi rõ ý kiến: Đồng ý) / Bộ môn
        bm_anchors = page.search_for("Ghi rõ ý kiến") or page.search_for("Bộ môn")
        if bm_anchors:
            a = bm_anchors[-1]
            sig_w = random.randint(140, 180)
            sig_h = random.randint(55, 70)
            sig_img = create_handwritten_signature(size=(sig_w, sig_h))
            sx = int(((a.x0 + a.x1)/2.0)*zoom - sig_w/2.0)
            sy = int(a.y1 * zoom + random.randint(20, 35))
            img_pil.paste(sig_img, (sx, sy), sig_img)
            
        # 3. Faculty signature + stamp under Khoa quản môn học
        fac_anchors = page.search_for("Khoa") or page.search_for("quản")
        if fac_anchors:
            a = fac_anchors[-1]
            sig_w = random.randint(145, 185)
            sig_h = random.randint(55, 70)
            sig_img = create_handwritten_signature(size=(sig_w, sig_h))
            sx = int(((a.x0 + a.x1)/2.0)*zoom - sig_w/2.0)
            sy = int(a.y1 * zoom + random.randint(20, 35))
            img_pil.paste(sig_img, (sx, sy), sig_img)
            
            stamp_size = 180
            fac_stamp = create_circular_stamp(center_text="BAN CHỦ NHIỆM\nKHOA", size=(stamp_size, stamp_size))
            st_x = int(sx + sig_w//2 - stamp_size//2 + random.randint(-8, 8))
            st_y = int(sy + sig_h//2 - stamp_size//2 + random.randint(-8, 8))
            img_pil.paste(fac_stamp, (st_x, st_y), fac_stamp)
            
        # 4. PDT Stamp under Phòng Đào tạo
        pdt_anchors = page.search_for("Phòng") or page.search_for("Đào tạo")
        if pdt_anchors:
            a = pdt_anchors[-1]
            stamp_size = 180
            pdt_stamp = create_circular_stamp(center_text="PHÒNG\nĐÀO TẠO", size=(stamp_size, stamp_size))
            st_x2 = int(((a.x0 + a.x1)/2.0)*zoom - stamp_size//2)
            st_y2 = int(a.y1 * zoom + random.randint(15, 30))
            img_pil.paste(pdt_stamp, (st_x2, st_y2), pdt_stamp)
            
    # 3. SPECIAL CASE: CHUYEN HE (1 Student signature + 2 Stamps: BGH & PDT)
    elif is_chuyen_he_form:
        # Student signature under Kính đơn
        kd_anchors = page.search_for("Kính đơn")
        if kd_anchors:
            a = kd_anchors[-1]
            sig_w = random.randint(145, 185)
            sig_h = random.randint(55, 75)
            sig_img = create_handwritten_signature(size=(sig_w, sig_h))
            sx = int(((a.x0 + a.x1)/2.0)*zoom - sig_w/2.0)
            sy = int(a.y1 * zoom + random.randint(8, 18))
            img_pil.paste(sig_img, (sx, sy), sig_img)
            
        # BGH stamp under Ban Giám hiệu duyệt
        bgh_anchors = page.search_for("Ban Giám hiệu duyệt") or page.search_for("Ban Giám hiệu")
        stamp_size = 180
        if bgh_anchors:
            a = bgh_anchors[-1]
            bx = int(((a.x0 + a.x1)/2.0)*zoom - stamp_size/2.0)
            by = int(a.y1 * zoom + random.randint(18, 35))
        else:
            bx, by = int(img_w * 0.20), int(img_h * 0.55)
        bgh_stamp = create_circular_stamp(center_text="BAN GIÁM HIỆU", size=(stamp_size, stamp_size))
        img_pil.paste(bgh_stamp, (bx, by), bgh_stamp)
        
        # PDT stamp under Trưởng P.Đào tạo
        pdt_anchors = page.search_for("Trưởng P.Đào tạo") or page.search_for("Đề xuất của P. Đào tạo") or page.search_for("Phòng Đào tạo")
        if pdt_anchors:
            a = pdt_anchors[-1]
            px = int(((a.x0 + a.x1)/2.0)*zoom - stamp_size/2.0)
            py = int(a.y1 * zoom + random.randint(18, 35))
        else:
            px, py = int(img_w * 0.65), int(img_h * 0.65)
        pdt_stamp = create_circular_stamp(center_text="PHÒNG\nĐÀO TẠO", size=(stamp_size, stamp_size))
        img_pil.paste(pdt_stamp, (px, py), pdt_stamp)
        
    # 4. STANDARD FORMS (1 Student Signature right + 1 PDT Stamp left)
    else:
        # Student signature
        sig_w = random.randint(145, 185)
        sig_h = random.randint(55, 75)
        sig_img = create_handwritten_signature(size=(sig_w, sig_h))
        anchors = page.search_for("Sinh viên ký tên") or page.search_for("Người làm đơn") or page.search_for("Kính đơn")
        if anchors:
            a = anchors[-1]
            sx = int(((a.x0 + a.x1)/2.0)*zoom - sig_w/2.0)
            sy = int(a.y1 * zoom + random.randint(8, 20))
        else:
            sx, sy = int(img_w * 0.75), int(img_h * 0.82)
        sx = max(10, min(img_w - sig_w - 10, sx))
        sy = max(10, min(img_h - sig_h - 10, sy))
        img_pil.paste(sig_img, (sx, sy), sig_img)
        
        # PDT stamp left
        stamp_size = 180
        pdt_stamp = create_circular_stamp(center_text="PHÒNG\nĐÀO TẠO", size=(stamp_size, stamp_size))
        anchors_stamp = page.search_for("Xác nhận của Phòng Đào tạo") or page.search_for("Ý kiến của Phòng Đào tạo") or page.search_for("Phòng Đào tạo")
        if anchors_stamp:
            a = anchors_stamp[-1]
            st_x = int(((a.x0 + a.x1)/2.0)*zoom - stamp_size/2.0)
            st_y = int(a.y1 * zoom + random.randint(12, 28))
        else:
            st_x, st_y = int(img_w * 0.18), int(img_h * 0.84)
        st_x = max(10, min(img_w - stamp_size - 10, st_x))
        st_y = max(10, min(img_h - stamp_size - 10, st_y))
        img_pil.paste(pdt_stamp, (st_x, st_y), pdt_stamp)
        
    doc.close()
    return img_pil

def main():
    start_time = time.time()
    sys.stdout.reconfigure(encoding='utf-8')
    
    out_root = os.path.join(BASE_DIR, "print_dataset_100")
    pdf_dir = os.path.join(out_root, "pdf_individual")
    src_dir = os.path.join(out_root, "docx_and_html_sources")
    temp_dir = os.path.join(out_root, ".temp_build")
    
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    print("=" * 80)
    print("🖨️ BẮT ĐẦU XUẤT BỘ DỮ LIỆU IN ẤN 100 MẪU (PHƯƠNG ÁN B: ĐÃ NHÚNG MỘC & CHỮ KÝ)")
    print("📁 Thư mục xuất:", out_root)
    print("🎯 Quy cách: 100% file PDF chuẩn A4, độ phân giải 300 DPI sắc nét, có đủ mộc đỏ và chữ ký xanh")
    print("=" * 80)
    
    converter = PDFConverter()
    manifest_records = []
    sample_index = 1
    
    try:
        # ---------------------------------------------------------
        # PART 1: 9 FORM TEMPLATES (9 x 10 = 90 SAMPLES)
        # ---------------------------------------------------------
        for t_info in TEMPLATES_9:
            t_id = t_info["type_id"]
            t_name = t_info["name"]
            t_file = t_info["file"]
            cohort_pool = t_info["cohorts"]
            
            print(f"\n📄 Đang sinh 10 mẫu có chữ ký & mộc: [{t_id}] {t_name}...")
            
            for k in range(10):
                sample_code = f"PRINT_{sample_index:03d}"
                cohort_chosen = cohort_pool[k % len(cohort_pool)]
                random.seed(sample_index * 883 + 991)
                
                st = generate_mock_student(cohort_year=cohort_chosen)
                doc, rec = fill_document(t_file, st)
                
                # Save source docx
                docx_path = os.path.join(src_dir, f"{sample_code}_{t_id}.docx")
                doc.save(docx_path)
                
                # Convert to intermediate PDF
                temp_pdf = os.path.join(temp_dir, f"{sample_code}_raw.pdf")
                converter.convert(docx_path, temp_pdf)
                
                # Render to 300 DPI PIL Image and paste real stamps & signatures
                page_img_pil = render_form_page_with_stamps_and_sigs(temp_pdf, t_id, dpi=300)
                
                # Save as 300 DPI A4 PDF
                final_pdf_path = os.path.join(pdf_dir, f"{sample_code}.pdf")
                page_img_pil.save(final_pdf_path, "PDF", resolution=300.0)
                
                manifest_entry = {
                    "sample_id": sample_code,
                    "index": sample_index,
                    "document_type": t_name,
                    "type_id": t_id,
                    "category": "academic_form_docx",
                    "cohort": f"K{str(cohort_chosen)[2:]}",
                    "student_name": st["student_name"],
                    "student_id": st["student_id"],
                    "faculty": st["faculty"],
                    "major": st["major"],
                    "class_name": st["class_name"],
                    "dob": st["dob"],
                    "phone": st["phone"],
                    "has_signature": True,
                    "has_stamp": (t_id != "T01_DIEM_I"),
                    "pdf_file": f"pdf_individual/{sample_code}.pdf",
                    "source_file": f"docx_and_html_sources/{sample_code}_{t_id}.docx",
                    "pages": 1,
                    "fields_recorded": rec
                }
                manifest_records.append(manifest_entry)
                
                print(f"   [{sample_code}] K{str(cohort_chosen)[2:]} | SV: {st['student_name']:<22} | MSSV: {st['student_id']} | Mộc & Chữ ký: ĐÃ NHÚNG | 1 Trang A4")
                sample_index += 1
                
    finally:
        converter.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    # ---------------------------------------------------------
    # PART 2: 10 ACADEMIC TRANSCRIPT SAMPLES (PRINT_091 - PRINT_100)
    # ---------------------------------------------------------
    print(f"\n📊 Đang sinh 10 mẫu Bảng điểm học tập trực tuyến (K20 - K26)...")
    transcript_cohorts = [
        (2026, "AI",   "Vừa nhập học HK01 (Tuần 1)", False),
        (2026, "CNTT", "Vừa nhập học HK01 (Tuần 1)", True),
        (2025, "DDT",  "Năm 2 (Đã có điểm Năm 1)", False),
        (2025, "CK",   "Năm 2 (Đã có điểm Năm 1)", True),
        (2024, "OTO",  "Năm 3 (Đã có điểm Năm 1, 2)", False),
        (2024, "QTKD", "Năm 3 (Đã có điểm Năm 1, 2)", True),
        (2023, "AI",   "Năm 4 (Chuẩn theo mẫu gốc - 6 HK)", True),
        (2023, "CNTT", "Năm 4 (6 HK hoàn thành)", False),
        (2022, "DDT",  "Năm cuối (8 HK hoàn thành)", True),
        (2020, "CK",   "Đã tốt nghiệp (Toàn khóa 147 TC)", False)
    ]
    
    for k, (c_yr, c_major, c_desc, is_clc) in enumerate(transcript_cohorts):
        sample_code = f"PRINT_{sample_index:03d}"
        random.seed(sample_index * 919 + 773)
        
        name = generate_random_name()
        mssv = f"{str(c_yr)[2:]}{random.randint(110000, 149999)}"
        
        html_code = generate_transcript_html(
            student_name=name,
            student_id=mssv,
            cohort_year=c_yr,
            major_key=c_major,
            is_clc=is_clc
        )
        
        html_file = os.path.join(src_dir, f"{sample_code}_TRANSCRIPT_{c_major}.html")
        with open(html_file, "w", encoding="utf-8") as f_h:
            f_h.write(html_code)
            
        pdf_path = os.path.join(pdf_dir, f"{sample_code}.pdf")
        
        cmd = [
            EDGE_EXE,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_file
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        
        doc_fitz = fitz.open(pdf_path)
        page_cnt = len(doc_fitz)
        doc_fitz.close()
        
        manifest_entry = {
            "sample_id": sample_code,
            "index": sample_index,
            "document_type": "Bảng điểm học tập trực tuyến",
            "type_id": "T10_ACADEMIC_TRANSCRIPT",
            "category": "academic_transcript_portal",
            "cohort": f"K{str(c_yr)[2:]}",
            "student_name": name,
            "student_id": mssv,
            "major": c_major,
            "has_signature": False,
            "has_stamp": False,
            "status_description": c_desc,
            "pdf_file": f"pdf_individual/{sample_code}.pdf",
            "source_file": f"docx_and_html_sources/{sample_code}_TRANSCRIPT_{c_major}.html",
            "pages": page_cnt
        }
        manifest_records.append(manifest_entry)
        
        print(f"   [{sample_code}] K{str(c_yr)[2:]} - {c_major:<5} | SV: {name:<22} | MSSV: {mssv} | {c_desc}")
        sample_index += 1
        
    # ---------------------------------------------------------
    # PART 3: CREATE MASTER MERGED PDF
    # ---------------------------------------------------------
    print(f"\n📑 Đang tạo file Master gộp toàn bộ 100 tài liệu: ALL_100_PAGES_MASTER.pdf...")
    master_pdf_path = os.path.join(out_root, "ALL_100_PAGES_MASTER.pdf")
    master_doc = fitz.open()
    
    for rec in manifest_records:
        pdf_file = os.path.join(out_root, rec["pdf_file"])
        if os.path.exists(pdf_file):
            item_doc = fitz.open(pdf_file)
            master_doc.insert_pdf(item_doc)
            item_doc.close()
            
    master_doc.save(master_pdf_path)
    total_master_pages = len(master_doc)
    master_doc.close()
    
    # ---------------------------------------------------------
    # PART 4: SAVE DATASET_MANIFEST.JSON
    # ---------------------------------------------------------
    manifest_path = os.path.join(out_root, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f_m:
        json.dump({
            "dataset_name": "HCMUTE Academic Document Benchmark (100 Print-Ready Samples with Embedded Stamps & Signatures)",
            "generated_date": "2026-08-30",
            "total_samples": len(manifest_records),
            "master_pdf": "ALL_100_PAGES_MASTER.pdf",
            "total_pages": total_master_pages,
            "features": "Embedded vivid red official circular stamps + natural blue handwritten signatures",
            "categories": {
                "academic_forms_with_stamps_and_signatures": 90,
                "academic_transcripts_portal": 10
            },
            "samples": manifest_records
        }, f_m, ensure_ascii=False, indent=2)
        
    elapsed = time.time() - start_time
    print("=" * 80)
    print("🎉 HOÀN TẤT XUẤT BỘ DỮ LIỆU IN ẤN 100 MẪU (ĐÃ CÓ MỘC & CHỮ KÝ)!")
    print(f"📁 Thư mục: {out_root}")
    print(f"📑 File master: {master_pdf_path}")
    print(f"📊 Metadata: {manifest_path}")
    print(f"⏱️ Tổng thời gian: {elapsed:.2f} giây")
    print("=" * 80)

if __name__ == "__main__":
    main()
