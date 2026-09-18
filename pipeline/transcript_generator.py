# -*- coding: utf-8 -*-
"""
Academic Transcript Portal Generator for Vietnamese Universities (HCMUTE / VNU style)
Generates pixel-perfect web portal transcript images across cohorts (K20-K26) and majors,
and extracts precise ground truth bounding boxes for YOLOv11 detection.
"""

import os
import sys
import json
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from selenium import webdriver
from selenium.webdriver.edge.options import Options

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline.ontology import CLASS_MAP, CLASS_NAMES
from pipeline.curriculum_data import CURRICULUM_DATABASE

EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

def get_grade_info(score_10):
    score_10 = round(score_10, 1)
    if score_10 >= 9.0:
        return score_10, 4.0, "A+", "Đạt"
    elif score_10 >= 8.5:
        return score_10, 3.7, "A", "Đạt"
    elif score_10 >= 8.0:
        return score_10, 3.5, "B+", "Đạt"
    elif score_10 >= 7.0:
        return score_10, 3.0, "B", "Đạt"
    elif score_10 >= 6.5:
        return score_10, 2.5, "C+", "Đạt"
    elif score_10 >= 5.5:
        return score_10, 2.0, "C", "Đạt"
    elif score_10 >= 5.0:
        return score_10, 1.5, "D+", "Đạt"
    elif score_10 >= 4.0:
        return score_10, 1.0, "D", "Đạt"
    else:
        return score_10, 0.0, "F", "Không đạt"

def get_rank_vn(gpa_4):
    if gpa_4 >= 3.6:
        return "Xuất sắc"
    elif gpa_4 >= 3.2:
        return "Giỏi"
    elif gpa_4 >= 2.5:
        return "Khá"
    elif gpa_4 >= 2.0:
        return "Trung bình"
    else:
        return "Yếu"

def get_cohort_semesters(cohort_year, major_key="AI"):
    """
    Returns the list of semesters (completed + in-progress) for a given cohort year (2020-2026).
    Current time: HK01 Năm học 2026-2027.
    """
    curriculum = CURRICULUM_DATABASE.get(major_key, CURRICULUM_DATABASE["AI"])
    all_raw_sems = curriculum["semesters"]
    
    y0 = cohort_year
    y1 = y0 + 1
    y2 = y0 + 2
    y3 = y0 + 3
    y4 = y0 + 4
    
    # Calculate how many semesters this cohort has reached by 2026-2027 HK01
    # 2020 (K20): 8 sems completed
    # 2021 (K21): 8 sems completed
    # 2022 (K22): 8 sems completed
    # 2023 (K23): 6 completed + 1 in-progress (7 sems)
    # 2024 (K24): 4 completed + 1 in-progress (5 sems)
    # 2025 (K25): 2 completed + 1 in-progress (3 sems)
    # 2026 (K26): 1 in-progress (1 sem)
    
    cohort_diff = 2026 - cohort_year
    if cohort_diff >= 4:
        num_completed = 8
        has_in_progress = False
    elif cohort_diff == 3: # K23
        num_completed = 6
        has_in_progress = True
    elif cohort_diff == 2: # K24
        num_completed = 4
        has_in_progress = True
    elif cohort_diff == 1: # K25
        num_completed = 2
        has_in_progress = True
    else: # K26
        num_completed = 0
        has_in_progress = True
        
    final_sems = []
    
    for idx, (raw_y, raw_s, courses) in enumerate(all_raw_sems):
        sem_year = raw_y.format(y0=y0, y1=y1, y2=y2, y3=y3, y4=y4)
        
        if idx < num_completed:
            final_sems.append((sem_year, raw_s, courses, False)) # Completed
        elif idx == num_completed and has_in_progress:
            # In progress semester (HK01 2026-2027)
            in_prog_year = "Năm học: 2026-2027"
            in_prog_sem = "Học kỳ: HK01"
            # Take 3-4 courses
            in_prog_courses = courses[:min(4, len(courses))]
            final_sems.append((in_prog_year, in_prog_sem, in_prog_courses, True)) # In progress
            break
        else:
            break
            
    return final_sems, curriculum

def generate_transcript_html(student_name, student_id, cohort_year=2023, major_key="AI", is_clc=True):
    semesters, curriculum = get_cohort_semesters(cohort_year, major_key)
    
    clc_tag = "CLC_" if is_clc else "_"
    prog_code = f"{str(cohort_year)[2:]}110{clc_tag}{curriculum['code_prefix']}_{curriculum['program_name']}"
    
    stt_counter = 1
    cum_credits = 0
    cum_score10_weighted = 0.0
    cum_score4_weighted = 0.0
    
    table_sections_html = []
    
    for year_title, sem_title, course_list, in_progress in semesters:
        sem_credits = 0
        sem_score10_weighted = 0.0
        sem_score4_weighted = 0.0
        
        rows_html = []
        for code, name, cred in course_list:
            is_pass_only = "*" in name
            
            if not in_progress:
                score_val = random.uniform(6.0, 9.8)
                s10, s4, letter, res = get_grade_info(score_val)
                if is_pass_only:
                    letter = "P"
                    res = "Đạt"
                s10_str = f"{s10}"
                s4_str = f"{s4}"
                letter_str = f"{letter}"
                res_html = '<span class="check-icon">✔</span>'
                
                sem_credits += cred
                if not is_pass_only:
                    sem_score10_weighted += s10 * cred
                    sem_score4_weighted += s4 * cred
            else:
                # In progress semester (e.g. K26 or current semester)
                s10_str = ""
                s4_str = ""
                letter_str = ""
                res_html = ""
                
            row_id = f"row_{stt_counter}"
            
            # Tags for bounding box detection
            code_tag = f'<span class="bbox-tag" data-cls="course_code">{code}</span>'
            name_tag = f'<span class="bbox-tag" data-cls="course_name">{name}</span>'
            tc_tag = f'<span class="bbox-tag" data-cls="credits">{cred}</span>'
            s10_tag = f'<span class="bbox-tag" data-cls="score">{s10_str}</span>' if s10_str else ''
            s4_tag = f'<span class="bbox-tag" data-cls="score">{s4_str}</span>' if s4_str else ''
            letter_tag = f'<span class="bbox-tag" data-cls="score">{letter_str}</span>' if letter_str else ''
            
            rows_html.append(f"""
            <tr id="{row_id}">
                <td class="col-stt">{stt_counter}</td>
                <td class="col-code">{code_tag}</td>
                <td class="col-name">{name_tag}</td>
                <td class="col-tc">{tc_tag}</td>
                <td class="col-s10">{s10_tag}</td>
                <td class="col-s4">{s4_tag}</td>
                <td class="col-let">{letter_tag}</td>
                <td class="col-res">{res_html}</td>
                <td class="col-note"></td>
                <td class="col-act"><span class="chevron-icon">⌄</span></td>
            </tr>
            """)
            stt_counter += 1
            
        if not in_progress:
            cum_credits += sem_credits
            cum_score10_weighted += sem_score10_weighted
            cum_score4_weighted += sem_score4_weighted
            
            sem_gpa10 = round(sem_score10_weighted / max(1, sem_credits), 2)
            sem_gpa4 = round(sem_score4_weighted / max(1, sem_credits), 2)
            sem_rank = get_rank_vn(sem_gpa4)
            sem_ren_luyen = random.randint(75, 95)
            
            cum_gpa10 = round(cum_score10_weighted / max(1, cum_credits), 2)
            cum_gpa4 = round(cum_score4_weighted / max(1, cum_credits), 2)
            cum_rank = get_rank_vn(cum_gpa4)
            
            summary_html = f"""
            <tr class="summary-row">
                <td colspan="10">
                    <div class="summary-box">
                        <div class="summary-col">
                            <div>- Số tín chỉ đạt học kỳ: <span class="bbox-tag" data-cls="credits">{sem_credits}</span></div>
                            <div>- Điểm TB học kỳ(Hệ 10): <span class="bbox-tag" data-cls="score">{sem_gpa10:.2f} ({sem_rank})</span></div>
                            <div>- Điểm TB học kỳ(Hệ 4): <span class="bbox-tag" data-cls="score">{sem_gpa4:.2f} ({sem_rank})</span></div>
                            <div>- Điểm rèn luyện học kỳ: <span class="bbox-tag" data-cls="score">{sem_ren_luyen} (Tốt)</span></div>
                        </div>
                        <div class="summary-col">
                            <div>- Tổng số tín chỉ tích luỹ: <span class="bbox-tag" data-cls="credits">{cum_credits}</span></div>
                            <div>- Điểm TB tích luỹ(Hệ 10): <span class="bbox-tag" data-cls="score">{cum_gpa10:.2f} ({cum_rank})</span></div>
                            <div>- Điểm TB tích luỹ(Hệ 4): <span class="bbox-tag" data-cls="score">{cum_gpa4:.2f} ({cum_rank})</span></div>
                            <div>- Điểm rèn luyện toàn khóa: <span class="bbox-tag" data-cls="score">78.00</span></div>
                        </div>
                    </div>
                </td>
            </tr>
            """
        else:
            # In progress summary (matching NguyenBachTung_Academic_Transcript.jpg)
            summary_html = f"""
            <tr class="summary-row">
                <td colspan="10">
                    <div class="summary-box">
                        <div class="summary-col">
                            <div>- Số tín chỉ đạt học kỳ: <span class="bbox-tag" data-cls="credits">0</span></div>
                            <div>- Điểm TB học kỳ(Hệ 10): (Không xếp loại)</div>
                            <div>- Điểm TB học kỳ(Hệ 4): (Không xếp loại)</div>
                            <div>- Điểm rèn luyện học kỳ: ()</div>
                        </div>
                        <div class="summary-col">
                            <div>- Tổng số tín chỉ tích luỹ: <span class="bbox-tag" data-cls="credits">{cum_credits}</span></div>
                            <div>- Điểm TB tích luỹ(Hệ 10): (Không xếp loại)</div>
                            <div>- Điểm TB tích luỹ(Hệ 4): (Không xếp loại)</div>
                            <div>- Điểm rèn luyện toàn khóa: <span class="bbox-tag" data-cls="score">78.00</span></div>
                        </div>
                    </div>
                </td>
            </tr>
            """
            
        table_sections_html.append(f"""
        <tr class="year-header-row"><td colspan="10" class="year-header"><span class="bbox-tag" data-cls="date">{year_title}</span></td></tr>
        <tr class="sem-header-row"><td colspan="10" class="sem-header"><span class="bbox-tag" data-cls="date">{sem_title}</span></td></tr>
        {''.join(rows_html)}
        {summary_html}
        """)
        
    full_html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>KẾT QUẢ HỌC TẬP - {student_name}</title>
<style>
    * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}
    body {{
        background-color: #eef2f6;
        color: #333333;
        padding-bottom: 40px;
    }}
    .top-navbar {{
        background-color: #00407a;
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 20px;
        height: 48px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }}
    .nav-left {{
        display: flex;
        align-items: center;
        gap: 15px;
    }}
    .nav-menu-icon {{
        font-size: 20px;
    }}
    .nav-title {{
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    .nav-right {{
        display: flex;
        align-items: center;
        gap: 16px;
    }}
    .flag-icon {{
        width: 22px;
        height: 15px;
        background: #da251d;
        position: relative;
        display: inline-block;
        border-radius: 2px;
    }}
    .flag-icon::after {{
        content: "★";
        color: #ff0;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 11px;
    }}
    .avatar-icon {{
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background-color: #4a90e2;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 14px;
        font-weight: bold;
    }}
    
    .main-container {{
        max-width: 1040px;
        margin: 15px auto;
        background: #ffffff;
        border-radius: 4px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        overflow: hidden;
    }}
    
    .page-title-ribbon {{
        background: linear-gradient(135deg, #003366 0%, #004d99 100%);
        color: #ffffff;
        display: inline-block;
        padding: 10px 30px 10px 20px;
        font-size: 17px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 15px;
        clip-path: polygon(0 0, 93% 0, 100% 100%, 0% 100%);
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }}
    
    .filter-card {{
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        margin: 0 20px 20px 20px;
        padding: 16px 20px;
        background-color: #ffffff;
    }}
    .filter-row {{
        display: flex;
        gap: 20px;
        margin-bottom: 14px;
    }}
    .filter-group {{
        flex: 1;
        position: relative;
    }}
    .filter-group label {{
        position: absolute;
        top: -8px;
        left: 12px;
        background: #ffffff;
        padding: 0 6px;
        font-size: 11px;
        color: #64748b;
        font-weight: 600;
    }}
    .filter-select {{
        width: 100%;
        padding: 10px 12px;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        font-size: 13px;
        color: #1e293b;
        background-color: #fff;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .radio-row {{
        display: flex;
        gap: 30px;
        font-size: 13px;
        color: #334155;
        align-items: center;
        padding-top: 4px;
    }}
    .radio-item {{
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .radio-dot {{
        width: 14px;
        height: 14px;
        border-radius: 50%;
        border: 2px solid #0056b3;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .radio-dot.active::after {{
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #0056b3;
    }}
    
    .grade-table-container {{
        padding: 0 20px 20px 20px;
    }}
    table.grade-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }}
    table.grade-table th {{
        background-color: #003870;
        color: #ffffff;
        font-weight: 600;
        padding: 10px 6px;
        border: 1px solid #002b57;
        text-align: center;
    }}
    table.grade-table td {{
        padding: 7px 8px;
        border: 1px solid #e2e8f0;
        color: #1e293b;
    }}
    
    .col-stt {{ width: 42px; text-align: center; font-weight: 500; }}
    .col-code {{ width: 110px; text-align: left; color: #0f172a; font-weight: 500; }}
    .col-name {{ text-align: left; }}
    .col-tc {{ width: 55px; text-align: center; }}
    .col-s10 {{ width: 75px; text-align: center; font-weight: 500; }}
    .col-s4 {{ width: 70px; text-align: center; }}
    .col-let {{ width: 70px; text-align: center; font-weight: 600; }}
    .col-res {{ width: 65px; text-align: center; }}
    .col-note {{ width: 65px; text-align: center; }}
    .col-act {{ width: 55px; text-align: center; color: #94a3b8; font-size: 14px; }}
    
    .check-icon {{
        color: #22c55e;
        font-size: 13px;
        display: inline-block;
        width: 16px;
        height: 16px;
        background: #dcfce7;
        border-radius: 50%;
        line-height: 16px;
        text-align: center;
    }}
    
    .year-header-row td {{
        background-color: #ffffff;
        font-weight: bold;
        color: #000000;
        font-size: 13px;
        padding: 10px 8px 6px 8px;
        border-bottom: none;
    }}
    .sem-header-row td {{
        background-color: #cbe3fb;
        font-weight: bold;
        color: #003870;
        font-size: 12.5px;
        padding: 7px 10px;
        border-top: none;
    }}
    
    .summary-box {{
        background-color: #fef3c7;
        border: 1px solid #fde68a;
        padding: 10px 16px;
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        font-weight: 600;
        color: #000000;
        line-height: 1.6;
    }}
    .summary-col {{
        flex: 1;
    }}
    
    .fab-btn {{
        position: fixed;
        bottom: 25px;
        right: 30px;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background-color: #00407a;
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.3);
    }}
</style>
</head>
<body>

<div class="top-navbar">
    <div class="nav-left">
        <span class="nav-menu-icon">≡</span>
        <span class="nav-title">TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT TP.HCM</span>
    </div>
    <div class="nav-right">
        <div class="flag-icon"></div>
        <span style="font-size: 17px;">🔔</span>
        <div class="avatar-icon">👤</div>
    </div>
</div>

<div class="main-container">
    <div class="page-title-ribbon">KẾT QUẢ HỌC TẬP</div>
    
    <div class="filter-card">
        <div class="filter-row">
            <div class="filter-group">
                <label>Chương trình đào tạo</label>
                <div class="filter-select">
                    <span class="bbox-tag" data-cls="degree_info">{prog_code}</span>
                    <span>⌄</span>
                </div>
            </div>
        </div>
        
        <div class="filter-row">
            <div class="filter-group">
                <label>Năm học</label>
                <div class="filter-select">
                    <span>Tất cả</span>
                    <span>⌄</span>
                </div>
            </div>
            <div class="filter-group">
                <label>Học kỳ</label>
                <div class="filter-select">
                    <span>Tất cả</span>
                    <span>⌄</span>
                </div>
            </div>
        </div>
        
        <div class="radio-row">
            <div class="radio-item">
                <div class="radio-dot active"></div>
                <span>Xem điểm sinh viên: <b class="bbox-tag" data-cls="student_name">{student_name}</b> (<span class="bbox-tag" data-cls="student_id">{student_id}</span>)</span>
            </div>
            <div class="radio-item">
                <div class="radio-dot"></div>
                <span>Xem điểm theo chương trình đào tạo</span>
            </div>
        </div>
    </div>
    
    <div class="grade-table-container">
        <table class="grade-table">
            <thead>
                <tr>
                    <th>STT</th>
                    <th>Mã môn học</th>
                    <th>Tên môn học</th>
                    <th>Số TC</th>
                    <th>Điểm hệ 10</th>
                    <th>Điểm hệ 4</th>
                    <th>Điểm chữ</th>
                    <th>Kết quả</th>
                    <th>Ghi chú</th>
                    <th>Chi tiết</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_sections_html)}
            </tbody>
        </table>
    </div>
</div>

<div class="fab-btn">⚙</div>
</body>
</html>
"""
    return full_html

def generate_transcript_sample_with_driver(driver, student_name, student_id, cohort_year, major_key, output_base_name, output_dir, is_clc=True):
    os.makedirs(output_dir, exist_ok=True)
    html_content = generate_transcript_html(student_name, student_id, cohort_year, major_key, is_clc)
    
    html_file = os.path.join(output_dir, f"{output_base_name}.html")
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    driver.get(f"file:///{os.path.abspath(html_file)}")
    
    scroll_height = driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);")
    driver.set_window_size(1120, scroll_height + 50)
    
    img_file = os.path.join(output_dir, f"{output_base_name}.png")
    driver.save_screenshot(img_file)
    
    extracted_boxes = driver.execute_script("""
        const tags = document.querySelectorAll('.bbox-tag');
        const boxes = [];
        tags.forEach(el => {
            const r = el.getBoundingClientRect();
            const text = el.innerText.trim();
            const cls = el.getAttribute('data-cls');
            if (r.width > 0 && r.height > 0 && text) {
                boxes.push({
                    text: text,
                    class_name: cls,
                    box: [Math.round(r.left + window.scrollX), Math.round(r.top + window.scrollY), Math.round(r.right + window.scrollX), Math.round(r.bottom + window.scrollY)]
                });
            }
        });
        return boxes;
    """)
    
    img_pil = Image.open(img_file).convert("RGB")
    img_w, img_h = img_pil.size
    
    lbl_file = os.path.join(output_dir, f"{output_base_name}.txt")
    yolo_lines = []
    
    vis_img = img_pil.copy()
    draw = ImageDraw.Draw(vis_img)
    
    CLASS_COLORS = {
        "student_name": (0, 102, 204),
        "student_id": (204, 0, 102),
        "degree_info": (153, 51, 102),
        "course_code": (128, 0, 128),
        "course_name": (0, 153, 76),
        "credits": (102, 102, 153),
        "score": (204, 0, 0),
        "date": (102, 51, 153)
    }
    
    for item in extracted_boxes:
        cls_name = item["class_name"]
        cls_id = CLASS_MAP.get(cls_name, 0)
        x0, y0, x1, y1 = item["box"]
        
        x0 = max(0, min(img_w, x0))
        y0 = max(0, min(img_h, y0))
        x1 = max(0, min(img_w, x1))
        y1 = max(0, min(img_h, y1))
        
        bw = x1 - x0
        bh = y1 - y0
        if bw <= 1 or bh <= 1:
            continue
            
        xc = (x0 + x1) / 2.0 / img_w
        yc = (y0 + y1) / 2.0 / img_h
        w_norm = bw / img_w
        h_norm = bh / img_h
        
        yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        color = CLASS_COLORS.get(cls_name, (255, 0, 0))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.rectangle([x0, max(0, y0 - 11), x0 + len(cls_name)*5 + 4, y0], fill=color)
        draw.text((x0 + 2, max(0, y0 - 11)), cls_name, fill=(255, 255, 255))
        
    with open(lbl_file, "w", encoding="utf-8") as f_lbl:
        f_lbl.write("\n".join(yolo_lines) + "\n")
        
    vis_file = os.path.join(output_dir, f"{output_base_name}_annotated.png")
    vis_img.save(vis_file)
    
    return {
        "image": img_file,
        "label": lbl_file,
        "visualization": vis_file,
        "num_bboxes": len(yolo_lines),
        "dimensions": (img_w, img_h)
    }

