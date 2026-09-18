# -*- coding: utf-8 -*-
"""
YOLO Coordinate Extractor, Stamp/Signature Placer & Visualizer
Provides accurate text search, IoU deduplication, scan effects, and visualization.
"""

import os
import random
import fitz # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
from pipeline.ontology import CLASS_NAMES, CLASS_MAP, CLASS_COLORS
from pipeline.stamp_generator import create_circular_stamp, create_handwritten_signature
from pipeline.scan_simulator import apply_scan_effects

def calculate_iou(box1, box2):
    """Calculates Intersection over Union for two [x0, y0, x1, y1] boxes."""
    x0 = max(box1[0], box2[0])
    y0 = max(box1[1], box2[1])
    x1 = min(box1[2], box2[2])
    y1 = min(box1[3], box2[3])
    
    inter_w = max(0.0, x1 - x0)
    inter_h = max(0.0, y1 - y0)
    inter_area = inter_w * inter_h
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def deduplicate_bboxes(bboxes, iou_thresh=0.7):
    """Removes duplicate or highly overlapping bounding boxes of the same class."""
    if not bboxes:
        return []
    
    # Sort by area descending
    sorted_boxes = sorted(bboxes, key=lambda b: (b["box"][2]-b["box"][0])*(b["box"][3]-b["box"][1]), reverse=True)
    kept = []
    
    for b in sorted_boxes:
        is_dup = False
        for k in kept:
            if b["class_id"] == k["class_id"] and calculate_iou(b["box"], k["box"]) > iou_thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append(b)
            
    return kept

def extract_and_generate_dataset_sample(
    pdf_path,
    recorded_fields,
    output_base_name,
    output_dir,
    dpi=200
):
    """
    Extracts coordinates from PDF using fitz.search_for, places stamps and signatures,
    applies scan simulation, writes YOLO .txt labels, and outputs visualization images.
    """
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    
    generated_files = []

    for page_idx, page in enumerate(doc):
        # Render high-res page image
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        img_w, img_h = img_pil.size
        pw, ph = page.rect.width, page.rect.height
        
        raw_bboxes = []
        
        # 1. Search for all recorded text fields
        for field in recorded_fields:
            text = field["text"].strip()
            cls_name = field["class_name"]
            cls_id = CLASS_MAP.get(cls_name, 0)
            
            if not text:
                continue
                
            # Skip searching for single standalone digits to avoid matching every number globally
            if len(text) <= 2 and text.isdigit():
                continue
                
            # Search on page
            rects = page.search_for(text)
            if not rects and len(text) > 20:
                # Try search by first 15 chars if text is long
                rects = page.search_for(text[:15])
                
            for r in rects:
                # Convert to image pixel coordinates
                px0 = r.x0 * zoom
                py0 = r.y0 * zoom
                px1 = r.x1 * zoom
                py1 = r.y1 * zoom
                
                # Add slight padding
                px0 = max(0, px0 - 2)
                py0 = max(0, py0 - 2)
                px1 = min(img_w, px1 + 2)
                py1 = min(img_h, py1 + 2)
                
                if px1 > px0 and py1 > py0:
                    raw_bboxes.append({
                        "box": [px0, py0, px1, py1],
                        "class_id": cls_id,
                        "text": text
                    })

        # 2. Place Stamps and Signatures (on last page or signature anchors)
        student_sig_anchors = [
            "(Ký và ghi rõ họ tên)",
            "(Ký, ghi rõ họ tên)",
            "Sinh viên ký tên",
            "Sinh viên làm đơn",
            "Người làm đơn",
            "Người viết đơn",
            "Người đề nghị",
            "Kính đơn",
            "Sinh viên"
        ]
        # Student signature placement
        sig_placed = False
        for kw in student_sig_anchors:
            kw_rects = page.search_for(kw)
            if kw_rects:
                if kw == "Sinh viên":
                    bottom_rects = [r for r in kw_rects if r.y0 > 0.65 * ph and r.x0 > 0.55 * pw]
                elif kw == "Kính đơn":
                    bottom_rects = [r for r in kw_rects if r.y0 > 0.28 * ph and r.x0 > 0.35 * pw]
                else:
                    bottom_rects = [r for r in kw_rects if r.y0 > 0.35 * ph and r.x0 > 0.45 * pw]
                    
                if not bottom_rects:
                    continue
                
                anchor = bottom_rects[-1]
                
                sig_img = create_handwritten_signature(size=(random.randint(150, 185), random.randint(50, 75)))
                sw, sh = sig_img.size
                
                anchor_center_x = (anchor.x0 + anchor.x1) / 2.0 * zoom
                sx = int(anchor_center_x - sw / 2.0)
                sy = int(anchor.y1 * zoom + random.randint(4, 12))
                
                sx = max(20, min(img_w - sw - 20, sx))
                sy = max(20, min(img_h - sh - 20, sy))
                
                img_pil.paste(sig_img, (sx, sy), sig_img)
                raw_bboxes.append({
                    "box": [sx, sy, sx + sw, sy + sh],
                    "class_id": CLASS_MAP["signature"],
                    "text": "signature"
                })
                sig_placed = True
                break

        if not sig_placed and page_idx == len(doc) - 1:
            sx, sy = int(img_w * 0.68), int(img_h * 0.86)
            sig_img = create_handwritten_signature(size=(190, 75))
            sw, sh = sig_img.size
            img_pil.paste(sig_img, (sx, sy), sig_img)
            raw_bboxes.append({
                "box": [sx, sy, sx + sw, sy + sh],
                "class_id": CLASS_MAP["signature"],
                "text": "signature"
            })

        # Special signature & stamp handling for "Đơn xin nhận và trả điểm I"
        is_diem_i_form = bool(page.search_for("NHẬN ĐIỂM I") or page.search_for("TRẢ ĐIỂM I"))
        
        if is_diem_i_form:
            # 1. Signature under TRƯỞNG KHOA
            tk_rects = page.search_for("TRƯỞNG KHOA")
            if tk_rects:
                anchor = [r for r in tk_rects if r.y0 > 0.65 * ph and r.x0 < 0.35 * pw][-1]
                sig_img = create_handwritten_signature(size=(random.randint(130, 160), random.randint(45, 65)))
                sw, sh = sig_img.size
                sx = int((anchor.x0 + anchor.x1)/2.0 * zoom - sw/2.0)
                sy = int(anchor.y1 * zoom + random.randint(4, 10))
                sx = max(20, min(img_w - sw - 20, sx))
                sy = max(20, min(img_h - sh - 20, sy))
                img_pil.paste(sig_img, (sx, sy), sig_img)
                raw_bboxes.append({"box": [sx, sy, sx+sw, sy+sh], "class_id": CLASS_MAP["signature"], "text": "signature"})
                
            # 2. Signature under TRƯỞNG PHÒNG ĐÀO TẠO
            tpdt_rects = page.search_for("TRƯỞNG PHÒNG ĐÀO TẠO")
            if tpdt_rects:
                anchor = [r for r in tpdt_rects if r.y0 > 0.65 * ph][-1]
                sig_img = create_handwritten_signature(size=(random.randint(130, 160), random.randint(45, 65)))
                sw, sh = sig_img.size
                sx = int((anchor.x0 + anchor.x1)/2.0 * zoom - sw/2.0)
                sy = int(anchor.y1 * zoom + random.randint(4, 10))
                sx = max(20, min(img_w - sw - 20, sx))
                sy = max(20, min(img_h - sh - 20, sy))
                img_pil.paste(sig_img, (sx, sy), sig_img)
                raw_bboxes.append({"box": [sx, sy, sx+sw, sy+sh], "class_id": CLASS_MAP["signature"], "text": "signature"})
                
            # 3. Signature under GV chấm thi 1
            gv1_rects = page.search_for("GV chấm thi 1")
            if gv1_rects:
                anchor = [r for r in gv1_rects if r.y0 > 0.7 * ph][-1]
                sig_img = create_handwritten_signature(size=(random.randint(110, 135), random.randint(38, 52)))
                sw, sh = sig_img.size
                sx = int((anchor.x0 + anchor.x1)/2.0 * zoom - sw/2.0)
                sy = int(anchor.y1 * zoom + random.randint(4, 8))
                sx = max(20, min(img_w - sw - 20, sx))
                sy = max(20, min(img_h - sh - 20, sy))
                img_pil.paste(sig_img, (sx, sy), sig_img)
                raw_bboxes.append({"box": [sx, sy, sx+sw, sy+sh], "class_id": CLASS_MAP["signature"], "text": "signature"})
                
            # 4. Signature under GV chấm thi 2
            gv2_rects = page.search_for("GV chấm thi 2")
            if gv2_rects:
                anchor = [r for r in gv2_rects if r.y0 > 0.7 * ph][-1]
                sig_img = create_handwritten_signature(size=(random.randint(110, 135), random.randint(38, 52)))
                sw, sh = sig_img.size
                sx = int((anchor.x0 + anchor.x1)/2.0 * zoom - sw/2.0)
                sy = int(anchor.y1 * zoom + random.randint(4, 8))
                sx = max(20, min(img_w - sw - 20, sx))
                sy = max(20, min(img_h - sh - 20, sy))
                img_pil.paste(sig_img, (sx, sy), sig_img)
                raw_bboxes.append({"box": [sx, sy, sx+sw, sy+sh], "class_id": CLASS_MAP["signature"], "text": "signature"})

        # Authority signatures for Department / Faculty approvals (ONLY for BM12 Thay The)
        # Authority signatures for Department / Faculty approvals (ONLY for BM12 Thay The)
        is_thay_the_form = bool(page.search_for("THAY THẾ") or page.search_for("Thay thế") or page.search_for("(Ghi rõ ý kiến"))
        placed_stamps_count = 0
        
        if is_thay_the_form:
            # 1. Dept head signature
            dept_sig_anchors = ["(Ghi rõ ý kiến: Đồng ý)", "(Ghi rõ ý kiến)"]
            for kw in dept_sig_anchors:
                kw_rects = page.search_for(kw)
                if kw_rects:
                    bottom_rects = [r for r in kw_rects if r.y0 > 0.35 * ph and r.x0 < 0.45 * pw]
                    if bottom_rects:
                        anchor = bottom_rects[-1]
                        sig_img = create_handwritten_signature(size=(random.randint(130, 165), random.randint(45, 65)))
                        sw, sh = sig_img.size
                        anchor_center_x = (anchor.x0 + anchor.x1) / 2.0 * zoom
                        sx = int(anchor_center_x - sw / 2.0)
                        sy = int(anchor.y1 * zoom + random.randint(4, 12))
                        sx = max(20, min(img_w - sw - 20, sx))
                        sy = max(20, min(img_h - sh - 20, sy))
                        img_pil.paste(sig_img, (sx, sy), sig_img)
                        raw_bboxes.append({
                            "box": [sx, sy, sx + sw, sy + sh],
                            "class_id": CLASS_MAP["signature"],
                            "text": "signature"
                        })
                        break

            # 2. Faculty head signature & overlapping Khoa stamp
            faculty_sig_anchors = ["Khoa quản môn học", "Khoa quản lý môn học"]
            for kw in faculty_sig_anchors:
                kw_rects = page.search_for(kw)
                if kw_rects:
                    bottom_rects = [r for r in kw_rects if r.y0 > 0.35 * ph and 0.25 * pw < r.x0 < 0.65 * pw]
                    if bottom_rects:
                        anchor = bottom_rects[-1]
                        anchor_center_x = (anchor.x0 + anchor.x1) / 2.0 * zoom
                        
                        # 2a. Faculty signature
                        sig_img = create_handwritten_signature(size=(random.randint(130, 165), random.randint(45, 65)))
                        sw, sh = sig_img.size
                        sx = int(anchor_center_x - sw / 2.0)
                        sy = int(anchor.y1 * zoom + random.randint(4, 12))
                        sx = max(20, min(img_w - sw - 20, sx))
                        sy = max(20, min(img_h - sh - 20, sy))
                        img_pil.paste(sig_img, (sx, sy), sig_img)
                        raw_bboxes.append({
                            "box": [sx, sy, sx + sw, sy + sh],
                            "class_id": CLASS_MAP["signature"],
                            "text": "signature"
                        })
                        
                        # 2b. Overlapping Faculty Stamp
                        stamp_size = random.randint(165, 190)
                        stamp_img = create_circular_stamp(center_text="BAN CHỦ NHIỆM\nKHOA", size=(stamp_size, stamp_size))
                        stw, sth = stamp_img.size
                        st_x = int(anchor_center_x - stw / 2.0 + random.randint(-15, 15))
                        st_y = int(anchor.y1 * zoom + random.randint(4, 16))
                        st_x = max(20, min(img_w - stw - 20, st_x))
                        st_y = max(20, min(img_h - sth - 20, st_y))
                        img_pil.paste(stamp_img, (st_x, st_y), stamp_img)
                        raw_bboxes.append({
                            "box": [st_x, st_y, st_x + stw, st_y + sth],
                            "class_id": CLASS_MAP["stamp"],
                            "text": "stamp"
                        })
                        placed_stamps_count += 1
                        break

            # 3. PĐT Stamp on the right under "Phòng Đào tạo"
            pdt_rects = page.search_for("Phòng Đào tạo")
            if pdt_rects:
                bottom_pdt = [r for r in pdt_rects if r.y0 > 0.35 * ph and r.x0 > 0.48 * pw]
                if bottom_pdt:
                    anchor = bottom_pdt[-1]
                    stamp_size = random.randint(175, 200)
                    stamp_img = create_circular_stamp(center_text="PHÒNG\nĐÀO TẠO", size=(stamp_size, stamp_size))
                    stw, sth = stamp_img.size
                    anchor_center_x = (anchor.x0 + anchor.x1) / 2.0 * zoom
                    st_x = int(anchor_center_x - stw / 2.0 + random.randint(-6, 6))
                    st_y = int(anchor.y1 * zoom + random.randint(8, 20))
                    st_x = max(20, min(img_w - stw - 20, st_x))
                    st_y = max(20, min(img_h - sth - 20, st_y))
                    img_pil.paste(stamp_img, (st_x, st_y), stamp_img)
                    raw_bboxes.append({
                        "box": [st_x, st_y, st_x + stw, st_y + sth],
                        "class_id": CLASS_MAP["stamp"],
                        "text": "stamp"
                    })
                    placed_stamps_count += 1

        # Department / Faculty Stamp placement (Supports dual stamps if multiple authorities exist)
        # Note: Do not place stamps on Diem I form or BM12 (already handled)
        left_stamp_anchors = [
            "Ban Giám hiệu duyệt",
            "Ban Giám hiệu",
            "TRƯỞNG KHOA",
            "Trưởng khoa",
            "Ý kiến PĐT",
            "Xác nhận của PĐT",
            "Ý kiến của PĐT",
            "Xác nhận của Phòng Đào tạo",
            "Ý kiến của Phòng Đào tạo",
            "Chân thành cám ơn",
            "Chân thành cảm ơn",
            "Chân thành"
        ]
        right_stamp_anchors = [
            "Trưởng P.Đào tạo",
            "TRƯỞNG PHÒNG ĐÀO TẠO",
            "Đề xuất của P. Đào tạo"
        ]
        
        # Only place stamps if NOT Diem I form and NOT BM12
        if not is_diem_i_form and not is_thay_the_form:
            # 1. Left Authority Stamp
            for kw in left_stamp_anchors:
                kw_rects = page.search_for(kw)
                if kw_rects:
                    bottom_rects = [r for r in kw_rects if r.y0 > 0.35 * ph and r.x0 < 0.55 * pw]
                    if bottom_rects:
                        anchor = bottom_rects[-1]
                        stamp_size = random.randint(175, 205)
                        
                        if "Ban Giám hiệu" in kw:
                            c_text = "BAN\nGIÁM HIỆU"
                        elif "KHOA" in kw.upper() or "Khoa" in kw:
                            c_text = "BAN CHỦ NHIỆM\nKHOA"
                        else:
                            c_text = "PHÒNG\nĐÀO TẠO"
                            
                        stamp_img = create_circular_stamp(center_text=c_text, size=(stamp_size, stamp_size))
                        stw, sth = stamp_img.size
                        
                        anchor_center_x = (anchor.x0 + anchor.x1) / 2.0 * zoom
                        st_x = int(anchor_center_x - stw / 2.0 + random.randint(-6, 6))
                        st_y = int(anchor.y1 * zoom + random.randint(8, 20))
                        
                        st_x = max(20, min(img_w - stw - 20, st_x))
                        st_y = max(20, min(img_h - sth - 20, st_y))
                        
                        img_pil.paste(stamp_img, (st_x, st_y), stamp_img)
                        raw_bboxes.append({
                            "box": [st_x, st_y, st_x + stw, st_y + sth],
                            "class_id": CLASS_MAP["stamp"],
                            "text": "stamp"
                        })
                        placed_stamps_count += 1
                        break

            # 2. Right Authority Stamp (for forms with dual authority like Ban Giám hiệu + PĐT)
            has_bgh = bool(page.search_for("Ban Giám hiệu duyệt") or page.search_for("Ban Giám hiệu"))
            if has_bgh:
                for kw in right_stamp_anchors:
                    kw_rects = page.search_for(kw)
                    if kw_rects:
                        bottom_rects = [r for r in kw_rects if r.y0 > 0.35 * ph and r.x0 > 0.55 * pw]
                        if bottom_rects:
                            anchor = bottom_rects[-1]
                            stamp_size = random.randint(175, 205)
                            
                            c_text = "PHÒNG\nĐÀO TẠO"
                            stamp_img = create_circular_stamp(center_text=c_text, size=(stamp_size, stamp_size))
                            stw, sth = stamp_img.size
                            
                            anchor_center_x = (anchor.x0 + anchor.x1) / 2.0 * zoom
                            st_x = int(anchor_center_x - stw / 2.0 + random.randint(-6, 6))
                            st_y = int(anchor.y1 * zoom + random.randint(8, 20))
                            
                            st_x = max(20, min(img_w - stw - 20, st_x))
                            st_y = max(20, min(img_h - sth - 20, st_y))
                            
                            img_pil.paste(stamp_img, (st_x, st_y), stamp_img)
                            raw_bboxes.append({
                                "box": [st_x, st_y, st_x + stw, st_y + sth],
                                "class_id": CLASS_MAP["stamp"],
                                "text": "stamp"
                            })
                            placed_stamps_count += 1
                            break

        # Fallback if no specific anchor found on last page (skip for Diem I and Thay the)
        if not is_diem_i_form and not is_thay_the_form and placed_stamps_count == 0 and page_idx == len(doc) - 1:
            st_x, st_y = int(img_w * 0.18), int(img_h * 0.84)
            stamp_img = create_circular_stamp(size=(195, 195))
            stw, sth = stamp_img.size
            img_pil.paste(stamp_img, (st_x, st_y), stamp_img)
            raw_bboxes.append({
                "box": [st_x, st_y, st_x + stw, st_y + sth],
                "class_id": CLASS_MAP["stamp"],
                "text": "stamp"
            })

        # Deduplicate bboxes
        clean_bboxes = deduplicate_bboxes(raw_bboxes, iou_thresh=0.75)

        # 3. Apply Scan Effects & transform bboxes
        scanned_img, final_bboxes = apply_scan_effects(img_pil, clean_bboxes)
        
        # 4. Save Final Scan Image
        page_suffix = f"_p{page_idx+1}" if len(doc) > 1 else ""
        sample_name = f"{output_base_name}{page_suffix}"
        img_save_path = os.path.join(output_dir, "images", f"{sample_name}.png")
        scanned_img.save(img_save_path, "PNG")

        # 5. Write YOLO format label file
        label_save_path = os.path.join(output_dir, "labels", f"{sample_name}.txt")
        yolo_lines = []
        for b in final_bboxes:
            bx0, by0, bx1, by1 = b["box"]
            bw = bx1 - bx0
            bh = by1 - by0
            
            # Normalize to 0-1
            x_center = (bx0 + bw / 2.0) / img_w
            y_center = (by0 + bh / 2.0) / img_h
            norm_w = bw / img_w
            norm_h = bh / img_h
            
            # Clamp bounds
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.001, min(1.0, norm_w))
            norm_h = max(0.001, min(1.0, norm_h))
            
            yolo_lines.append(f"{b['class_id']} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        with open(label_save_path, "w", encoding="utf-8") as f_lbl:
            f_lbl.write("\n".join(yolo_lines) + "\n")

        # 6. Generate Visual Inspection Image (Drawing Bounding Boxes)
        vis_img = scanned_img.copy()
        draw = ImageDraw.Draw(vis_img)
        
        try:
            font_label = ImageFont.truetype("arialbd.ttf", 11)
        except:
            font_label = ImageFont.load_default()
            
        for b in final_bboxes:
            bx0, by0, bx1, by1 = b["box"]
            cls_id = b["class_id"]
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            color = CLASS_COLORS.get(cls_id, (255, 0, 0))
            
            # Draw rectangle
            draw.rectangle([bx0, by0, bx1, by1], outline=color, width=2)
            
            # Draw label tag
            tag_text = f"{cls_name}"
            tb = draw.textbbox((bx0, max(0, by0 - 14)), tag_text, font=font_label)
            draw.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1], fill=color)
            draw.text((bx0, max(0, by0 - 14)), tag_text, font=font_label, fill=(255, 255, 255))

        vis_save_path = os.path.join(output_dir, "visualizations", f"{sample_name}_annotated.png")
        vis_img.save(vis_save_path, "PNG")
        
        generated_files.append({
            "sample_name": sample_name,
            "image": img_save_path,
            "label": label_save_path,
            "visualization": vis_save_path,
            "num_bboxes": len(final_bboxes)
        })

    doc.close()
    return generated_files
