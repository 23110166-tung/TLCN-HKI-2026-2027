# -*- coding: utf-8 -*-
"""
Module: Auto Document Scanner & Perspective Rectification (Thử nghiệm)
Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh

Tính năng tiền xử lý tài liệu hành chính:
1. Phát hiện 4 góc tờ giấy và nắn phẳng phối cảnh (Perspective Warping / Dewarping).
2. Khử bóng đổ không đều và chuẩn hóa độ sáng nền giấy (Shadow & Illumination Flattening).
3. Bảo toàn nguyên vẹn màu mực con dấu đỏ và chữ ký viết tay.
"""

import cv2
import numpy as np
from PIL import Image


def order_points(pts):
    """
    Sắp xếp 4 đỉnh theo thứ tự chuẩn: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # Top-left (tổng x+y nhỏ nhất)
    rect[2] = pts[np.argmax(s)]       # Bottom-right (tổng x+y lớn nhất)

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    # Top-right (hiệu y-x nhỏ nhất)
    rect[3] = pts[np.argmax(diff)]    # Bottom-left (hiệu y-x lớn nhất)
    return rect


def four_point_transform(image_bgr, pts):
    """
    Biến đổi phối cảnh (Perspective Transform) duỗi thẳng tài liệu về hình chữ nhật chuẩn A4.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Chiều rộng mới
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Chiều cao mới
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth < 100 or maxHeight < 100:
        return image_bgr

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_bgr, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)
    return warped


def remove_document_shadows(image_bgr):
    """
    Khử bóng đổ và cân bằng độ chiếu sáng nền giấy (Shadow & Uneven Illumination Removal):
    Ước tính mặt chiếu sáng nền theo từng kênh để bảo toàn màu mộc đỏ và chữ ký.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
    result_planes = []

    for plane in cv2.split(image_bgr):
        dilated = cv2.morphologyEx(plane, cv2.MORPH_DILATE, kernel)
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # Pha trộn mềm để giữ độ tự nhiên của văn bản in
        blended = cv2.addWeighted(plane, 0.40, norm, 0.60, 0)
        result_planes.append(blended)

    cleaned = cv2.merge(result_planes)
    return cleaned


class AutoDocumentScanner:
    """
    Bộ quét và chỉnh phẳng tài liệu thông minh (Smart Document Scanner)
    """

    def __init__(self, target_scale_h=900.0):
        self.target_scale_h = target_scale_h

    def detect_paper_corners(self, image_bgr):
        """
        Tìm 4 góc tờ giấy trong bức ảnh chụp.
        """
        h_orig, w_orig = image_bgr.shape[:2]
        ratio = h_orig / float(self.target_scale_h)
        scaled_w = int(w_orig / ratio)
        scaled_h = int(self.target_scale_h)

        resized = cv2.resize(image_bgr, (scaled_w, scaled_h))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thử nhiều cấu hình dò cạnh
        for low_th, high_th in [(40, 140), (25, 90), (15, 60)]:
            edged = cv2.Canny(blurred, low_th, high_th)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

            for c in contours:
                area = cv2.contourArea(c)
                # Tờ giấy tài liệu phải chiếm tối thiểu 30% diện tích ảnh
                if area < (scaled_w * scaled_h * 0.30):
                    continue
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                if len(approx) == 4 and cv2.isContourConvex(approx):
                    # Khôi phục tọa độ về kích thước ảnh gốc
                    pts = approx.reshape(4, 2).astype("float32") * ratio
                    return pts

        return None

    def scan(self, input_image, auto_warp=True, auto_shadow_removal=True):
        """
        Thực hiện tiền xử lý quét tài liệu hoàn chỉnh.
        Đầu vào: PIL Image hoặc np.ndarray (RGB/BGR).
        Đầu ra: (PIL.Image đã quét, metadata_dict).
        """
        if isinstance(input_image, Image.Image):
            rgb_np = np.array(input_image.convert("RGB"))
            bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        else:
            bgr_np = input_image.copy()

        corners = None
        was_warped = False
        processed_bgr = bgr_np

        # 1. Phát hiện 4 góc và nắn phẳng phối cảnh (Perspective Dewarp)
        if auto_warp:
            corners = self.detect_paper_corners(bgr_np)
            if corners is not None:
                processed_bgr = four_point_transform(bgr_np, corners)
                was_warped = True

        # 2. Khử bóng đổ và cân bằng sáng nền giấy (Shadow & Lighting Normalization)
        was_shadow_removed = False
        if auto_shadow_removal:
            processed_bgr = remove_document_shadows(processed_bgr)
            was_shadow_removed = True

        # Chuyển về định dạng PIL RGB
        out_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        out_pil = Image.fromarray(out_rgb)

        metadata = {
            "perspective_warped": was_warped,
            "corners_detected": corners.tolist() if corners is not None else None,
            "shadow_removed": was_shadow_removed,
            "original_size": (bgr_np.shape[1], bgr_np.shape[0]),
            "scanned_size": out_pil.size
        }

        return out_pil, metadata


# Singleton instance
default_scanner = AutoDocumentScanner()

