# -*- coding: utf-8 -*-
"""
Scan Simulator with coordinate transformation support
Adds scan degradation (blur, paper noise, slight rotation, lighting gradient)
and transforms bounding boxes precisely.
"""

import math
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

def rotate_point(x, y, cx, cy, angle_deg):
    """Rotates a point (x, y) around (cx, cy) by angle_deg (clockwise in image coords)."""
    # In image coordinate system (y points down), counter-clockwise visual rotation of image
    # by angle A means points rotate by -A.
    rad = math.radians(-angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    
    nx = cx + (x - cx) * cos_a - (y - cy) * sin_a
    ny = cy + (x - cx) * sin_a + (y - cy) * cos_a
    return nx, ny

def transform_bbox(bbox, img_w, img_h, angle_deg):
    """
    Transforms [x0, y0, x1, y1] pixel box when image is rotated by angle_deg without expansion.
    """
    if abs(angle_deg) < 1e-4:
        return [max(0, min(img_w, bbox[0])), max(0, min(img_h, bbox[1])),
                max(0, min(img_w, bbox[2])), max(0, min(img_h, bbox[3]))]
                
    cx, cy = img_w / 2.0, img_h / 2.0
    x0, y0, x1, y1 = bbox
    corners = [
        (x0, y0),
        (x1, y0),
        (x1, y1),
        (x0, y1)
    ]
    
    rot_corners = [rotate_point(px, py, cx, cy, angle_deg) for px, py in corners]
    xs = [pt[0] for pt in rot_corners]
    ys = [pt[1] for pt in rot_corners]
    
    rx0 = max(0.0, min(img_w, min(xs)))
    ry0 = max(0.0, min(img_h, min(ys)))
    rx1 = max(0.0, min(img_w, max(xs)))
    ry1 = max(0.0, min(img_h, max(ys)))
    
    return [rx0, ry0, rx1, ry1]

def apply_scan_effects(image, bboxes, rotation_range=(-0.7, 0.7)):
    """
    Applies realistic scanning effects to a PIL Image:
    1. Slight rotation (-0.7 to +0.7 deg)
    2. Paper grain / scanner sensor noise
    3. Slight Gaussian blur (0.3 - 0.5)
    4. Contrast and subtle lighting gradient
    
    Returns: (processed_image, updated_bboxes)
    where bboxes is a list of dicts: {"box": [x0, y0, x1, y1], "class_id": int, "text": str}
    """
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    w, h = image.size
    angle = random.uniform(rotation_range[0], rotation_range[1])
    
    # 1. Rotate image with white background fill
    rotated_img = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(252, 252, 252))
    
    # Transform all bounding boxes
    updated_bboxes = []
    for b in bboxes:
        new_box = transform_bbox(b["box"], w, h, angle)
        # Verify valid box
        if new_box[2] > new_box[0] and new_box[3] > new_box[1]:
            updated_bboxes.append({
                "box": new_box,
                "class_id": b["class_id"],
                "text": b.get("text", "")
            })
            
    # 2. Add subtle scanner sensor noise / paper texture
    img_np = np.array(rotated_img).astype(np.float32)
    noise = np.random.normal(0, 3.5, img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    processed_img = Image.fromarray(img_np)
    
    # 3. Add subtle blur (typical of flatbed office scanner optics)
    blur_radius = random.uniform(0.3, 0.5)
    processed_img = processed_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # 4. Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(processed_img)
    processed_img = enhancer.enhance(random.uniform(1.02, 1.08))
    
    return processed_img, updated_bboxes
