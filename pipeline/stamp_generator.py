# -*- coding: utf-8 -*-
"""
Procedural Generator for Realistic Stamps and Hand-drawn Signatures
Generates transparent RGBA PIL Images.
"""

import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def create_circular_stamp(
    outer_text="TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT TP.HCM",
    center_text="PHÒNG\nĐÀO TẠO",
    sub_text="★ TP. HỒ CHÍ MINH ★",
    size=(320, 320)
):
    """
    Creates a realistic Vietnamese official red round stamp with transparent background.
    """
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    w, h = size
    cx, cy = w / 2, h / 2
    r_outer = min(w, h) / 2 - 12
    r_inner = r_outer - 10
    
    # Stamp color (Official red/crimson ink)
    red_color = (215 + random.randint(-15, 15), 25 + random.randint(-10, 15), 25 + random.randint(-10, 15), 230)
    
    # Outer circle (thick)
    draw.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], outline=red_color, width=4)
    # Inner circle (thin)
    draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], outline=red_color, width=2)
    
    # Try loading Arial or standard bold font
    try:
        font_ring = ImageFont.truetype("arialbd.ttf", 18)
        font_center = ImageFont.truetype("arialbd.ttf", 20)
        font_sub = ImageFont.truetype("arialbd.ttf", 13)
    except:
        font_ring = ImageFont.load_default()
        font_center = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # Draw curved outer text along top arc
    text_chars = list(outer_text)
    n_chars = len(text_chars)
    if n_chars > 0:
        angle_span = 200 # degrees
        start_angle = 180 + (180 - angle_span) / 2
        r_text = (r_inner + r_outer) / 2 - 1
        
        for i, char in enumerate(text_chars):
            frac = i / max(1, n_chars - 1)
            angle_deg = start_angle + frac * angle_span
            angle_rad = math.radians(angle_deg)
            
            char_x = cx + r_text * math.cos(angle_rad)
            char_y = cy + r_text * math.sin(angle_rad)
            
            # Create a small image for rotated character
            char_img = Image.new("RGBA", (40, 40), (255, 255, 255, 0))
            char_draw = ImageDraw.Draw(char_img)
            char_draw.text((10, 10), char, font=font_ring, fill=red_color)
            
            rot_angle = -(angle_deg + 90)
            rotated_char = char_img.rotate(rot_angle, resample=Image.BICUBIC)
            
            img.paste(rotated_char, (int(char_x - 20), int(char_y - 20)), rotated_char)

    # Draw sub text along bottom arc
    sub_chars = list(sub_text)
    n_sub = len(sub_chars)
    if n_sub > 0:
        sub_span = 100
        start_sub = 90 - sub_span / 2
        r_sub = (r_inner + r_outer) / 2 - 2
        for i, char in enumerate(sub_chars):
            frac = i / max(1, n_sub - 1)
            angle_deg = start_sub + frac * sub_span
            angle_rad = math.radians(angle_deg)
            
            char_x = cx + r_sub * math.cos(angle_rad)
            char_y = cy + r_sub * math.sin(angle_rad)
            
            char_img = Image.new("RGBA", (36, 36), (255, 255, 255, 0))
            char_draw = ImageDraw.Draw(char_img)
            char_draw.text((10, 10), char, font=font_sub, fill=red_color)
            
            rot_angle = -(angle_deg - 90)
            rotated_char = char_img.rotate(rot_angle, resample=Image.BICUBIC)
            img.paste(rotated_char, (int(char_x - 18), int(char_y - 18)), rotated_char)

    # Draw Center Text
    lines = center_text.split("\n")
    total_h = len(lines) * 26
    start_y = cy - total_h / 2
    for idx, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font_center)
        tw = bbox[2] - bbox[0]
        tx = cx - tw / 2
        ty = start_y + idx * 26
        draw.text((tx, ty), line, font=font_center, fill=red_color)

    # Add realistic stamp ink noise and subtle texture
    arr = np.array(img)
    alpha = arr[:, :, 3]
    mask = alpha > 0
    
    # Salt and pepper ink dropout
    noise = np.random.rand(*alpha.shape)
    dropout = (noise < 0.12) & mask
    arr[:, :, 3][dropout] = np.random.randint(40, 160, size=np.sum(dropout))
    
    # Slight color variation
    ink_fade = (noise > 0.85) & mask
    arr[:, :, 0][ink_fade] = np.clip(arr[:, :, 0][ink_fade] + np.random.randint(-20, 20, size=np.sum(ink_fade)), 0, 255)
    
    stamped_img = Image.fromarray(arr)
    # Slight blur to soften edges like wet ink
    stamped_img = stamped_img.filter(ImageFilter.GaussianBlur(radius=0.4))
    
    # Random slight rotation of stamp
    stamped_img = stamped_img.rotate(random.uniform(-15, 15), resample=Image.BICUBIC, expand=True)
    return stamped_img


def create_handwritten_signature(name="Nguyễn Văn A", size=(280, 120)):
    """
    Generates a realistic smooth bezier handwritten signature in dark blue or black ink.
    """
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    w, h = size
    
    # Ink color (ballpoint pen blue / dark navy / black)
    ink_colors = [
        (10, 30, 140, 240),
        (20, 50, 170, 235),
        (15, 25, 80, 245),
        (25, 25, 25, 240)
    ]
    color = random.choice(ink_colors)
    
    # Generate smooth control points for signature stroke
    num_points = random.randint(14, 22)
    xs = np.linspace(25, w - 25, num_points)
    ys = [h/2 + random.uniform(-h*0.35, h*0.35) for _ in range(num_points)]
    
    # Add energetic loops and ascender/descenders
    for i in range(1, num_points - 1):
        if i % 3 == 0:
            ys[i] -= random.uniform(20, 45) # ascender loop
        elif i % 4 == 0:
            ys[i] += random.uniform(15, 35) # descender stroke
            
    # Interpolate with cubic spline / smooth curve
    t_vals = np.linspace(0, 1, 150)
    # Parametric smoothing
    points = []
    for k in range(len(xs) - 1):
        p0 = (xs[max(0, k - 1)], ys[max(0, k - 1)])
        p1 = (xs[k], ys[k])
        p2 = (xs[k + 1], ys[k + 1])
        p3 = (xs[min(len(xs) - 1, k + 2)], ys[min(len(xs) - 1, k + 2)])
        
        # Catmull-Rom spline segment
        for step in range(12):
            t = step / 12.0
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * ((2*p1[0]) + (-p0[0] + p2[0])*t + (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0])*t2 + (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0])*t3)
            y = 0.5 * ((2*p1[1]) + (-p0[1] + p2[1])*t + (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1])*t2 + (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1])*t3)
            points.append((x, y))
            
    # Draw strokes with variable thickness
    for i in range(len(points) - 1):
        pt1 = points[i]
        pt2 = points[i + 1]
        width = random.choice([2, 3, 3, 4])
        draw.line([pt1, pt2], fill=color, width=width)
        
    # Underline flourish
    flourish_start = (random.uniform(20, 50), random.uniform(h*0.75, h*0.9))
    flourish_mid = (w * 0.55, random.uniform(h*0.65, h*0.8))
    flourish_end = (random.uniform(w*0.8, w*0.95), random.uniform(h*0.7, h*0.9))
    
    f_points = []
    for t in np.linspace(0, 1, 30):
        # Quadratic bezier
        bx = (1-t)**2 * flourish_start[0] + 2*(1-t)*t * flourish_mid[0] + t**2 * flourish_end[0]
        by = (1-t)**2 * flourish_start[1] + 2*(1-t)*t * flourish_mid[1] + t**2 * flourish_end[1]
        f_points.append((bx, by))
        
    for i in range(len(f_points) - 1):
        draw.line([f_points[i], f_points[i+1]], fill=color, width=random.choice([2, 3]))

    # Soften
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    # Random slant
    img = img.rotate(random.uniform(-4, 4), resample=Image.BICUBIC, expand=True)
    return img
