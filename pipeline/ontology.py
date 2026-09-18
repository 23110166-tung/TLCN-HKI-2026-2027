# -*- coding: utf-8 -*-
"""
Ontology and Class Definitions for YOLOv11 Academic Forms Dataset
"""

CLASS_NAMES = [
    "student_name",     # 0
    "student_id",       # 1
    "class_name",       # 2
    "faculty",          # 3
    "major",            # 4
    "dob",              # 5
    "pob",              # 6
    "phone",            # 7
    "email",            # 8
    "course_code",      # 9
    "course_name",      # 10
    "credits",          # 11
    "score",            # 12
    "date",             # 13
    "reason",           # 14
    "degree_info",      # 15
    "signature",        # 16
    "stamp"             # 17
]

CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

CLASS_COLORS = {
    0: (0, 102, 204),      # student_name: Blue
    1: (204, 0, 102),      # student_id: Magenta
    2: (0, 153, 76),       # class_name: Green
    3: (255, 128, 0),      # faculty: Orange
    4: (153, 51, 255),     # major: Purple
    5: (0, 204, 204),      # dob: Cyan
    6: (102, 102, 0),      # pob: Olive
    7: (255, 0, 0),        # phone: Red
    8: (0, 0, 153),        # email: Dark Blue
    9: (128, 0, 128),      # course_code: Purple
    10: (0, 128, 0),       # course_name: Dark Green
    11: (204, 102, 0),     # credits: Brown
    12: (204, 0, 0),       # score: Crimson
    13: (102, 0, 204),     # date: Indigo
    14: (0, 102, 102),     # reason: Teal
    15: (153, 0, 76),      # degree_info: Wine
    16: (0, 0, 255),       # signature: Royal Blue
    17: (255, 0, 0)        # stamp: Bright Red
}
