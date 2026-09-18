# -*- coding: utf-8 -*-
"""
Mock Data Generator for Vietnamese Academic Forms (HCMUTE / University forms)
"""
import random
from datetime import datetime, timedelta

FIRST_NAMES_MALE = ["Văn", "Đức", "Minh", "Hoàng", "Quốc", "Gia", "Thành", "Hữu", "Anh", "Tuấn", "Huy", "Hải", "Khánh", "Trọng", "Phúc", "Duy"]
FIRST_NAMES_FEMALE = ["Thị", "Ngọc", "Thảo", "Phương", "Quỳnh", "Thùy", "Ánh", "Hương", "Mai", "Trang", "Linh", "Thanh", "Bích", "Tuyết", "Như"]
MIDDLE_NAMES = ["Bảo", "Hải", "Tuấn", "Minh", "Ngọc", "Văn", "Đình", "Nhật", "Hoài", "Khánh", "Hồng", "Kim", "Xuân", "Thanh"]
LAST_NAMES = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý", "Đinh", "Trịnh"]

PLACES_OF_BIRTH = [
    "TP. Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Cần Thơ", "Hải Phòng",
    "Đồng Nai", "Bình Dương", "Bà Rịa - Vũng Tàu", "Long An", "Tiền Giang",
    "Bến Tre", "Đồng Tháp", "Vĩnh Long", "An Giang", "Kiên Giang",
    "Lâm Đồng", "Đắk Lắk", "Gia Lai", "Bình Định", "Khánh Hòa",
    "Quảng Nam", "Quảng Ngãi", "Huế", "Nghệ An", "Thanh Hóa"
]

FACULTIES_AND_MAJORS = [
    {
        "faculty": "Khoa Công nghệ Thông tin",
        "majors": ["Công nghệ Thông tin", "Kỹ thuật Phần mềm", "Trí tuệ Nhân tạo", "An toàn Thông tin", "Hệ thống Thông tin"],
        "class_prefix": "21110"
    },
    {
        "faculty": "Khoa Điện - Điện tử",
        "majors": ["Kỹ thuật Điện tử - Viễn thông", "Kỹ thuật Điều khiển & Tự động hóa", "Hệ thống Nhúng và IoT", "Kỹ thuật Điện"],
        "class_prefix": "21151"
    },
    {
        "faculty": "Khoa Cơ khí Chế tạo máy",
        "majors": ["Kỹ thuật Cơ khí", "Kỹ thuật Cơ điện tử", "Kỹ thuật Robot & Trí tuệ Nhân tạo", "Kỹ thuật Công nghiệp"],
        "class_prefix": "21141"
    },
    {
        "faculty": "Khoa Cơ khí Động lực",
        "majors": ["Công nghệ Kỹ thuật Ô tô", "Kỹ thuật Năng lượng Tái tạo", "Kỹ thuật Nhiệt"],
        "class_prefix": "21145"
    },
    {
        "faculty": "Khoa Kinh tế",
        "majors": ["Quản trị Kinh doanh", "Kế toán", "Thương mại Điện tử", "Logistics và Quản lý Chuỗi cung ứng"],
        "class_prefix": "21131"
    },
    {
        "faculty": "Khoa Công nghệ Hóa học & Thực phẩm",
        "majors": ["Công nghệ Thực phẩm", "Công nghệ Kỹ thuật Hóa học", "Công nghệ Môi trường"],
        "class_prefix": "21116"
    }
]

COURSES_POOL = [
    {"code": "PRTE230385", "name": "Kỹ thuật lập trình", "credits": 3},
    {"code": "COSC130185", "name": "Nhập môn công nghệ thông tin", "credits": 3},
    {"code": "MATH141601", "name": "Toán cao cấp 1", "credits": 4},
    {"code": "MATH141701", "name": "Toán cao cấp 2", "credits": 4},
    {"code": "PHYS130902", "name": "Vật lý đại cương 1", "credits": 3},
    {"code": "DBMS330284", "name": "Hệ quản trị cơ sở dữ liệu", "credits": 3},
    {"code": "NETW330480", "name": "Mạng máy tính", "credits": 3},
    {"code": "DSAA330584", "name": "Cấu trúc dữ liệu và giải thuật", "credits": 3},
    {"code": "OOPL330184", "name": "Lập trình hướng đối tượng", "credits": 3},
    {"code": "SOEN330680", "name": "Công nghệ phần mềm", "credits": 3},
    {"code": "AI430185", "name": "Trí tuệ nhân tạo", "credits": 3},
    {"code": "ENGL130137", "name": "Anh văn 1", "credits": 3},
    {"code": "ENGL230237", "name": "Anh văn 2", "credits": 3},
    {"code": "ENGL330337", "name": "Anh văn 3", "credits": 3},
    {"code": "LLCT130105", "name": "Triết học Mác - Lênin", "credits": 3},
    {"code": "LLCT120205", "name": "Kinh tế chính trị Mác - Lênin", "credits": 2},
    {"code": "LLCT120405", "name": "Chủ nghĩa xã hội khoa học", "credits": 2},
    {"code": "EEEN230140", "name": "Kỹ thuật điện tử căn bản", "credits": 3},
    {"code": "MECH130120", "name": "Cơ học lý thuyết", "credits": 3},
    {"code": "STAT230101", "name": "Xác suất thống kê", "credits": 3},
    {"code": "WEBP330880", "name": "Lập trình Web", "credits": 3},
    {"code": "MOBI330980", "name": "Lập trình thiết bị di động", "credits": 3},
    {"code": "DISM230201", "name": "Toán rời rạc", "credits": 3},
    {"code": "COMP140185", "name": "Kiến trúc máy tính", "credits": 4}
]

REASONS = [
    "Trùng lịch thi đột xuất với kỳ thi cấp chứng chỉ quốc tế",
    "Bị sốt xuất huyết phải nhập viện điều trị theo chỉ định bác sĩ",
    "Gia đình có việc hiếu đột xuất ở quê phải về gấp",
    "Chuyển ngành học để phù hợp với định hướng nghề nghiệp",
    "Đi thực tập thực tế tại doanh nghiệp theo phân công của khoa",
    "Bị tai nạn giao thông phải điều trị ngoại trú",
    "Nguyện vọng nâng cao trình độ chuyên môn tích lũy hai văn bằng",
    "Đã hoàn thành chứng chỉ ngoại ngữ quốc tế tương đương chuẩn đầu ra",
    "Xin học môn thay thế do môn học trong chương trình cũ không còn mở lớp",
    "Bảo lưu kết quả học tập để điều trị bệnh phục hồi sức khỏe"
]

PREV_UNIVERSITIES = [
    "ĐH Bách Khoa ĐHQG-HCM",
    "ĐH Khoa học Tự nhiên ĐHQG-HCM",
    "ĐH Công nghiệp TP.HCM",
    "ĐH Sài Gòn",
    "ĐH Tôn Đức Thắng",
    "ĐH Giao thông Vận tải TP.HCM",
    "ĐH Cần Thơ",
    "CĐ Kỹ thuật Cao Thắng"
]

def generate_random_name():
    is_male = random.random() < 0.6
    last = random.choice(LAST_NAMES)
    mid = random.choice(MIDDLE_NAMES)
    first = random.choice(FIRST_NAMES_MALE if is_male else FIRST_NAMES_FEMALE)
    return f"{last} {mid} {first}"

def generate_student_id(cohort_year=None):
    if cohort_year is not None:
        year = str(cohort_year)[-2:]
    else:
        year = random.choice(["20", "21", "22", "23", "24", "25", "26"])
    faculty_num = random.choice([110, 115, 141, 145, 131, 116, 119])
    seq = random.randint(101, 999)
    return f"{year}{faculty_num}{seq:03d}"

def generate_class_name(student_id):
    year = student_id[:2]
    faculty_code = student_id[2:5]
    section = random.choice(["CLA", "CL1", "CL2", "01A", "02B", "1A", "2A"])
    return f"{year}{faculty_code}{section}"

def generate_phone():
    prefixes = ["090", "091", "093", "094", "096", "097", "098", "035", "036", "037", "077", "078", "079", "083", "085"]
    prefix = random.choice(prefixes)
    suffix = random.randint(1000000, 9999999)
    return f"{prefix}{suffix}"

def generate_dob(cohort_year=None):
    if cohort_year is not None:
        yr_val = int(str(cohort_year)[-2:])
        # 18 years old when enrolling: year = (2000 + yr_val) - 18
        year = 2000 + yr_val - 18
    else:
        year = random.randint(2001, 2008)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{day:02d}/{month:02d}/{year}"

def generate_date(offset_days_range=(-60, 0)):
    base_date = datetime(2026, 8, 20)
    delta = random.randint(offset_days_range[0], offset_days_range[1])
    dt = base_date + timedelta(days=delta)
    return dt.strftime("%d/%m/%Y"), f"{dt.day:02d}", f"{dt.month:02d}", f"{dt.year}"

def generate_mock_student(cohort_year=None):
    name = generate_random_name()
    sid = generate_student_id(cohort_year)
    cname = generate_class_name(sid)
    fac_group = random.choice(FACULTIES_AND_MAJORS)
    faculty = fac_group["faculty"]
    major = random.choice(fac_group["majors"])
    dob = generate_dob(cohort_year)
    pob = random.choice(PLACES_OF_BIRTH)
    phone = generate_phone()
    email = f"{sid}@student.hcmute.edu.vn"
    
    cpa = round(random.uniform(6.5, 9.2), 2)
    cpa_4 = round(random.uniform(2.5, 3.85), 2)
    credits_accum = random.randint(45, 135)
    credits_term = random.randint(14, 24)
    term_gpa = round(random.uniform(6.8, 9.4), 2)
    
    # Second major for dual program
    other_fac_group = random.choice([g for g in FACULTIES_AND_MAJORS if g["faculty"] != faculty])
    second_faculty = other_fac_group["faculty"]
    second_major = random.choice(other_fac_group["majors"])
    
    # Courses
    num_courses = random.randint(3, 5)
    courses = random.sample(COURSES_POOL, num_courses)
    course_data = []
    for c in courses:
        score = round(random.uniform(5.5, 9.5), 1)
        course_data.append({
            "code": c["code"],
            "name": c["name"],
            "credits": c["credits"],
            "score": score,
            "group": f"{random.randint(1, 15):02d}",
            "exam_date": f"{random.randint(1, 28):02d}/{random.randint(9, 12):02d}/2026",
            "room": f"{random.choice(['A1', 'A2', 'B1', 'B2', 'C1'])}-{random.randint(101, 505)}",
            "period": f"{random.choice(['1-3', '4-6', '7-9', '10-12'])}"
        })
        
    # Certificate info (TOEIC or IELTS)
    cert_type = random.choice(["TOEIC", "TOEIC", "IELTS"])
    if cert_type == "TOEIC":
        # Realistic TOEIC Listening & Reading (5 to 495 each, step 5)
        # Total scores common for graduation: 500, 550, 600, 650, 700, 750, 800
        cert_total_val = random.choice([500, 550, 600, 650, 700, 750])
        # Split into listening and reading
        listening_val = random.randint(int(cert_total_val * 0.48 / 5), int(cert_total_val * 0.55 / 5)) * 5
        listening_val = min(495, max(150, listening_val))
        reading_val = cert_total_val - listening_val
        
        cert_name = "TOEIC Quốc tế (IIG Việt Nam)"
        cert_listening = str(listening_val)
        cert_speaking = "---"
        cert_reading = str(reading_val)
        cert_writing = "---"
        cert_total = str(cert_total_val)
        cert_place = "IIG Việt Nam - Chi nhánh TP.HCM"
    else:
        # IELTS Academic (0.0 to 9.0 in steps of 0.5)
        ielts_bands = [5.0, 5.5, 6.0, 6.5, 7.0]
        overall = random.choice(ielts_bands)
        l_band = random.choice([overall - 0.5, overall, overall + 0.5])
        r_band = random.choice([overall - 0.5, overall, overall + 0.5])
        s_band = random.choice([overall - 0.5, overall, overall + 0.5])
        w_band = round((overall * 4 - (l_band + r_band + s_band)) * 2) / 2
        w_band = min(9.0, max(4.5, w_band))
        
        provider = random.choice(["IDP Education", "Hội đồng Anh (British Council)"])
        cert_name = f"IELTS Academic ({provider})"
        cert_listening = f"{l_band:.1f}"
        cert_speaking = f"{s_band:.1f}"
        cert_reading = f"{r_band:.1f}"
        cert_writing = f"{w_band:.1f}"
        cert_total = f"{overall:.1f}"
        cert_place = f"{provider} - Chi nhánh TP.HCM"

    date_full, day_str, month_str, year_str = generate_date()
    
    return {
        "student_name": name,
        "student_id": sid,
        "class_name": cname,
        "faculty": faculty,
        "major": major,
        "dob": dob,
        "pob": pob,
        "phone": phone,
        "email": email,
        "cpa": str(cpa),
        "cpa_4": str(cpa_4),
        "credits_accum": str(credits_accum),
        "credits_term": str(credits_term),
        "term_gpa": str(term_gpa),
        "second_faculty": second_faculty,
        "second_major": second_major,
        "second_curriculum": f"Cử nhân {second_major}",
        "reason": random.choice(REASONS),
        "prev_university": random.choice(PREV_UNIVERSITIES),
        "prev_major": major,
        "prev_class": f"DH{sid[2:5]}A",
        "prev_student_id": f"19{sid[2:]}",
        "degree_number": f"B{random.randint(100000, 999999)}",
        "book_number": f"{random.randint(100, 999)}/QĐ-ĐHSPKT",
        "cert_type": cert_type,
        "cert_name": cert_name,
        "cert_listening": cert_listening,
        "cert_speaking": cert_speaking,
        "cert_reading": cert_reading,
        "cert_writing": cert_writing,
        "cert_total": cert_total,
        "cert_place": cert_place,
        "cert_date": f"{random.randint(1, 28):02d}/{random.randint(1, 8):02d}/2026",
        "date_str": date_full,
        "day": day_str,
        "month": month_str,
        "year": year_str,
        "semester": random.choice(["1", "2", "Hè"]),
        "academic_year": "2025-2026",
        "courses": course_data
    }
