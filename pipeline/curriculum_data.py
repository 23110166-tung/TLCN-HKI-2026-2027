# -*- coding: utf-8 -*-
"""
Rich Curriculum Repository for 8 Engineering, Technology, and Business Majors at HCMUTE.
Provides full 8-semester course sequences with realistic codes, credits, and subject names.
"""

CURRICULUM_DATABASE = {
    "AI": {
        "faculty": "Khoa Công nghệ Thông tin",
        "program_name": "Trí tuệ nhân tạo",
        "code_prefix": "AI",
        "semesters": [
            ("Năm học: {y0}-{y1}", "Học kỳ: HK01", [
                ("EHQT130137", "Anh văn 1", 3),
                ("MATH143001", "Đại số tuyến tính và cấu trúc đại số", 4),
                ("PHED110513", "Giáo dục thể chất 1 *", 1),
                ("INPR130285", "Nhập môn lập trình", 3),
                ("INIT130185", "Nhập môn ngành CNTT", 3),
                ("GELA220405", "Pháp luật đại cương", 2),
                ("MATH132401", "Toán 1", 3),
                ("LLCT130105", "Triết học Mác - Lênin", 3)
            ]),
            ("Năm học: {y0}-{y1}", "Học kỳ: HK02", [
                ("LLCT120405", "Chủ nghĩa xã hội khoa học", 2),
                ("GDQP008031", "Giáo dục quốc phòng 1(ĐH) *", 1),
                ("GDQP008032", "Giáo dục quốc phòng 2(ĐH) *", 1),
                ("GDQP008033", "Giáo dục quốc phòng 3(ĐH) *", 2),
                ("PHED110613", "Giáo dục thể chất 2 *", 1),
                ("LLCT120205", "Kinh tế chính trị Mác - Lênin", 2),
                ("PRTE230385", "Kỹ thuật lập trình", 3),
                ("PHYS111202", "Thí nghiệm Vật lý 1 (Thực hành)", 1),
                ("MATH132501", "Toán 2", 3),
                ("SYTH220491", "Tư duy hệ thống", 2),
                ("PHYS130902", "Vật lý 1", 3),
                ("MATH132901", "Xác suất thống kê ứng dụng", 3)
            ]),
            ("Năm học: {y1}-{y2}", "Học kỳ: HK01", [
                ("DASA230179", "Cấu trúc dữ liệu và giải thuật", 3),
                ("DBSY230184", "Cơ sở dữ liệu", 3),
                ("KARA112330", "Không thủ đạo *", 1),
                ("OOPR230279", "Lập trình hướng đối tượng", 3),
                ("INMA220305", "Nhập môn quản trị học", 2),
                ("INSO321005", "Nhập môn xã hội học", 2),
                ("DIGR230485", "Toán rời rạc và lý thuyết đồ thị", 3),
                ("LLCT120314", "Tư tưởng Hồ Chí Minh", 2)
            ]),
            ("Năm học: {y1}-{y2}", "Học kỳ: HK02", [
                ("EHQT130237", "Anh văn 2", 3),
                ("CLCO332779", "Điện toán đám mây", 3),
                ("EEEN231780", "Điện tử căn bản", 3),
                ("OPSY330280", "Hệ điều hành", 3),
                ("CAAL230180", "Kiến trúc máy tính và hợp ngữ", 3),
                ("WIPR230579", "Lập trình trên Windows", 3),
                ("LLCT220514", "Lịch sử Đảng CSVN", 2),
                ("NEES330380", "Mạng máy tính căn bản", 3)
            ]),
            ("Năm học: {y2}-{y3}", "Học kỳ: HK01", [
                ("INSE330380", "An toàn thông tin", 3),
                ("PROJ215879", "Đồ án CNTT", 1),
                ("DBMS330284", "Hệ quản trị cơ sở dữ liệu", 3),
                ("IPPA233277", "Lập trình Python", 3),
                ("WEPR330479", "Lập trình Web", 3),
                ("PRBE214262", "Thực tập điện tử căn bản (Thực hành)", 1),
                ("ARIN330585", "Trí tuệ nhân tạo", 3)
            ]),
            ("Năm học: {y2}-{y3}", "Học kỳ: HK02", [
                ("MALE431085", "Học máy", 3),
                ("DLEA432085", "Học sâu", 3),
                ("MOPR331279", "Lập trình di động", 3),
                ("INDS331085", "Nhập môn Khoa học dữ liệu", 3),
                ("MAAI330985", "Toán cho trí tuệ nhân tạo", 3),
                ("DIPR430685", "Xử lý ảnh số", 3),
                ("NLPR431585", "Xử lý ngôn ngữ tự nhiên", 3)
            ]),
            ("Năm học: {y3}-{y4}", "Học kỳ: HK01", [
                ("ITEN420885", "Chuyên đề Doanh nghiệp", 2),
                ("AIOT331185", "Trí tuệ nhân tạo cho IOT", 3),
                ("BDPR431385", "Xử lý dữ liệu lớn", 3),
                ("INPR431085", "Thực tập tốt nghiệp", 3)
            ]),
            ("Năm học: {y3}-{y4}", "Học kỳ: HK02", [
                ("GRAD400185", "Khóa luận tốt nghiệp", 10)
            ])
        ]
    },
    "CNTT": {
        "faculty": "Khoa Công nghệ Thông tin",
        "program_name": "Công nghệ thông tin",
        "code_prefix": "CNTT",
        "semesters": [
            ("Năm học: {y0}-{y1}", "Học kỳ: HK01", [
                ("ENGL130137", "Anh văn 1", 3),
                ("MATH143001", "Đại số tuyến tính", 3),
                ("INPR130285", "Nhập môn lập trình", 3),
                ("INIT130185", "Nhập môn ngành CNTT", 3),
                ("MATH132401", "Giải tích 1", 3),
                ("LLCT130105", "Triết học Mác - Lênin", 3),
                ("PHED110513", "Giáo dục thể chất 1 *", 1)
            ]),
            ("Năm học: {y0}-{y1}", "Học kỳ: HK02", [
                ("ENGL230237", "Anh văn 2", 3),
                ("PRTE230385", "Kỹ thuật lập trình", 3),
                ("MATH132501", "Giải tích 2", 3),
                ("MATH132901", "Xác suất thống kê", 3),
                ("PHYS130902", "Vật lý đại cương", 3),
                ("LLCT120205", "Kinh tế chính trị Mác - Lênin", 2),
                ("PHED110613", "Giáo dục thể chất 2 *", 1)
            ]),
            ("Năm học: {y1}-{y2}", "Học kỳ: HK01", [
                ("DASA230179", "Cấu trúc dữ liệu và giải thuật", 3),
                ("DBSY230184", "Cơ sở dữ liệu", 3),
                ("OOPR230279", "Lập trình hướng đối tượng", 3),
                ("CAAL230180", "Kiến trúc máy tính", 3),
                ("DIGR230485", "Toán rời rạc", 3),
                ("LLCT120405", "Chủ nghĩa xã hội khoa học", 2)
            ]),
            ("Năm học: {y1}-{y2}", "Học kỳ: HK02", [
                ("OPSY330280", "Hệ điều hành", 3),
                ("NEES330380", "Mạng máy tính", 3),
                ("WEPR330479", "Lập trình Web", 3),
                ("SEEN330180", "Công nghệ phần mềm", 3),
                ("DBMS330284", "Hệ quản trị CSDL", 3),
                ("LLCT120314", "Tư tưởng Hồ Chí Minh", 2)
            ]),
            ("Năm học: {y2}-{y3}", "Học kỳ: HK01", [
                ("INSE330380", "An toàn thông tin", 3),
                ("MOPR331279", "Lập trình di động", 3),
                ("SOAR330580", "Kiến trúc phần mềm", 3),
                ("SEPR330880", "Đồ án Công nghệ phần mềm", 2),
                ("WEAD330779", "Lập trình Web nâng cao", 3),
                ("LLCT220514", "Lịch sử Đảng CSVN", 2)
            ]),
            ("Năm học: {y2}-{y3}", "Học kỳ: HK02", [
                ("CLCO332779", "Điện toán đám mây", 3),
                ("DATA330980", "Khai phá dữ liệu", 3),
                ("DEVO430180", "DevOps căn bản", 3),
                ("PROJ430280", "Đồ án chuyên ngành CNTT", 2),
                ("MALE431085", "Nhập môn Trí tuệ nhân tạo", 3)
            ]),
            ("Năm học: {y3}-{y4}", "Học kỳ: HK01", [
                ("INTR430180", "Thực tập tốt nghiệp", 3),
                ("SEMI420280", "Chuyên đề tốt nghiệp", 3),
                ("ITMA430380", "Quản trị dự án CNTT", 3)
            ]),
            ("Năm học: {y3}-{y4}", "Học kỳ: HK02", [
                ("THES400180", "Khóa luận tốt nghiệp CNTT", 10)
            ])
        ]
    },
    "DDT": {
        "faculty": "Khoa Điện - Điện tử",
        "program_name": "Điện tử - Viễn thông",
        "code_prefix": "DDT",
        "semesters": [
            ("Năm học: {y0}-{y1}", "Học kỳ: HK01", [
                ("ENGL130137", "Anh văn 1", 3),
                ("MATH143001", "Đại số tuyến tính", 3),
                ("MATH132401", "Toán cao cấp 1", 3),
                ("PHYS130902", "Vật lý đại cương 1", 3),
                ("INPR130285", "Nhập môn lập trình C", 3),
                ("LLCT130105", "Triết học Mác - Lênin", 3),
                ("PHED110513", "Giáo dục thể chất 1 *", 1)
            ]),
            ("Năm học: {y0}-{y1}", "Học kỳ: HK02", [
                ("ENGL230237", "Anh văn 2", 3),
                ("MATH132501", "Toán cao cấp 2", 3),
                ("EEEN231780", "Mạch điện 1", 3),
                ("ELCI230180", "Linh kiện điện tử", 3),
                ("PHYS111202", "Thí nghiệm Vật lý", 1),
                ("LLCT120205", "Kinh tế chính trị Mác - Lênin", 2),
                ("PHED110613", "Giáo dục thể chất 2 *", 1)
            ]),
            ("Năm học: {y1}-{y2}", "Học kỳ: HK01", [
                ("DICI230280", "Mạch điện tử số", 3),
                ("ANCI230380", "Mạch điện tử tương tự", 3),
                ("SIGS330180", "Tín hiệu và hệ thống", 3),
                ("EEEN330580", "Mạch điện 2", 3),
                ("LLCT120405", "Chủ nghĩa xã hội khoa học", 2)
            ]),
            ("Năm học: {y1}-{y2}", "Học kỳ: HK02", [
                ("MICR330480", "Vi xử lý - Vi điều khiển", 3),
                ("COSY330280", "Hệ thống viễn thông", 3),
                ("DSIP330680", "Xử lý tín hiệu số DSP", 3),
                ("ACOM330880", "Truyền thông vô tuyến", 3),
                ("LLCT120314", "Tư tưởng Hồ Chí Minh", 2)
            ]),
            ("Năm học: {y2}-{y3}", "Học kỳ: HK01", [
                ("EMBS330980", "Hệ thống nhúng căn bản", 3),
                ("RFEN331080", "Kỹ thuật cao tần và Anten", 3),
                ("PRDT321180", "Đồ án Điện tử viễn thông 1", 2),
                ("OPCO331280", "Thông tin quang", 3),
                ("LLCT220514", "Lịch sử Đảng CSVN", 2)
            ]),
            ("Năm học: {y2}-{y3}", "Học kỳ: HK02", [
                ("IOTP430180", "Thiết kế hệ thống IoT", 3),
                ("NETW430280", "Mạng truyền thông công nghiệp", 3),
                ("PRDT420380", "Đồ án Điện tử viễn thông 2", 2),
                ("VLSI430480", "Thiết kế vi mạch VLSI", 3)
            ]),
            ("Năm học: {y3}-{y4}", "Học kỳ: HK01", [
                ("INTR430580", "Thực tập tốt nghiệp", 3),
                ("SEMI420680", "Chuyên đề viễn thông", 3)
            ]),
            ("Năm học: {y3}-{y4}", "Học kỳ: HK02", [
                ("THES400180", "Đồ án tốt nghiệp Điện tử", 10)
            ])
        ]
    },
    "CK": {
        "faculty": "Khoa Cơ khí Chế tạo máy",
        "program_name": "Cơ khí chế tạo máy",
        "code_prefix": "CK",
        "semesters": [
            ("Năm học: {y0}-{y1}", "Học kỳ: HK01", [
                ("ENGL130137", "Anh văn 1", 3),
                ("MATH132401", "Toán cao cấp 1", 3),
                ("PHYS130902", "Vật lý 1", 3),
                ("MECH130120", "Cơ học lý thuyết", 3),
                ("DRWG130220", "Vẽ kỹ thuật cơ khí", 3),
                ("LLCT130105", "Triết học Mác - Lênin", 3),
                ("PHED110513", "Giáo dục thể chất 1 *", 1)
            ]),
            ("Năm học: {y0}-{y1}", "Học kỳ: HK02", [
                ("ENGL230237", "Anh văn 2", 3),
                ("MATH132501", "Toán cao cấp 2", 3),
                ("MATE230320", "Vật liệu học cơ khí", 3),
                ("SOMT230420", "Sức bền vật liệu", 3),
                ("LLCT120205", "Kinh tế chính trị Mác - Lênin", 2),
                ("CADM230520", "Thiết kế CAD 2D/3D", 3),
                ("PHED110613", "Giáo dục thể chất 2 *", 1)
            ]),
            ("Năm học: {y1}-{y2}", "Học kỳ: HK01", [
                ("TOME230620", "Nguyên lý máy", 3),
                ("MADE230720", "Chi tiết máy", 3),
                ("MANU230820", "Công nghệ chế tạo máy 1", 3),
                ("LLCT120405", "Chủ nghĩa xã hội khoa học", 2),
                ("HYDR230920", "Thủy lực và khí nén", 3)
            ]),
            ("Năm học: {y1}-{y2}", "Học kỳ: HK02", [
                ("MANU331020", "Công nghệ chế tạo máy 2", 3),
                ("PRME321120", "Đồ án Chi tiết máy", 2),
                ("METO331220", "Dung sai và kỹ thuật đo", 3),
                ("CNCM331320", "Lập trình gia công CNC", 3),
                ("LLCT120314", "Tư tưởng Hồ Chí Minh", 2)
            ]),
            ("Năm học: {y2}-{y3}", "Học kỳ: HK01", [
                ("MOLD331420", "Thiết kế khuôn mẫu", 3),
                ("ROBO331520", "Robot công nghiệp", 3),
                ("PRCK321620", "Đồ án Công nghệ chế tạo", 2),
                ("PLC331720", "Điều khiển tự động PLC", 3),
                ("LLCT220514", "Lịch sử Đảng CSVN", 2)
            ]),
            ("Năm học: {y2}-{y3}", "Học kỳ: HK02", [
                ("CAM331820", "Hệ thống CAD/CAM/CAE", 3),
                ("MAIN331920", "Bảo trì và chẩn đoán thiết bị", 3),
                ("QUAL430120", "Quản lý chất lượng", 3),
                ("PRCK420220", "Đồ án Tự động hóa sản xuất", 2)
            ]),
            ("Năm học: {y3}-{y4}", "Học kỳ: HK01", [
                ("INTR430320", "Thực tập tốt nghiệp", 3),
                ("SPEC420420", "Chuyên đề Cơ khí", 3)
            ]),
            ("Năm học: {y3}-{y4}", "Học kỳ: HK02", [
                ("THES400120", "Khóa luận tốt nghiệp Cơ khí", 10)
            ])
        ]
    },
    "OTO": {
        "faculty": "Khoa Cơ khí Động lực",
        "program_name": "Công nghệ Kỹ thuật Ô tô",
        "code_prefix": "OTO",
        "semesters": [
            ("Năm học: {y0}-{y1}", "Học kỳ: HK01", [
                ("ENGL130137", "Anh văn 1", 3),
                ("MATH132401", "Toán cao cấp 1", 3),
                ("PHYS130902", "Vật lý 1", 3),
                ("DRWG130220", "Vẽ kỹ thuật ô tô", 3),
                ("MECH130120", "Cơ học lý thuyết", 3),
                ("LLCT130105", "Triết học Mác - Lênin", 3),
                ("PHED110513", "Giáo dục thể chất 1 *", 1)
            ]),
            ("Năm học: {y0}-{y1}", "Học kỳ: HK02", [
                ("ENGL230237", "Anh văn 2", 3),
                ("MATH132501", "Toán cao cấp 2", 3),
                ("SOMT230420", "Sức bền vật liệu", 3),
                ("THER230130", "Nhiệt động học và truyền nhiệt", 3),
                ("LLCT120205", "Kinh tế chính trị Mác - Lênin", 2),
                ("ELAU230230", "Điện - Điện tử ô tô căn bản", 3),
                ("PHED110613", "Giáo dục thể chất 2 *", 1)
            ]),
            ("Năm học: {y1}-{y2}", "Học kỳ: HK01", [
                ("ENGI230330", "Lý thuyết động cơ đốt trong", 3),
                ("AUCH230430", "Cấu tạo gầm ô tô", 3),
                ("MADE230720", "Chi tiết máy ô tô", 3),
                ("HYDR230920", "Thủy lực khí nén trên ô tô", 3),
                ("LLCT120405", "Chủ nghĩa xã hội khoa học", 2)
            ]),
            ("Năm học: {y1}-{y2}", "Học kỳ: HK02", [
                ("ELSY330530", "Hệ thống điện và điện tử ô tô", 3),
                ("PRAU320630", "Đồ án Động cơ đốt trong", 2),
                ("DYNA330730", "Lý thuyết ô tô", 3),
                ("DIAG330830", "Chẩn đoán kỹ thuật ô tô", 3),
                ("LLCT120314", "Tư tưởng Hồ Chí Minh", 2)
            ]),
            ("Năm học: {y2}-{y3}", "Học kỳ: HK01", [
                ("HYBR330930", "Ô tô Hybrid và Ô tô điện", 3),
                ("PRAU321030", "Đồ án Thiết kế ô tô", 2),
                ("TEST331130", "Kiểm định ô tô", 3),
                ("SEMI331230", "Hệ thống điều khiển tự động ô tô", 3),
                ("LLCT220514", "Lịch sử Đảng CSVN", 2)
            ]),
            ("Năm học: {y2}-{y3}", "Học kỳ: HK02", [
                ("BODY331330", "Thiết kế và khí động học ô tô", 3),
                ("SAFE331430", "An toàn chủ động và thụ động trên ô tô", 3),
                ("MGMT430130", "Quản lý dịch vụ kỹ thuật ô tô", 3)
            ]),
            ("Năm học: {y3}-{y4}", "Học kỳ: HK01", [
                ("INTR430230", "Thực tập tốt nghiệp Ô tô", 3),
                ("SPEC420330", "Chuyên đề Ô tô hiện đại", 3)
            ]),
            ("Năm học: {y3}-{y4}", "Học kỳ: HK02", [
                ("THES400130", "Khóa luận tốt nghiệp Ô tô", 10)
            ])
        ]
    },
    "QTKD": {
        "faculty": "Khoa Kinh tế",
        "program_name": "Quản trị kinh doanh",
        "code_prefix": "QTKD",
        "semesters": [
            ("Năm học: {y0}-{y1}", "Học kỳ: HK01", [
                ("ENGL130137", "Anh văn 1", 3),
                ("MATH130140", "Toán cao cấp cho kinh tế", 3),
                ("INMA130240", "Nhập môn Quản trị học", 3),
                ("MICR130340", "Kinh tế vi mô", 3),
                ("LLCT130105", "Triết học Mác - Lênin", 3),
                ("PHED110513", "Giáo dục thể chất 1 *", 1)
            ]),
            ("Năm học: {y0}-{y1}", "Học kỳ: HK02", [
                ("ENGL230237", "Anh văn 2", 3),
                ("MACR130440", "Kinh tế vĩ mô", 3),
                ("ACCT130540", "Nguyên lý kế toán", 3),
                ("STAT130640", "Thống kê kinh doanh", 3),
                ("LLCT120205", "Kinh tế chính trị Mác - Lênin", 2),
                ("PHED110613", "Giáo dục thể chất 2 *", 1)
            ]),
            ("Năm học: {y1}-{y2}", "Học kỳ: HK01", [
                ("MARK230740", "Marketing căn bản", 3),
                ("FINA230840", "Tài chính doanh nghiệp", 3),
                ("ORGB230940", "Hành vi tổ chức", 3),
                ("BULW231040", "Luật kinh doanh", 3),
                ("LLCT120405", "Chủ nghĩa xã hội khoa học", 2)
            ]),
            ("Năm học: {y1}-{y2}", "Học kỳ: HK02", [
                ("HRMA331140", "Quản trị nguồn nhân lực", 3),
                ("OPMA331240", "Quản trị vận hành sản xuất", 3),
                ("ECOM331340", "Thương mại điện tử", 3),
                ("MISY331440", "Hệ thống thông tin quản lý", 3),
                ("LLCT120314", "Tư tưởng Hồ Chí Minh", 2)
            ]),
            ("Năm học: {y2}-{y3}", "Học kỳ: HK01", [
                ("STMA331540", "Quản trị chiến lược", 3),
                ("PRBA321640", "Đồ án Quản trị kinh doanh", 2),
                ("LOGI331740", "Quản trị chuỗi cung ứng Logistics", 3),
                ("INBU331840", "Kinh doanh quốc tế", 3),
                ("LLCT220514", "Lịch sử Đảng CSVN", 2)
            ]),
            ("Năm học: {y2}-{y3}", "Học kỳ: HK02", [
                ("ENTR331940", "Khởi sự doanh nghiệp", 3),
                ("CRMA332040", "Quản trị quan hệ khách hàng CRM", 3),
                ("DIGM332140", "Digital Marketing", 3)
            ]),
            ("Năm học: {y3}-{y4}", "Học kỳ: HK01", [
                ("INTR430140", "Thực tập tốt nghiệp Quản trị", 3),
                ("SEMI420240", "Chuyên đề Quản trị doanh nghiệp", 3)
            ]),
            ("Năm học: {y3}-{y4}", "Học kỳ: HK02", [
                ("THES400140", "Khóa luận tốt nghiệp Quản trị", 10)
            ])
        ]
    }
}
