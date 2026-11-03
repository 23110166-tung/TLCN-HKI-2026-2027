# -*- coding: utf-8 -*-
"""
HCMUTE DocAI – Ứng dụng Web Nhận dạng, Trích xuất & Thẩm định Đơn học vụ & Bảng điểm Sinh viên
Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh (HCMUTE)
Tiểu luận chuyên ngành AI - HKI 2026-2027

GVHD: TS. Bùi Mạnh Quân
Sinh viên thực hiện:
  - Nguyễn Bách Tùng (MSSV: 23110166) - Nhóm trưởng
  - Nguyễn Hưng Nguyên (MSSV: 23110035)
  - Trần Cẩm Long (MSSV: 23110122)
"""

import os
import sys
import json
import glob
import importlib
import pandas as pd
from PIL import Image
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pipeline.inference_engine
importlib.reload(pipeline.inference_engine)

from pipeline.inference_engine import (
    DocumentLayoutAndOCR,
    DOMAIN_ADAPTED_V2_YOLO_WEIGHTS,
    DOMAIN_ADAPTED_YOLO_WEIGHTS,
    SYNTHETIC_YOLO_WEIGHTS,
    DEFAULT_YOLO_WEIGHTS,
    DEFAULT_OCR_WEIGHTS,
    STATUS_COLORS
)
from pipeline.rule_engine import AcademicRuleEngine, FORM_REQUIREMENTS, SCHOOL_NAME
from pipeline.ontology import CLASS_NAMES, CLASS_MAP, CLASS_COLORS

# =====================================================================
# 1. CẤU HÌNH TRANG STREAMLIT & GIAO DIỆN HỌC THUẬT HCMUTE
# =====================================================================
st.set_page_config(
    page_title="HCMUTE DocAI – Thẩm định Đơn học vụ & Bảng điểm",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho phong cách hiện đại, học thuật với gam màu xanh Navy HCMUTE (#1B365D)
st.markdown("""
<style>
    /* Header & Branding */
    .hcmute-header {
        background: linear-gradient(135deg, #1B365D 0%, #2E4A7D 70%, #0072CE 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .hcmute-header h1 {
        color: #FFFFFF !important;
        font-size: 26px !important;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.3px;
    }
    .hcmute-header h3 {
        color: #E2EDF8 !important;
        font-size: 15px !important;
        font-weight: 500;
        margin: 0 0 10px 0;
    }
    .hcmute-header .badge-container {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .hcmute-badge {
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        color: #FFFFFF;
        font-weight: 500;
    }

    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #E5E9F0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        text-align: center;
    }
    .metric-card .title {
        font-size: 13px;
        color: #6C7A89;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-card .value {
        font-size: 26px;
        font-weight: 700;
        color: #1B365D;
    }

    /* Status Badges */
    .status-valid {
        background-color: #E6F7E9;
        color: #1E7E34;
        border: 1px solid #28A745;
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
    }
    .status-warning {
        background-color: #FFF8E6;
        color: #B7791F;
        border: 1px solid #E2A03F;
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
    }
    .status-invalid {
        background-color: #FDE8E8;
        color: #C53030;
        border: 1px solid #DC3545;
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
    }

    /* Entity Data Card */
    .entity-box {
        background-color: #F8FAFC;
        border-left: 4px solid #2E4A7D;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Banner tiêu đề trường & thông tin đề tài
st.markdown(f"""
<div class="hcmute-header">
    <h1>🎓 {SCHOOL_NAME}</h1>
    <h3>HỆ THỐNG NHẬN DẠNG, TRÍCH XUẤT VÀ KIỂM TRA TÍNH HỢP LỆ CỦA THÔNG TIN TỪ ĐƠN HỌC VỤ & BẢNG ĐIỂM BẰNG HỌC SÂU</h3>
    <div class="badge-container">
        <span class="hcmute-badge">🏛️ Khoa CNTT - Bộ môn Trí tuệ Nhân tạo</span>
        <span class="hcmute-badge">👨‍🏫 GVHD: TS. Bùi Mạnh Quân</span>
        <span class="hcmute-badge">👥 Nhóm SV: Nguyễn Bách Tùng (23110166), Nguyễn Hưng Nguyên, Trần Cẩm Long</span>
        <span class="hcmute-badge">🚀 YOLOv11 + VietOCR + Academic Rule Engine</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# 2. KHỞI TẠO VÀ CACHE MÔ HÌNH HỌC SÂU (YOLOV11 + VIETOCR)
# =====================================================================
@st.cache_resource(show_spinner=False)
def get_inference_engine(yolo_weight_path, conf_threshold, device, engine_version="v2.3_kb_grounding_tilt_signatures"):
    """Nạp mô hình 1 lần duy nhất vào bộ nhớ RAM/VRAM để tăng tốc suy luận."""
    engine = DocumentLayoutAndOCR(
        yolo_model_path=yolo_weight_path,
        ocr_model_name="vgg_transformer",
        ocr_weights_path=DEFAULT_OCR_WEIGHTS,
        device=device,
        conf_threshold=conf_threshold,
        imgsz=1280
    )
    return engine


# =====================================================================
# 3. SIDEBAR CẤU HÌNH & CHỌN NGUỒN DỮ LIỆU ĐẦU VÀO
# =====================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")

    if st.button("🔄 Tải lại mô hình & Xóa bộ nhớ đệm (Clear Cache)", use_container_width=True):
        st.cache_resource.clear()
        if "inference_result" in st.session_state:
            del st.session_state["inference_result"]
        if "last_processed_file" in st.session_state:
            del st.session_state["last_processed_file"]
        st.rerun()

    # 1. Chọn Trọng số YOLO
    yolo_options = {}
    if os.path.exists(DOMAIN_ADAPTED_V2_YOLO_WEIGHTS):
        yolo_options["YOLOv11 Domain-Adapted v2 (Kéo mAP lên 34.72% - Khuyên dùng)"] = DOMAIN_ADAPTED_V2_YOLO_WEIGHTS
    if os.path.exists(DOMAIN_ADAPTED_YOLO_WEIGHTS) and DOMAIN_ADAPTED_YOLO_WEIGHTS != DOMAIN_ADAPTED_V2_YOLO_WEIGHTS:
        yolo_options["YOLOv11 Domain-Adapted v1"] = DOMAIN_ADAPTED_YOLO_WEIGHTS
    if os.path.exists(SYNTHETIC_YOLO_WEIGHTS):
        yolo_options["YOLOv11 Synthetic Baseline (Stage 2 - 98.6% mAP)"] = SYNTHETIC_YOLO_WEIGHTS

    if not yolo_options:
        yolo_options["Mặc định (yolo11n.pt)"] = "yolo11n.pt"

    selected_yolo_name = st.selectbox(
        "Mô hình YOLOv11 Layout Detection:",
        options=list(yolo_options.keys()),
        index=0
    )
    chosen_yolo_path = yolo_options[selected_yolo_name]

    # 2. Ngưỡng tin cậy (Confidence)
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        conf_thresh = st.slider("Ngưỡng IoU/Conf:", min_value=0.15, max_value=0.85, value=0.35, step=0.05)
    with col_c2:
        device_choice = st.selectbox("Thiết bị:", ["cpu", "cuda:0"], index=0)

    # 3. Loại biểu mẫu học vụ (Hoặc Tự suy luận)
    form_type_options = ["Tự động suy luận từ nội dung (Auto-detect)"] + list(FORM_REQUIREMENTS.keys())
    selected_form_raw = st.selectbox("Loại biểu mẫu học vụ:", options=form_type_options, index=0)
    selected_form_code = None if selected_form_raw.startswith("Tự động") else selected_form_raw

    st.markdown("---")
    st.header("📂 Nguồn Dữ liệu Kiểm thử")

    input_mode = st.radio(
        "Phương thức cấp dữ liệu:",
        ["Chọn mẫu kiểm thử thực tế có sẵn", "Tải ảnh từ máy tính (Upload)"],
        index=0
    )

    sample_image_path = None
    uploaded_image_file = None

    if input_mode == "Chọn mẫu kiểm thử thực tế có sẵn":
        sample_category = st.selectbox(
            "Nhóm mẫu biểu mẫu:",
            ["Ảnh chụp Camera thực tế (Smartphone - 120 ảnh test)", "Ảnh quét Scan sạch (Synthetic Tests)"]
        )

        if "Camera" in sample_category:
            test_camera_dir = os.path.join(BASE_DIR, "dataset_yolo_domain_adaptation", "images", "test")
            camera_files = sorted(glob.glob(os.path.join(test_camera_dir, "*.jpg")))
            if camera_files:
                sample_map = {os.path.basename(f): f for f in camera_files}
                chosen_sample_name = st.selectbox("Chọn ảnh chụp smartphone:", list(sample_map.keys()), index=0)
                sample_image_path = sample_map[chosen_sample_name]
            else:
                st.warning("Chưa tìm thấy tập ảnh test camera.")
        else:
            test_syn_dir = os.path.join(BASE_DIR, "dataset_yolo", "images", "test")
            syn_files = sorted(glob.glob(os.path.join(test_syn_dir, "*.*")))[:15]
            if syn_files:
                sample_map = {os.path.basename(f): f for f in syn_files}
                chosen_sample_name = st.selectbox("Chọn bản scan sạch:", list(sample_map.keys()), index=0)
                sample_image_path = sample_map[chosen_sample_name]
            else:
                st.warning("Chưa tìm thấy tập ảnh test synthetic.")
    else:
        uploaded_image_file = st.file_uploader(
            "Tải lên ảnh tài liệu học vụ (PNG, JPG, JPEG):",
            type=["png", "jpg", "jpeg"]
        )

    st.markdown("---")
    st.info("💡 **Gợi ý kiểm thử:** Hãy chọn các mẫu ảnh camera có góc nghiêng hoặc bóng đổ để quan sát năng lực thích nghi miền (Domain Adaptation) của YOLOv11.")


# =====================================================================
# 4. TIẾN TRÌNH XỬ LÝ SUY LUẬN (INFERENCE EXECUTION)
# =====================================================================

# Xác định ảnh đầu vào
target_image_pil = None
target_file_label = ""

if input_mode == "Chọn mẫu kiểm thử thực tế có sẵn" and sample_image_path and os.path.exists(sample_image_path):
    target_image_pil = Image.open(sample_image_path).convert("RGB")
    target_file_label = os.path.basename(sample_image_path)
elif input_mode == "Tải ảnh từ máy tính (Upload)" and uploaded_image_file is not None:
    target_image_pil = Image.open(uploaded_image_file).convert("RGB")
    target_file_label = uploaded_image_file.name

if target_image_pil is None:
    st.warning("👉 Vui lòng chọn một mẫu tài liệu có sẵn từ thanh bên trái hoặc tải lên ảnh tài liệu của bạn để bắt đầu phân tích.")
    st.stop()

# Tải mô hình
with st.spinner("⏳ Đang khởi tạo mô hình YOLOv11 Layout và VietOCR fine-tuned..."):
    engine = get_inference_engine(chosen_yolo_path, conf_thresh, device_choice)

# Nút thực hiện nhận dạng & thẩm định
st.markdown("### 📄 Tài liệu đang chọn: `" + target_file_label + "`")
process_col1, process_col2 = st.columns([1, 4])
with process_col1:
    btn_run = st.button("🚀 BẮT ĐẦU PHÂN TÍCH & THẨM ĐỊNH", type="primary", use_container_width=True)

# Session state lưu kết quả phân tích
if "last_processed_file" not in st.session_state or st.session_state["last_processed_file"] != target_file_label or btn_run:
    with st.spinner("🔍 Đang phát hiện vùng bố cục YOLOv11, trích xuất chữ VietOCR và thẩm định quy chế học vụ..."):
        res = engine.process_document(target_image_pil, form_type=selected_form_code, validate=True)
        res["document_file"] = target_file_label
        st.session_state["inference_result"] = res
        st.session_state["last_processed_file"] = target_file_label

result = st.session_state.get("inference_result")

if not result:
    st.stop()

# Lấy các thành phần kết quả
validation = result.get("validation", {})
entities = result.get("entities", {})
courses = result.get("courses_table", [])
raw_detections = result.get("raw_detections", [])
verification = result.get("verification", {})
status = validation.get("status", "WARNING")
form_type = validation.get("form_type", "UNKNOWN")
form_name = validation.get("form_type_name", validation.get("form_name", "Không xác định"))
compliance_score = validation.get("compliance_score", 0.0)
summary = validation.get("summary", {})


# =====================================================================
# 5. KHỐI THẺ TỔNG QUAN (METRIC HEADERS)
# =====================================================================
st.markdown("---")
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

with m_col1:
    status_html = ""
    if status == "VALID":
        status_html = '<span class="status-valid">🟢 HỢP LỆ (VALID)</span>'
    elif status == "WARNING":
        status_html = '<span class="status-warning">🟡 CẢNH BÁO (WARNING)</span>'
    else:
        status_html = '<span class="status-invalid">🔴 VI PHẠM (INVALID)</span>'
    st.markdown(f'<div class="metric-card"><div class="title">Trạng thái Hồ sơ</div><div>{status_html}</div></div>', unsafe_allow_html=True)

with m_col2:
    st.markdown(f'<div class="metric-card"><div class="title">Điểm Tuân thủ Quy chế</div><div class="value" style="color: {"#1E7E34" if compliance_score>=75 else "#C53030"};">{compliance_score}%</div></div>', unsafe_allow_html=True)

with m_col3:
    st.markdown(f'<div class="metric-card"><div class="title">Vùng Layout phát hiện</div><div class="value">{len(raw_detections)}</div></div>', unsafe_allow_html=True)

with m_col4:
    sig_count = verification.get("signature_count", 0)
    stamp_count = verification.get("stamp_count", 0)
    st.markdown(f'<div class="metric-card"><div class="title">Chữ ký & Mộc đỏ</div><div class="value">{sig_count} ký | {stamp_count} mộc</div></div>', unsafe_allow_html=True)

with m_col5:
    st.markdown(f'<div class="metric-card"><div class="title">Dòng Bảng điểm trích xuất</div><div class="value">{len(courses)} dòng</div></div>', unsafe_allow_html=True)


# =====================================================================
# 6. GIAO DIỆN 4 TABS CHI TIẾT
# =====================================================================
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs([
    "👁️ Trực quan hóa Layout (YOLOv11)",
    "📋 Dữ liệu Cấu trúc & OCR (VietOCR)",
    "⚖️ Thẩm định Quy chế (Academic Rule Engine)",
    "🔬 Benchmark Nghiên cứu (RQ1 & RQ2 Hub)"
])


# ---------------------------------------------------------------------
# TAB 1: TRỰC QUAN HÓA LAYOUT (YOLOV11)
# ---------------------------------------------------------------------
with tab1:
    st.subheader("🖼️ Trực quan hóa Phân tích Bố cục Tài liệu")
    
    t1_c1, t1_c2, t1_c3 = st.columns([1, 1, 2])
    with t1_c1:
        toggle_labels = st.checkbox("Hiển thị nhãn lớp & điểm tin cậy", value=True)
    with t1_c2:
        toggle_banner = st.checkbox("Hiển thị thanh trạng thái thẩm định", value=True)
    with t1_c3:
        st.caption(f"Loại biểu mẫu nhận diện: **{form_name} ({form_type})** | Nguồn: {validation.get('form_type_source', 'N/A')}")

    # Tạo ảnh trực quan hóa
    annotated_img = engine.create_visualization(target_image_pil, result, show_labels=toggle_labels, show_banner=toggle_banner)

    col_view1, col_view2 = st.columns(2)
    with col_view1:
        st.markdown("**Ảnh tài liệu gốc (Input Document):**")
        st.image(target_image_pil, use_container_width=True)
    with col_view2:
        st.markdown("**Ảnh dự đoán phân vùng Bố cục (YOLOv11 Layout Detections):**")
        st.image(annotated_img, use_container_width=True)

    # Chi tiết các Bounding Boxes
    with st.expander(f"🔍 Danh sách chi tiết {len(raw_detections)} vùng Bounding Box phát hiện được:"):
        det_data = []
        for idx, d in enumerate(raw_detections, 1):
            det_data.append({
                "STT": idx,
                "Lớp đối tượng": d.get("class_name"),
                "Tọa độ [x0, y0, x1, y1]": str(d.get("box")),
                "Độ tin cậy YOLO": f"{d.get('confidence', 0.0):.3f}",
                "Văn bản trích xuất (VietOCR)": d.get("text", "")
            })
        st.dataframe(pd.DataFrame(det_data), use_container_width=True)


# ---------------------------------------------------------------------
# TAB 2: DỮ LIỆU CẤU TRÚC & OCR (VIETOCR)
# ---------------------------------------------------------------------
with tab2:
    st.subheader("📋 Kết quả Trích xuất Dữ liệu Cấu trúc & OCR Tiếng Việt")

    col_e1, col_e2 = st.columns([1, 1])

    with col_e1:
        st.markdown("#### 👤 Thông tin Định danh Sinh viên (Identity Fields)")
        field_labels = {
            "student_name": "Họ và tên sinh viên",
            "student_id": "Mã số sinh viên (MSSV)",
            "class_name": "Lớp sinh hoạt",
            "faculty": "Khoa đào tạo",
            "major": "Chuyên ngành",
            "dob": "Ngày tháng năm sinh",
            "pob": "Nơi sinh",
            "phone": "Số điện thoại liên hệ",
            "email": "Email sinh viên / trường",
            "date": "Ngày lập đơn",
            "reason": "Lý do làm đơn",
            "degree_info": "Bậc đào tạo / Khóa học"
        }

        has_any_entity = False
        for k, label in field_labels.items():
            if k in entities:
                has_any_entity = True
                val = entities[k]
                if isinstance(val, list):
                    val_str = ", ".join(str(v) for v in val)
                else:
                    val_str = str(val)
                st.markdown(f"""
                <div class="entity-box">
                    <span style="font-size: 12px; color: #6C7A89; font-weight: 600;">{label}</span><br>
                    <strong style="font-size: 15px; color: #1B365D;">{val_str}</strong>
                </div>
                """, unsafe_allow_html=True)

        if not has_any_entity:
            st.info("Chưa phát hiện trường định danh rõ ràng từ ảnh này.")

    with col_e2:
        st.markdown("#### 🔏 Tính Toàn vẹn Hồ sơ (Pháp lý & Chữ ký)")
        st.markdown(f"""
        - **Mộc dấu đỏ (Stamp):** {"✅ Đã phát hiện (" + str(verification.get("stamp_count", 0)) + " mộc)" if verification.get("has_stamp") else "❌ Không có con dấu"}
        - **Chữ ký (Signature):** {"✅ Đã phát hiện (" + str(verification.get("signature_count", 0)) + " chữ ký)" if verification.get("has_signature") else "❌ Không có chữ ký"}
        - **Yêu cầu loại mẫu:** {validation.get("form_spec", {}).get("signers", "Theo quy chế chung")}
        """)

        st.markdown("#### 💾 Xuất Dữ liệu Cấu trúc")
        json_str = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        st.download_button(
            label="📥 Tải xuống Dữ liệu JSON (Full Extraction)",
            data=json_str,
            file_name=f"{os.path.splitext(target_file_label)[0]}_extracted.json",
            mime="application/json",
            use_container_width=True
        )

    # Bảng điểm học phần
    st.markdown("---")
    st.markdown("#### 📊 Bảng Điểm & Học Phần Trích Xuất (Gom Cụm Dòng Trục Y)")
    if courses:
        course_display_list = []
        for idx, c in enumerate(courses, 1):
            course_display_list.append({
                "STT": idx,
                "Mã học phần": c.get("course_code", "N/A"),
                "Tên học phần": c.get("course_name", "N/A"),
                "Số tín chỉ": c.get("credits", "N/A"),
                "Điểm trích xuất (Hệ 10 | 4 | Chữ)": c.get("score", "N/A"),
                "Độ tin cậy OCR min": f"{c.get('min_ocr_confidence', 0.0):.3f}" if c.get("min_ocr_confidence") else "N/A"
            })
        df_courses = pd.DataFrame(course_display_list)
        st.dataframe(df_courses, use_container_width=True)

        csv_courses = df_courses.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 Tải Bảng Điểm CSV",
            data=csv_courses,
            file_name=f"{os.path.splitext(target_file_label)[0]}_courses.csv",
            mime="text/csv"
        )
    else:
        st.info("ℹ️ Biểu mẫu này không chứa bảng điểm học phần hoặc không nhận diện được dòng học phần.")


# ---------------------------------------------------------------------
# TAB 3: BÁO CÁO THẨM ĐỊNH QUY CHẾ (RULE ENGINE)
# ---------------------------------------------------------------------
with tab3:
    st.subheader("⚖️ Báo cáo Thẩm định Quy chế Nghiệp vụ Học vụ")
    
    st.markdown(f"**Kết luận thẩm định:** {validation.get('verdict_message', 'N/A')}")
    st.progress(float(compliance_score) / 100.0)

    # Thống kê tổng hợp số luật
    rule_stat_col1, rule_stat_col2, rule_stat_col3, rule_stat_col4 = st.columns(4)
    with rule_stat_col1:
        st.metric("Tổng số luật kiểm tra", summary.get("total_rules", 0))
    with rule_stat_col2:
        st.metric("Luật Thỏa mãn (Passed)", summary.get("passed", 0))
    with rule_stat_col3:
        st.metric("Vi phạm (Errors)", summary.get("errors", 0))
    with rule_stat_col4:
        st.metric("Cảnh báo (Warnings)", summary.get("warnings", 0))

    # Bảng chi tiết toàn bộ các luật
    details = validation.get("details", [])
    if details:
        st.markdown("#### 📋 Chi tiết kết quả kiểm tra từng luật quy chế:")
        
        filter_status = st.selectbox("Lọc kết quả kiểm tra:", ["Tất cả các luật", "Chỉ hiển thị Lỗi (ERROR)", "Chỉ hiển thị Cảnh báo (WARNING)", "Chỉ hiển thị Đạt (PASSED)"])

        rule_rows = []
        for r in details:
            passed = r.get("passed", False)
            sev = r.get("severity", "INFO")

            if filter_status == "Chỉ hiển thị Lỗi (ERROR)" and (passed or sev != "ERROR"):
                continue
            if filter_status == "Chỉ hiển thị Cảnh báo (WARNING)" and (passed or sev != "WARNING"):
                continue
            if filter_status == "Chỉ hiển thị Đạt (PASSED)" and not passed:
                continue

            rule_rows.append({
                "Mã luật": r.get("rule_id"),
                "Tên quy tắc": r.get("name"),
                "Mức độ": "❌ ERROR" if sev == "ERROR" and not passed else ("⚠️ WARNING" if sev == "WARNING" and not passed else "ℹ️ INFO"),
                "Trạng thái": "✅ ĐẠT" if passed else "❌ KHÔNG ĐẠT",
                "Thông điệp kiểm tra / Lý do": r.get("message"),
                "Trường liên quan": r.get("field", ""),
                "Giá trị thực tế": str(r.get("value", ""))
            })

        if rule_rows:
            st.dataframe(pd.DataFrame(rule_rows), use_container_width=True)
        else:
            st.success("Không có vi phạm nào trong bộ lọc đã chọn!")


# ---------------------------------------------------------------------
# TAB 4: BENCHMARK NGHIÊN CỨU & ĐỐI CHỨNG (RQ1 & RQ2)
# ---------------------------------------------------------------------
with tab4:
    st.subheader("🔬 Tổng hợp Kết quả Thực nghiệm Khoa học Đề tài")
    st.markdown("""
    Trang này tổng hợp các kết quả thực nghiệm định lượng nhằm trả lời 2 Câu hỏi Nghiên cứu (Research Questions) 
    được đặt ra trong Đề cương Tiểu luận chuyên ngành AI (HKI 2026-2027).
    """)

    # Khối 1: RQ1 - VietOCR
    st.markdown("### 1️⃣ Câu hỏi Nghiên cứu 1 (RQ1 - Fine-tuning VietOCR)")
    st.markdown("*Phương pháp Fine-tuning chuyên biệt mô hình Seq2Seq (VGG-Transformer) trên biểu mẫu học vụ tiếng Việt.*")

    col_rq1_1, col_rq1_2 = st.columns([1, 1])
    with col_rq1_1:
        rq1_csv_path = os.path.join(BASE_DIR, "HCMUTE_VietOCR_Finetuned_Artifacts", "vietocr_research_question_1_results.csv")
        if os.path.exists(rq1_csv_path):
            df_rq1 = pd.read_csv(rq1_csv_path)
            st.markdown("**Bảng số liệu đối chứng độc lập trên 867 ảnh crop văn bản:**")
            st.dataframe(df_rq1, use_container_width=True)
            st.success("🎯 **Kết luận RQ1:** Tỷ lệ Exact Match (EM) tăng vọt từ **51.67% lên 69.78% (+18.11%)**, lỗi ký tự CER giảm rõ rệt, giải quyết hiện tượng nhận diện nhầm dấu thanh tiếng Việt.")
    with col_rq1_2:
        rq1_chart_path = os.path.join(BASE_DIR, "HCMUTE_VietOCR_Finetuned_Artifacts", "vietocr_comparison_chart.png")
        if os.path.exists(rq1_chart_path):
            st.image(rq1_chart_path, caption="Biểu đồ đối chứng hiệu năng VietOCR Baseline vs Fine-tuned", use_container_width=True)

    st.markdown("---")

    # Khối 2: RQ2 - YOLOv11 Domain Adaptation
    st.markdown("### 2️⃣ Câu hỏi Nghiên cứu 2 (RQ2 - YOLOv11 Domain Adaptation)")
    st.markdown("*Khắc phục hiện tượng suy giảm hiệu năng miền (Domain Shift) trên 120 ảnh chụp smartphone thực tế.*")

    col_rq2_1, col_rq2_2 = st.columns([1, 1])
    with col_rq2_1:
        rq2_csv_path = os.path.join(BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2", "domain_adaptation_rq2_results.csv")
        if os.path.exists(rq2_csv_path):
            df_rq2 = pd.read_csv(rq2_csv_path)
            st.markdown("**Bảng số liệu đánh giá Zero-shot vs Domain-Adapted (Zero Data Leakage Test):**")
            st.dataframe(df_rq2, use_container_width=True)
            st.success("🎯 **Kết luận RQ2:** mAP@0.5 trên ảnh camera thực tế tăng từ **6.89% lên 34.72% (gấp 5.04 lần)**, Recall tăng từ **14.89% lên 34.11% (hơn gấp đôi)**.")
    with col_rq2_2:
        rq2_chart_path = os.path.join(BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2", "domain_adaptation_comparison_chart.png")
        if os.path.exists(rq2_chart_path):
            st.image(rq2_chart_path, caption="Biểu đồ so sánh các chỉ số hiệu năng trước và sau Thích nghi miền", use_container_width=True)

    # Chi tiết 18 lớp học thuật
    per_class_csv_path = os.path.join(BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2", "domain_adaptation_per_class.csv")
    if os.path.exists(per_class_csv_path):
        st.markdown("#### 📋 Hiệu năng Chi tiết trên 18 Lớp Đối tượng Biểu mẫu:")
        df_per_class = pd.read_csv(per_class_csv_path)
        st.dataframe(df_per_class, use_container_width=True)

    # Đường cong huấn luyện Loss/mAP
    train_curve_path = os.path.join(BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2", "training_results_curves.png")
    if os.path.exists(train_curve_path):
        st.markdown("#### 📈 Đồ thị Đường cong Huấn luyện & Đánh giá (50 Epochs):")
        st.image(train_curve_path, caption="Hàm mất mát và đường cong mAP qua 50 epochs huấn luyện trên Kaggle GPU", use_container_width=True)


# =====================================================================
# 7. FOOTER CHUẨN HỌC THUẬT
# =====================================================================
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6C7A89; font-size: 13px; padding: 10px 0;">
    <strong>HCMUTE DocAI</strong> – Đề tài Tiểu luận chuyên ngành AI (HKI 2026-2027) | {SCHOOL_NAME}<br>
    GVHD: TS. Bùi Mạnh Quân | Nhóm tác giả: Nguyễn Bách Tùng (23110166) - Nguyễn Hưng Nguyên - Trần Cẩm Long
</div>
""", unsafe_allow_html=True)

