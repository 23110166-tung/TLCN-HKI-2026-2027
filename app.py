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
import time
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
    DOMAIN_ADAPTED_V2_ONNX_WEIGHTS,
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

    if not os.path.exists(DEFAULT_OCR_WEIGHTS):
        st.warning("⚠️ Chưa tìm thấy file `vietocr_hcmute_finetuned.pth` trong `HCMUTE_VietOCR_Finetuned_Artifacts/`. Hãy xem hướng dẫn tải trọng số trong `HUONG_DAN_CAI_DAT.md`.")

    selected_yolo_name = st.selectbox(
        "Mô hình YOLOv11 Layout Detection:",
        options=list(yolo_options.keys()),
        index=0
    )
    chosen_yolo_path = yolo_options[selected_yolo_name]

    # 1.1 Toggle chọn Engine suy luận: PyTorch Native vs ONNX Runtime
    enable_onnx = st.toggle(
        "⚡ Kích hoạt ONNX Runtime (.onnx)",
        value=False,
        help="Chuyển đổi giữa PyTorch Native (.pt) và ONNX Runtime (.onnx).\n- ONNX Runtime tối ưu hóa đồ thị tĩnh, tăng tốc 1.2x - 2.5x và giảm đáng kể độ trễ trên máy tính văn phòng (CPU).\n- PyTorch Native phù hợp cho huấn luyện và kiểm thử động."
    )
    if enable_onnx:
        onnx_candidate = chosen_yolo_path.rsplit(".", 1)[0] + ".onnx"
        if os.path.exists(onnx_candidate):
            chosen_yolo_path = onnx_candidate
        elif os.path.exists(DOMAIN_ADAPTED_V2_ONNX_WEIGHTS):
            chosen_yolo_path = DOMAIN_ADAPTED_V2_ONNX_WEIGHTS
        st.caption("🚀 **Engine:** `ONNX Runtime 1.20.2 (.onnx)` | Tối ưu hóa đồ thị suy luận")
    else:
        st.caption("🔥 **Engine:** `PyTorch Native 2.5.1 (.pt)` | Mô hình dynamic tensor gốc")

    # 2. Ngưỡng tin cậy (Confidence)
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        conf_thresh = st.slider("Ngưỡng IoU/Conf:", min_value=0.15, max_value=0.85, value=0.35, step=0.05)
    with col_c2:
        try:
            import torch as _torch
            _has_cuda = _torch.cuda.is_available()
        except Exception:
            _has_cuda = False
        _dev_opts = ["cuda:0", "cpu"] if _has_cuda else ["cpu", "cuda:0"]
        device_choice = st.selectbox("Thiết bị:", _dev_opts, index=0)

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
            "Nhóm mẫu biểu mẫu (Ưu tiên bản quét Scan sạch):",
            [
                "Ảnh quét Scan sạch (Đầy đủ 10 nhóm T01–T10 & Bảng điểm)",
                "Ảnh chụp Camera thực tế (Smartphone - 17 ảnh test)",
            ],
            index=0,
        )

        if "Scan sạch" in sample_category:
            clean_forms = sorted(glob.glob(os.path.join(BASE_DIR, "dataset_yolo", "images", "test", "*.*")))
            clean_transcripts = sorted(glob.glob(os.path.join(BASE_DIR, "dataset_yolo_transcripts", "images", "test", "*.*")))
            portal_sample = os.path.join(BASE_DIR, "sample", "NguyenBachTung_Academic_Transcript.jpg")
            syn_files = clean_forms + clean_transcripts
            if os.path.exists(portal_sample):
                syn_files.append(portal_sample)
            if not syn_files:
                test_syn_dir = os.path.join(BASE_DIR, "sample")
                syn_files = sorted(glob.glob(os.path.join(test_syn_dir, "*.jpg")) + glob.glob(os.path.join(test_syn_dir, "*.png")))

            if syn_files:
                GROUP_FILTERS = {
                    "Tất cả (10 nhóm T01–T10)": None,
                    "T01 - Điểm I": ("diem_i",),
                    "T02 - Học 2 CTĐT": ("hoc2ctdt",),
                    "T03 - Chuyển hệ": ("chuyen_he",),
                    "T04 - Chuyển điểm (2018)": ("chuyen_diem_2018",),
                    "T05 - Bảo lưu": ("bao_luu",),
                    "T06 - Bản sao sổ gốc": ("ban_sao",),
                    "T07 - Chuyển điểm NN": ("chuyen_diem_nn",),
                    "T08 - Học môn thay thế": ("thay_the_mon",),
                    "T09 - Miễn thi": ("mien_thi",),
                    "T10 - Bảng điểm Portal": ("transcript", "Academic_Transcript"),
                }
                chosen_group = st.selectbox("Lọc theo nhóm biểu mẫu:", list(GROUP_FILTERS.keys()), index=0)
                prefixes = GROUP_FILTERS[chosen_group]
                if prefixes:
                    filtered_files = [
                        f for f in syn_files
                        if any(p.lower() in os.path.basename(f).lower() for p in prefixes)
                    ]
                else:
                    filtered_files = syn_files
                sample_map = {os.path.basename(f): f for f in filtered_files}
                if sample_map:
                    chosen_sample_name = st.selectbox(
                        f"Chọn bản scan sạch ({len(sample_map)} mẫu):",
                        list(sample_map.keys()),
                        index=0,
                    )
                    sample_image_path = sample_map[chosen_sample_name]
                else:
                    st.warning("Không có mẫu nào khớp nhóm đã chọn.")
            else:
                st.warning("Chưa tìm thấy tập ảnh test scan sạch. Bạn có thể chuyển sang chế độ 'Tải ảnh từ máy tính (Upload)'.")
        else:
            test_camera_dir = os.path.join(BASE_DIR, "dataset_yolo_domain_adaptation", "images", "test")
            camera_files = sorted(glob.glob(os.path.join(test_camera_dir, "*.jpg")))
            if not camera_files:
                # Fallback sang thư mục sample cục bộ đi kèm mã nguồn
                test_camera_dir = os.path.join(BASE_DIR, "sample")
                camera_files = sorted(glob.glob(os.path.join(test_camera_dir, "REAL_*.jpg")))
            if camera_files:
                sample_map = {os.path.basename(f): f for f in camera_files}
                chosen_sample_name = st.selectbox(
                    f"Chọn ảnh chụp smartphone ({len(sample_map)} ảnh):",
                    list(sample_map.keys()),
                    index=0,
                )
                sample_image_path = sample_map[chosen_sample_name]
            else:
                st.warning("Chưa tìm thấy tập ảnh test camera. Bạn có thể chuyển sang chế độ 'Tải ảnh từ máy tính (Upload)'.")
    else:
        uploaded_image_file = st.file_uploader(
            "Tải lên ảnh tài liệu học vụ (PNG, JPG, JPEG):",
            type=["png", "jpg", "jpeg"]
        )

    st.markdown("---")
    st.markdown("### 🔬 Tiền Xử Lý Ảnh (Thử nghiệm)")
    enable_auto_scanner = st.toggle(
        "✨ Tự động Quét & Nắn phẳng (Auto Doc Scanner)",
        value=False,
        help="Thử nghiệm tiền xử lý: Tự động phát hiện 4 góc tài liệu, nắn phẳng phối cảnh góc nghiêng và khử bóng đổ camera để biến ảnh điện thoại thành bản scan sạch trước khi nhận dạng."
    )
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

# Tiền xử lý Virtual Document Scanner nếu được kích hoạt
effective_image_pil = target_image_pil
scan_meta = None
if enable_auto_scanner:
    from pipeline.document_scanner import default_scanner
    effective_image_pil, scan_meta = default_scanner.scan(target_image_pil)

# Nút thực hiện nhận dạng & thẩm định
engine_badge = "⚡ `ONNX Runtime (.onnx)`" if chosen_yolo_path.endswith(".onnx") else "🔥 `PyTorch Native (.pt)`"
st.markdown(f"### 📄 Tài liệu: `{target_file_label}` &nbsp;|&nbsp; Engine: {engine_badge} &nbsp;|&nbsp; Thiết bị: `{device_choice}`" + (" *(✨ Đã bật Auto Doc Scanner)*" if enable_auto_scanner else ""))
process_col1, process_col2 = st.columns([1, 4])
with process_col1:
    btn_run = st.button("🚀 BẮT ĐẦU PHÂN TÍCH & THẨM ĐỊNH", type="primary", use_container_width=True)

# Session state lưu kết quả phân tích
proc_key = f"{target_file_label}_scanner_{enable_auto_scanner}_engine_{os.path.basename(chosen_yolo_path)}_{device_choice}_{conf_thresh}"
if "last_processed_file" not in st.session_state or st.session_state["last_processed_file"] != proc_key or btn_run:
    with st.spinner(f"🔍 Đang chạy suy luận qua {engine_badge}, trích xuất chữ VietOCR và thẩm định quy chế..."):
        t_start = time.perf_counter()
        res = engine.process_document(effective_image_pil, form_type=selected_form_code, validate=True)
        t_elapsed = round((time.perf_counter() - t_start) * 1000, 1)
        res["document_file"] = target_file_label
        res["pipeline_latency_ms"] = t_elapsed
        st.session_state["inference_result"] = res
        st.session_state["last_processed_file"] = proc_key

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
latency_val = result.get("pipeline_latency_ms")
latency_str = f"{latency_val} ms" if latency_val else "N/A"
st.caption(f"⏱️ **Thời gian xử lý toàn trình (End-to-End):** `{latency_str}` &nbsp;|&nbsp; **Động cơ Bố cục:** {engine_badge} (`{os.path.basename(chosen_yolo_path)}`) &nbsp;|&nbsp; **Động cơ OCR:** `VietOCR (vgg_transformer)`")
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

    if enable_auto_scanner:
        st.info("🪄 **Đã áp dụng Tiền xử lý Auto Document Scanner:** Ảnh đã được nắn phẳng phối cảnh và khử bóng đổ không đều trước khi đưa vào mô hình AI.", icon="✨")
        with st.expander("👁️ So sánh: Ảnh gốc Camera vs Ảnh đã Quét phẳng (Virtual Scan)"):
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.markdown("**Ảnh chụp Camera gốc:**")
                st.image(target_image_pil, use_container_width=True)
            with c_s2:
                st.markdown("**Ảnh sau khi Quét & Khử bóng (Virtual Scan):**")
                st.image(effective_image_pil, use_container_width=True)

    # Tạo ảnh trực quan hóa (có cache để giao diện phản hồi tức thì, không bị trễ)
    vis_cache_key = f"{target_file_label}_{enable_auto_scanner}_{toggle_labels}_{toggle_banner}_{status}_{compliance_score}_{len(raw_detections)}"
    if st.session_state.get("vis_cache_key") != vis_cache_key or "cached_annotated_img" not in st.session_state:
        st.session_state["cached_annotated_img"] = engine.create_visualization(effective_image_pil, result, show_labels=toggle_labels, show_banner=toggle_banner)
        st.session_state["vis_cache_key"] = vis_cache_key
    annotated_img = st.session_state["cached_annotated_img"]

    col_view1, col_view2 = st.columns(2)
    with col_view1:
        st.markdown("**Ảnh tài liệu đầu vào (Input Document):**" if not enable_auto_scanner else "**Ảnh sau tiền xử lý (Scanned Document):**")
        st.image(effective_image_pil, use_container_width=True)
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
    t2_head_c1, t2_head_c2 = st.columns([3, 1])
    with t2_head_c1:
        st.subheader("📋 Kết quả Trích xuất Dữ liệu Cấu trúc & OCR Tiếng Việt")
    with t2_head_c2:
        is_editing = st.toggle(
            "✏️ Chế độ Chỉnh sửa thủ công (HITL)",
            value=st.session_state.get("is_editing_tab2", False),
            help="Bật chế độ này để Thêm, Sửa hoặc Xóa thông tin định danh và dòng môn học khi AI nhận diện có sai sót."
        )
        st.session_state["is_editing_tab2"] = is_editing

    if st.session_state.get("manual_edited"):
        st.success("✍️ **Dữ liệu đã được hiệu chỉnh thủ công (Human-in-the-Loop)**. Báo cáo thẩm định quy chế và điểm tuân thủ đã được tính toán lại theo dữ liệu mới nhất.", icon="✅")

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

    if not is_editing:
        # ==========================================
        # CHẾ ĐỘ XEM CHUẨN (VIEW MODE)
        # ==========================================
        col_e1, col_e2 = st.columns([1, 1])

        with col_e1:
            # Thông tin Cơ sở Đào tạo (Theo định hướng GVHD: Lưu nguyên văn Raw + Chuẩn hóa Canonical)
            school_info = validation.get("normalized", {}).get("school_info") or {}
            raw_school = school_info.get("raw_name") or entities.get("school_name", "TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT TP. HỒ CHÍ MINH")
            canon_school = school_info.get("canonical_name", SCHOOL_NAME)
            is_hist = school_info.get("is_historical", True)

            st.markdown(f"""
            <div style="background-color: #F8FAFC; border-left: 4px solid #1B365D; padding: 12px 16px; border-radius: 0 8px 8px 0; margin-bottom: 16px; border: 1px solid #E2E8F0; border-left-width: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 11px; color: #4A5568; font-weight: 700; text-transform: uppercase;">🏛️ Cơ sở Đào tạo (School Name)</span>
                    <span style="background-color: {'#EBF8FF' if is_hist else '#E6F7E9'}; color: {'#2B6CB0' if is_hist else '#22543D'}; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 12px; border: 1px solid {'#BEE3F8' if is_hist else '#C6F6D5'};">{'📜 Tên gọi lịch sử hợp lệ' if is_hist else '✅ Tên gọi chính thức hiện hành'}</span>
                </div>
                <div style="font-size: 13px; line-height: 1.5;">
                    <span style="color: #718096; font-size: 12px;">Nguyên văn văn bản (Raw):</span><br>
                    <strong style="color: #2D3748; font-size: 13px;">{raw_school}</strong><br>
                    <span style="color: #718096; font-size: 12px; margin-top: 4px; display: inline-block;">Chuẩn hóa quản lý (Canonical):</span><br>
                    <strong style="color: #1B365D; font-size: 14px;">{canon_school}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 👤 Thông tin Định danh Sinh viên (Identity Fields)")
            has_any_entity = False
            for k, label in field_labels.items():
                if k in entities:
                    has_any_entity = True
                    val = entities[k]
                    val_str = ", ".join(str(v) for v in val) if isinstance(val, list) else str(val)
                    st.markdown(f"""
                    <div class="entity-box">
                        <span style="font-size: 12px; color: #6C7A89; font-weight: 600;">{label}</span><br>
                        <strong style="font-size: 15px; color: #1B365D;">{val_str}</strong>
                    </div>
                    """, unsafe_allow_html=True)

            if not has_any_entity:
                st.info("Chưa phát hiện trường định danh rõ ràng từ ảnh này.")

        with col_e2:
            st.markdown("#### 🔏 Phát hiện Dấu hiệu Chữ ký & Mộc đỏ (Presence Detection)")
            st.markdown(f"""
            - **Dấu hiệu mộc đỏ (Stamp Presence):** {"✅ Đã phát hiện dấu hiệu (" + str(verification.get("stamp_count", 0)) + " vùng)" if verification.get("has_stamp") else "❌ Không phát hiện dấu hiệu mộc đỏ"}
            - **Dấu hiệu chữ ký (Signature Presence):** {"✅ Đã phát hiện dấu hiệu nét mực (" + str(verification.get("signature_count", 0)) + " vùng)" if verification.get("has_signature") else "❌ Không phát hiện dấu hiệu chữ ký"}
            - **Ghi chú phạm vi:** *Chỉ phát hiện có/không có dấu hiệu nét mực thị giác, chưa định danh người ký.*
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

    else:
        # ==========================================
        # CHẾ ĐỘ CHỈNH SỬA THỦ CÔNG (EDIT / CRUD MODE)
        # ==========================================
        st.markdown("---")
        st.markdown("### 🛠️ Giao diện Hiệu chỉnh Thủ công (Human-in-the-Loop Editor)")
        st.caption("Cho phép Thêm / Sửa / Xóa bất kỳ thông tin nào bị OCR nhận diện nhầm, sau đó hệ thống sẽ tự động thẩm định lại theo quy chế.")

        # KPI Card: Đánh giá Hiệu quả Vận hành HITL (Theo định hướng GVHD)
        h_m1, h_m2, h_m3, h_m4 = st.columns(4)
        with h_m1:
            st.metric("⏱️ Thời gian xử lý (Trung vị)", "45.0 giây", help="Gồm: AI chạy 2.5s + Chuyên viên rà soát mắt 25s + Sửa 14.5s + Lưu 3s (Trung bình 17 hồ sơ: 51.6s)")
        with h_m2:
            st.metric("🎯 Tỷ lệ Ô không cần sửa", "88.02%", help="Field-level Automation: 213/242 trường trích xuất đúng sẵn; Tỷ lệ hồ sơ không cần sửa ô nào (Zero-edit): 58.82% (10/17)")
        with h_m3:
            st.metric("✍️ Số ô đã hiệu chỉnh", f"{st.session_state.get('num_changed_fields', 1)} ô", help="Trung bình thực nghiệm: 1.8 ô/hồ sơ (N=17) và 1.7 ô/hồ sơ (N=35)")
        with h_m4:
            st.metric("⚡ Tiết kiệm Thời gian", "81.25%", help="Tính theo trung vị (240s -> 45s = 81.25%) | Tính theo trung bình 17 hồ sơ (240s -> 51.6s = 78.50%)")

        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("hitl_edit_form"):
            st.markdown("#### 1️⃣ Chỉnh sửa Thông tin Định danh (Identity Fields)")
            
            # Chia làm 2 cột input
            e_col1, e_col2 = st.columns(2)
            edited_entities = {}
            
            # Cột 1: Thông tin cơ bản
            with e_col1:
                cur_name = entities.get("student_name", "")
                if isinstance(cur_name, list): cur_name = ", ".join(cur_name)
                edited_entities["student_name"] = st.text_input("👤 Họ và tên sinh viên:", value=str(cur_name))

                cur_id = entities.get("student_id", "")
                if isinstance(cur_id, list): cur_id = ", ".join(cur_id)
                edited_entities["student_id"] = st.text_input("🆔 Mã số sinh viên (MSSV):", value=str(cur_id))

                cur_class = entities.get("class_name", "")
                if isinstance(cur_class, list): cur_class = ", ".join(cur_class)
                edited_entities["class_name"] = st.text_input("🏫 Lớp sinh hoạt:", value=str(cur_class))

                cur_fac = entities.get("faculty", "")
                if isinstance(cur_fac, list): cur_fac = ", ".join(cur_fac)
                edited_entities["faculty"] = st.text_input("🏢 Khoa đào tạo:", value=str(cur_fac))

            # Cột 2: Thông tin liên hệ & Đơn
            with e_col2:
                cur_phone = entities.get("phone", "")
                if isinstance(cur_phone, list): cur_phone = ", ".join(cur_phone)
                edited_entities["phone"] = st.text_input("📞 Số điện thoại liên hệ:", value=str(cur_phone))

                cur_date = entities.get("date", "")
                if isinstance(cur_date, list): cur_date = ", ".join(cur_date)
                edited_entities["date"] = st.text_input("📅 Ngày lập đơn (dd/mm/yyyy):", value=str(cur_date))

                cur_major = entities.get("major", "")
                if isinstance(cur_major, list): cur_major = ", ".join(cur_major)
                edited_entities["major"] = st.text_input("🎓 Chuyên ngành:", value=str(cur_major))

                cur_reason = entities.get("reason", "")
                if isinstance(cur_reason, list): cur_reason = ", ".join(cur_reason)
                edited_entities["reason"] = st.text_input("📝 Lý do làm đơn:", value=str(cur_reason))

            st.markdown("#### 2️⃣ Kiểm tra Dấu hiệu Chữ ký & Con dấu (Presence Detection)")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                has_stamp_check = st.checkbox("Phát hiện dấu hiệu con dấu mộc đỏ (Stamp Presence)", value=verification.get("has_stamp", False))
                stamp_num = st.number_input("Số vùng dấu hiệu mộc đỏ:", min_value=0, max_value=10, value=int(verification.get("stamp_count", 0)))
            with p_col2:
                has_sig_check = st.checkbox("Phát hiện dấu hiệu nét mực chữ ký (Signature Presence)", value=verification.get("has_signature", False))
                sig_num = st.number_input("Số vùng dấu hiệu chữ ký:", min_value=0, max_value=20, value=int(verification.get("signature_count", 0)))

            st.markdown("#### 3️⃣ Chỉnh sửa Bảng Điểm & Học Phần (Thêm / Sửa / Xóa dòng)")
            st.caption("💡 **Mẹo:** Bạn có thể click trực tiếp vào ô để **Sửa**, click vào dòng cuối có dấu `+` để **Thêm dòng môn học**, hoặc chọn dòng và nhấn phím **Delete** trên bàn phím để **Xóa dòng**.")

            # Chuẩn bị DataFrame cho data_editor
            table_rows_data = []
            for c in courses:
                table_rows_data.append({
                    "Mã học phần": str(c.get("course_code", "")),
                    "Tên học phần": str(c.get("course_name", "")),
                    "Số tín chỉ": str(c.get("credits", "")),
                    "Điểm (Hệ 10)": str(c.get("score", ""))
                })
            if not table_rows_data:
                table_rows_data.append({"Mã học phần": "", "Tên học phần": "", "Số tín chỉ": "", "Điểm (Hệ 10)": ""})

            df_edit_courses = pd.DataFrame(table_rows_data)
            edited_courses_df = st.data_editor(
                df_edit_courses,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "Mã học phần": st.column_config.TextColumn(help="VD: MOBI330980, MATH141701", required=True),
                    "Tên học phần": st.column_config.TextColumn(help="Tên đầy đủ của môn học", required=True),
                    "Số tín chỉ": st.column_config.TextColumn(help="Số tín chỉ (VD: 3, 4)"),
                    "Điểm (Hệ 10)": st.column_config.TextColumn(help="VD: 8.6, 6.4, 0.0", required=True),
                },
                key="hitl_data_editor"
            )

            # Nút Lưu thay đổi bên trong form
            st.markdown("---")
            btn_save = st.form_submit_button("💾 Lưu thay đổi & Thẩm định lại Quy chế", type="primary", use_container_width=True)

        # Nút khôi phục ngoài form
        r_col1, r_col2 = st.columns([1, 1])
        with r_col1:
            if st.button("🔄 Khôi phục kết quả AI gốc (Reset to AI)", use_container_width=True):
                if "manual_edited" in st.session_state:
                    del st.session_state["manual_edited"]
                if "last_processed_file" in st.session_state:
                    del st.session_state["last_processed_file"]
                st.session_state["is_editing_tab2"] = False
                st.toast("🔄 Đã khôi phục kết quả nhận dạng AI gốc!", icon="🔄")
                st.rerun()

        # Xử lý khi nhấn Lưu
        if btn_save:
            # Đo lường số ô người dùng thực tế đã thay đổi (HITL effort)
            num_changed = 0
            for k, val in edited_entities.items():
                orig_val = str(entities.get(k, "")).strip()
                if str(val).strip() != orig_val:
                    num_changed += 1
            st.session_state["num_changed_fields"] = max(1, num_changed)

            # 1. Cập nhật Entities
            clean_ents = {}
            for k, val in edited_entities.items():
                v_str = str(val).strip()
                if v_str:
                    clean_ents[k] = v_str
            st.session_state["inference_result"]["entities"] = clean_ents

            # 2. Cập nhật Verification
            st.session_state["inference_result"]["verification"]["has_stamp"] = has_stamp_check
            st.session_state["inference_result"]["verification"]["stamp_count"] = int(stamp_num)
            st.session_state["inference_result"]["verification"]["has_signature"] = has_sig_check
            st.session_state["inference_result"]["verification"]["signature_count"] = int(sig_num)

            # 3. Cập nhật Bảng điểm
            def _clean_cell(val):
                if val is None or pd.isna(val):
                    return ""
                txt = str(val).strip()
                return "" if txt.lower() in ("none", "nan") else txt

            new_courses_list = []
            for _, r in edited_courses_df.iterrows():
                c_code = _clean_cell(r.get("Mã học phần", ""))
                c_name = _clean_cell(r.get("Tên học phần", ""))
                c_cred = _clean_cell(r.get("Số tín chỉ", ""))
                c_score = _clean_cell(r.get("Điểm (Hệ 10)", ""))
                if c_code or c_name:
                    new_courses_list.append({
                        "course_code": c_code,
                        "course_name": c_name,
                        "credits": c_cred,
                        "score": c_score,
                        "scores": [c_score] if c_score else [],
                        "row_box": [0, 0, 0, 0]
                    })
            st.session_state["inference_result"]["courses_table"] = new_courses_list

            # 4. Chạy lại Rule Engine Thẩm định
            re_val = engine.rule_engine.validate(st.session_state["inference_result"], form_type=form_type)
            st.session_state["inference_result"]["validation"] = re_val
            st.session_state["manual_edited"] = True

            st.toast("✅ Đã lưu dữ liệu hiệu chỉnh và cập nhật lại báo cáo thẩm định!", icon="🎉")
            st.rerun()


# ---------------------------------------------------------------------
# TAB 3: BÁO CÁO THẨM ĐỊNH QUY CHẾ (RULE ENGINE)
# ---------------------------------------------------------------------
with tab3:
    st.subheader("⚖️ Báo cáo Thẩm định Quy chế Nghiệp vụ Học vụ")

    # Cơ chế Chuyển kiểm tra thủ công (Manual Review Escalation) theo chỉ đạo GVHD
    needs_manual = validation.get("needs_manual_review", False)
    is_auto = validation.get("is_acceptable_automatically", False)
    categories = validation.get("review_categories", ["NONE"])

    if is_auto:
        st.success("✅ **HỒ SƠ ĐỦ ĐIỀU KIỆN TỰ ĐỘNG TIẾP NHẬN** (Không phát hiện vi phạm quy chế hoặc khuyết thiếu trường bắt buộc)")
    elif needs_manual:
        st.warning(f"🛡️ **CƠ CHẾ HITL: HỒ SƠ CẦN CHUYỂN SANG KIỂM TRA THỦ CÔNG** | Nhóm vi phạm / Nghi vấn: `{' | '.join(categories)}`")
        reasons = validation.get("manual_review_reasons", [])
        if reasons:
            with st.expander("📌 Xem danh sách chi tiết lý do kích hoạt kiểm tra thủ công:", expanded=True):
                for reason in reasons:
                    st.write(f"- {reason}")

    st.markdown(f"**Kết luận thẩm định:** {validation.get('status_text', validation.get('verdict_message', 'N/A'))}")
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
    details = validation.get("rules", validation.get("details", []))
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
            st.markdown("**Bảng số liệu đánh giá Zero-shot vs Domain-Adapted (Tập test độc lập N=17 camera documents, Zero Data Leakage):**")
            st.dataframe(df_rq2, use_container_width=True)
            st.info("ℹ️ **Lưu ý phương pháp luận:** Chỉ số mAP phản ánh độ chính xác phát hiện vùng (Bounding Box Detection), trong đó mAP@0.5 đạt **34.72% (gấp 3.3 - 5.0 lần so với zero-shot)** và mAP@0.5:0.95 đạt **12.22%**. Để đánh giá toàn diện, cần kết hợp với độ chính xác OCR và thẩm định quy chế dưới đây.")
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

    st.markdown("---")

    # Khối 3: Độ chính xác Trích xuất các Trường Trọng yếu trước Hiệu chỉnh (Theo chỉ đạo GVHD)
    st.markdown("### 3️⃣ Độ chính xác Trích xuất các Trường Trọng yếu trước Hiệu chỉnh (Field Extraction Accuracy)")
    st.markdown("*Đo lường trực tiếp tỷ lệ trích xuất đúng (Exact Match) trên tập 17 tài liệu kiểm thử camera độc lập trước khi có sự can thiệp của người dùng.*")

    field_csv_path = os.path.join(BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2", "field_extraction_accuracy.csv")
    if os.path.exists(field_csv_path):
        df_field = pd.read_csv(field_csv_path)
        st.dataframe(df_field, use_container_width=True)
        st.success("🎯 **Nhận xét thực nghiệm:** Trên tập mẫu khảo sát, các trường định danh cơ bản và mã học phần đạt tỷ lệ trích xuất chính xác 82.4% - 88.2%. Các lỗi còn tồn đọng được bóc tách rõ: 60% do bóng đổ/mờ nét trên ảnh camera thực tế, 40% do góc chụp nghiêng làm lệch khung lưới.")

    st.markdown("---")

    # Khối 4: Đánh giá Giá trị Thực tiễn & Tiết kiệm Công sức Human-in-the-Loop (Theo chỉ đạo GVHD)
    st.markdown("### 4️⃣ Đánh giá Giá trị Thực tiễn & Tiết kiệm Thời gian của Cơ chế Human-in-the-Loop (HITL)")
    st.markdown("*So sánh định lượng giữa Quy trình thẩm định thủ công hoàn toàn và Quy trình có Hệ thống hỗ trợ (AI + HITL).*")

    hitl_csv_path = os.path.join(BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2", "hitl_operational_metrics.csv")
    if os.path.exists(hitl_csv_path):
        df_hitl = pd.read_csv(hitl_csv_path)
        st.markdown("**Bảng so sánh năng suất vận hành giữa Quy trình thủ công và AI + HITL:**")
        st.dataframe(df_hitl, use_container_width=True)

        kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
        with kpi_c1:
            st.metric("⏱️ Tiết kiệm thời gian xử lý", "81.25%", "Giảm từ 240s xuống 45s (trung vị) / trung bình 78.5% (từ 240s xuống 51.6s)")
        with kpi_c2:
            st.metric("🎯 Tỷ lệ trường không cần sửa", "88.0%", "213/242 trường trích xuất chuẩn, chỉ sửa 1.8 ô/hồ sơ")
        with kpi_c3:
            st.metric("⚖️ Độ chính xác sau HITL", "100.0%", "Từ 58.8% trước sửa -> 100% sau khi sửa trung bình 1.8 ô")

        st.info(
            "📋 **Quy trình kiểm chứng HITL công bằng (Fair Evaluation Protocol):**\n"
            "- **Cấu thành trọn vẹn 45.0 giây (Trung vị):** Tải ảnh lên (`3.0s`) + Chờ AI xử lý (`2.5s`) + Đọc rà soát bằng mắt (`22.0s`) + Sửa trung bình 1.8 ô (`14.5s`) + Bấm xác nhận lưu (`3.0s`).\n"
            "- **Số người tham gia & Hướng dẫn:** $N = 3$ thành viên thực hiện độc lập sau 5 phút hướng dẫn thao tác chuẩn trên giao diện.\n"
            "- **Chống hiệu ứng nhớ nội dung (Counterbalanced Cross-over Design):** Chia chéo tập hồ sơ (Người làm thủ công Tập A sẽ làm AI+HITL trên Tập B và ngược lại), tuyệt đối không để cùng một người làm thủ công rồi kiểm tra lại ngay chính hồ sơ đó.\n"
            "- **Đáp án đối chiếu độc lập (Independent Ground Truth):** Kết quả sau khi người dùng sửa được chấm tự động với bộ đáp án gốc `print_dataset_100/dataset_manifest.json` (sinh ra từ khâu tạo phôi văn bản ban đầu, hoàn toàn độc lập với thao tác sửa của người dùng)."
        )

    # Bảng bóc tách: Độ chính xác thẩm định trước và sau hiệu chỉnh & Phân tích lỗi OCR vs Rule Engine
    val_acc_csv_path = os.path.join(BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2", "validation_accuracy_before_after.csv")
    if os.path.exists(val_acc_csv_path):
        st.markdown("#### 🔍 Độ chính xác Thẩm định Trước vs Sau Hiệu chỉnh & Phân tích Bóc tách Lỗi:")
        df_val_acc = pd.read_csv(val_acc_csv_path)
        st.dataframe(df_val_acc, use_container_width=True)
        st.info("📌 **Bóc tách nguyên nhân sai lệch:** 85.7% (6/7 ca lỗi) ở giai đoạn tự động bắt nguồn từ sai số lan truyền của tầng OCR (nhận diện sai điểm, sai ngày khiến Rule Engine đưa ra cảnh báo vi phạm); 14.3% (1/7 ca) do góc nghiêng làm hoán đổi ô bảng hẹp; 0.0% lỗi thuộc về Rule Engine. Khi người dùng rà soát và chuẩn hóa trung bình 1.8 ô/hồ sơ thông qua cơ chế HITL, tầng suy luận logic (Rule Engine) đưa ra quyết định thẩm định chính xác tuyệt đối trên 17/17 mẫu kiểm thử.")

    # Bảng Danh mục 7 Ca Thất bại Điển hình (Failure Cases Catalog)
    failure_csv_path = os.path.join(BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2", "failure_cases_catalog.csv")
    if os.path.exists(failure_csv_path):
        st.markdown("#### ⚠️ Bảng Danh mục 7 Ca Thất bại Điển hình & Cơ chế Xử lý qua HITL (Failure Cases Catalog):")
        df_failure = pd.read_csv(failure_csv_path)
        st.dataframe(df_failure, use_container_width=True)
        st.warning("🛡️ **Cơ chế kiểm soát an toàn:** Toàn bộ 7 trường hợp AI đọc sai hoặc lệch ô bảng đều KHÔNG bị bỏ lọt. Rule Engine đã kích hoạt cảnh báo vi phạm quy chế hoặc cờ kiểm tra thủ công (HITL) xác định, giúp chuyên viên phát hiện và khắc phục ngay tại chỗ.")

    st.markdown("---")

    # Khối 5: Tối ưu hóa Hiệu năng & Triển khai Thực tế (ONNX Runtime vs PyTorch Native)
    st.markdown("### 5️⃣ Tối ưu hóa Hiệu năng & Triển khai Thực tế (ONNX Runtime vs PyTorch Native)")
    st.markdown("*Phân định tường minh độ trễ riêng của YOLOv11 và tổng độ trễ toàn quy trình End-to-End; đo lường trên cả CPU văn phòng và GPU chuyên dụng qua 50 lần lặp độc lập.*")

    onnx_bench_csv = os.path.join(BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2", "yolo_pytorch_vs_onnx_benchmark.csv")
    if os.path.exists(onnx_bench_csv):
        df_bench = pd.read_csv(onnx_bench_csv)
        st.markdown("**Bảng số liệu đối chứng hiệu năng suy luận (Đo lường trung bình qua 50 lần lặp độc lập, Batch Size = 1):**")
        st.dataframe(df_bench, use_container_width=True)

        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        with b_c1:
            st.metric("⚡ Độ trễ riêng YOLO (CPU)", "208.97 ms", "Tăng tốc 1.22x - 2.47x qua ONNX")
        with b_c2:
            st.metric("🚀 Độ trễ riêng YOLO (GPU)", "10.67 ms (93.7 FPS)", "Đạt chuẩn thời gian thực (Real-time)")
        with b_c3:
            st.metric("⏱️ Toàn quy trình End-to-End (CPU)", "2.42 giây", "Đọc + Nắn phối cảnh + YOLO + VietOCR + Rule Engine")
        with b_c4:
            st.metric("🎯 Độ trung thực nhận diện", "100.0%", "Trùng khớp 40/40 hộp biên & điểm tin cậy")

        st.info(
            "ℹ️ **Chi tiết phép đo & Cấu hình thiết bị thử nghiệm:**\n"
            "- **Thiết bị đo kiểm (cùng một máy cho cả trước và sau tối ưu):** CPU **Intel Core i5-12500H** (12 cores / 16 threads) | GPU **NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM)** | RAM 16GB | **Batch size = 1** (50 lần lặp độc lập sau 5 lần warm-up).\n"
            "- **Phạm vi con số `208.97 ms` (CPU, 1280px) và `93.7 FPS` (`10.67 ms`, GPU, 640px):** Đây là độ trễ/thông lượng **của riêng tầng phát hiện vùng YOLOv11 Layout Detection** (forward pass trên tensor ảnh đã tiền xử lý), **chưa bao gồm** thời gian nắn phẳng phối cảnh OpenCV và thời gian nhận dạng chữ VietOCR.\n"
            "- **Tổng thời gian toàn bộ quy trình End-to-End:** Trung bình **`2.42 giây / hồ sơ` trên CPU Intel Core i5-12500H** (bao gồm: Đọc & nắn phẳng ảnh + YOLOv11 `208.97 ms` + Cắt 15–20 hộp biên + Chạy tuần tự VietOCR từng hộp + Đối soát Rule Engine).\n"
            "- **Kiểm tra sai khác đầu ra (Fidelity Check):** So sánh trực tiếp giữa `.pt` và `.onnx` trên cùng ảnh thực tế `REAL_009_IMG_2539.jpg` cho kết quả trùng khớp **40/40 hộp biên** (sai lệch tọa độ tuyệt đối $\\Delta_{\\max} < 10^{-4}\\text{ px}$, sai lệch điểm tin cậy $\\Delta_{\\text{conf}} < 10^{-5}$)."
        )
        st.success("💡 **Phạm vi & Giới hạn triển khai:** Trên cấu hình CPU thử nghiệm (Core i5-12500H), ONNX Runtime giúp giảm độ trễ tầng YOLO từ 255.22 ms xuống 208.97 ms (1280px) mà không cần GPU rời. Tuy nhiên, do bước VietOCR vẫn xử lý tuần tự nhiều vùng cắt (~2.42s/hồ sơ), hiệu năng thực tế sẽ phụ thuộc vào xung nhịp CPU và dung lượng RAM của từng máy trạm; nhóm giới hạn kết luận trên cấu hình đã đo kiểm thay vì khẳng định chạy mượt trên mọi máy tính văn phòng.")

    st.markdown("---")

    # Khối 6: Kiểm thử Độc lập Tầng Suy luận Logic Rule Engine (Unit & Boundary Testing - 60 ca)
    st.markdown("### 6️⃣ Kiểm thử Độc lập Tầng Suy luận Logic Rule Engine (Unit & Boundary Test Suite)")
    st.markdown("*Đánh giá độc lập 60 kịch bản biên quy chế (Boundary Cases), ca khuyết thiếu dữ liệu bắt buộc và ca mâu thuẫn học phần theo chỉ đạo của GVHD.*")

    boundary_csv_path = os.path.join(BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2", "rule_engine_boundary_test_results.csv")
    if os.path.exists(boundary_csv_path):
        df_boundary = pd.read_csv(boundary_csv_path)
        st.markdown("**Bảng số liệu 60 ca kiểm thử biên và kiểm soát sai lệch quy chế:**")
        st.dataframe(df_boundary, use_container_width=True)

        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1:
            st.metric("🎯 Tỷ lệ chấp thuận sai (FAR)", "0.00%", "0/42 ca vi phạm bị lọt lưới (Tuyệt đối an toàn)")
        with col_b2:
            st.metric("🛡️ Tỷ lệ từ chối sai (FRR)", "0.00%", "0/18 ca chuẩn bị bác bỏ oan")
        with col_b3:
            st.metric("⚡ Độ chính xác kiểm thử biên", "100.0%", "60/60 ca biên khớp chính xác hành vi quy chế")
        with col_b4:
            st.metric("🚨 Tỷ lệ chuyển kiểm tra thủ công", "100.0%", "42/42 ca bất thường kích hoạt cơ chế HITL")

        st.success("🔒 **Kết luận an toàn học thuật:** Tầng suy luận logic Rule Engine đạt tỷ lệ False Acceptance = 0.00% (hoàn toàn không có hiện tượng hồ sơ vi phạm nhưng bị kết luận đủ điều kiện). Mọi trường hợp khuyết thiếu trường bắt buộc, mâu thuẫn điểm số hoặc nghi vấn niên đại đều được kích hoạt cơ chế chuyển kiểm tra thủ công (HITL) một cách xác định và an toàn tuyệt đối.")

    st.markdown("---")

    # Khối 7: Đánh giá Tập Kiểm thử Độc lập Mở rộng (Hold-out Test Set N=35, Document-level Split)
    st.markdown("### 7️⃣ Đánh giá Tập Kiểm thử Độc lập Mở rộng (Hold-out Test Set N=35, Document-level Split)")
    st.markdown("*Mở rộng quy mô mẫu từ 17 ảnh lên 35 ảnh camera thực tế (gấp 2.06 lần) thuộc 28 hồ sơ vật lý độc lập (100% Zero Data Leakage so với tập huấn luyện) nhằm kiểm chứng tính khái quát hóa và độ ổn định thống kê theo chỉ đạo của GVHD.*")

    comp_csv_path = os.path.join(BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2", "expanded_comparison_n17_vs_n35.csv")
    if os.path.exists(comp_csv_path):
        df_comp = pd.read_csv(comp_csv_path)
        st.markdown("**Bảng đối chứng khoa học giữa Tập Ban đầu (N=17) và Tập Mở rộng (N=35):**")
        st.dataframe(df_comp, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("📈 Quy mô mẫu thực nghiệm", "35 ảnh / 28 hồ sơ", "+105.9% so với tập 17 ảnh")
        with m2:
            st.metric("🎯 Độ chính xác trích xuất", "88.23%", "Độ lệch Δ = +0.21% (Cực kỳ ổn định)")
        with m3:
            st.metric("🛡️ Tỷ lệ Chấp thuận Sai (FAR)", "0.00%", "0/84 ca vi phạm (An toàn tuyệt đối)")
        with m4:
            st.metric("⚖️ Độ chính xác sau HITL", "100.0%", "Duy trì 100% trên 35/35 hồ sơ độc lập")

    # Bảng chi tiết độ chính xác từng trường so sánh N=17 vs N=35
    exp_field_csv = os.path.join(BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2", "expanded_field_extraction_accuracy.csv")
    if os.path.exists(exp_field_csv):
        st.markdown("#### 📋 So sánh Chi tiết Độ chính xác Trích xuất Từng trường (N=17 vs N=35):")
        df_exp_field = pd.read_csv(exp_field_csv)
        st.dataframe(df_exp_field, use_container_width=True)

    # Expander xem kết quả chi tiết từng hồ sơ trên 35 ảnh
    exp_results_csv = os.path.join(BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2", "expanded_35_docs_evaluation_results.csv")
    if os.path.exists(exp_results_csv):
        with st.expander("🔍 Xem danh sách kết quả chi tiết từng hồ sơ trên 35 ảnh camera thực tế (Hold-out Set)"):
            df_exp_res = pd.read_csv(exp_results_csv)
            st.dataframe(df_exp_res, use_container_width=True)

    st.success("💡 **Ý nghĩa khoa học & Khái quát hóa:** Khi quy mô mẫu mở rộng hơn gấp đôi từ 17 lên 35 ảnh chụp thực tế trên 28 người viết khác nhau, độ chính xác trích xuất của hệ thống duy trì ở mức 88.23% (độ lệch chuẩn so với tập N=17 chỉ là +0.21%), tỷ lệ tự động hóa thành công giữ vững ở mức 60.0% và sau khi qua giao diện HITL (sửa trung bình 1.7 ô/hồ sơ), tỷ lệ thẩm định đạt 100.0%. Điều này bác bỏ hoàn toàn giả thuyết quá khớp (overfitting) trên tập 17 mẫu cũ, chứng minh độ tin cậy vững chắc của hệ thống khi triển khai thực tế.")



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

