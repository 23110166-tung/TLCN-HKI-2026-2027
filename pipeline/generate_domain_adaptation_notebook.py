# -*- coding: utf-8 -*-
"""
Generator for Kaggle Jupyter Notebook:
YOLOv11 Domain Adaptation Fine-Tuning on Real-World Smartphone Camera Documents.
Tiểu luận chuyên ngành AI - Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh.
"""

import os
import sys
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_NOTEBOOK = os.path.join(BASE_DIR, "train_yolov11_domain_adaptation_kaggle.ipynb")


def generate_domain_adaptation_notebook(output_path=OUTPUT_NOTEBOOK):
    cells = []

    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.splitlines()]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.splitlines()]
        })

    # CELL 1: Header Markdown
    add_md("""# 🎓 TIỂU LUẬN CHUYÊN NGÀNH AI - TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT THÀNH PHỐ HỒ CHÍ MINH
## Đề tài: Xây dựng hệ thống nhận dạng, trích xuất và kiểm tra tính hợp lệ của thông tin từ đơn học vụ và bảng điểm sinh viên bằng học sâu
### 🔬 THỰC NGHIỆM GIAI ĐOẠN 3: GIẢI QUYẾT HIỆN TƯỢNG DOMAIN SHIFT TRÊN ẢNH CHỤP CAMERA THỰC TẾ (YOLOV11)
- **Giảng viên hướng dẫn**: TS. Bùi Mạnh Quân
- **Sinh viên thực hiện**:
  - Nguyễn Bách Tùng (MSSV: 23110166) - Nhóm trưởng
  - Nguyễn Hưng Nguyên (MSSV: 23110035) - Thành viên
  - Trần Cẩm Long (MSSV: 23110122) - Thành viên

---
### 🎯 Mục tiêu khoa học (Trả lời Câu hỏi Nghiên cứu số 2):
1. **Đo lường hiện tượng Domain Shift**: Đánh giá Zero-shot mô hình YOLOv11n (huấn luyện trên dữ liệu tổng hợp Synthetic) khi chuyển sang nhận dạng 120 ảnh chụp camera thực tế từ smartphone.
2. **Kỹ thuật Data Augmentation nâng cao**: Áp dụng biến dạng phối cảnh (Random Perspective), chiếu sáng không đồng đều và bóng đổ cánh tay/điện thoại (Random Shadow), rung tay và mờ nét (Motion Blur).
3. **Fine-Tuning (Transfer Learning)**: Huấn luyện thích nghi miền dữ liệu (Domain Adaptation) trên tập ảnh chụp thực tế có chia tách strictly theo Document ID (Zero Data Leakage).
4. **Mục tiêu định lượng**: Nâng mAP@0.5 trên tập ảnh chụp camera thực tế từ **6.89% (zero-shot) lên > 50%**.""")

    # CELL 2: Environment Setup & GPU Check
    add_code("""# 1. KIỂM TRA MÔI TRƯỜNG PHẦN CỨNG & CÀI ĐẶT THƯ VIỆN
import os
import sys
import glob
import shutil
import zipfile
import torch

print("=" * 70)
print("🚀 HỆ THỐNG THỰC NGHIỆM DOMAIN ADAPTATION YOLOV11")
print("=" * 70)

cuda_ok = torch.cuda.is_available()
print(f"🖥️ CUDA Available : {cuda_ok}")
if cuda_ok:
    print(f"🎮 GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"📊 GPU Device Count: {torch.cuda.device_count()}")
    print(f"💾 VRAM Allocated : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ CẢNH BÁO: Đang chạy trên CPU! Vui lòng bật GPU T4 x2 trong Settings -> Accelerator.")

# Cài đặt Ultralytics và tabulate (không dùng --upgrade để tránh xung đột môi trường Kaggle)
import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "tabulate"])

from ultralytics import YOLO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

print("✅ Đã cài đặt và nạp thành công Ultralytics YOLOv11!")""")

    # CELL 3: Dataset Discovery & Preparation
    add_code("""# 2. TỰ ĐỘNG NHẬN DIỆN & GIẢI NÉN TẬP DỮ LIỆU DOMAIN ADAPTATION
import yaml
import subprocess

DATASET_ROOT = "/kaggle/working/dataset_yolo_domain_adaptation"
os.makedirs(DATASET_ROOT, exist_ok=True)

# Tìm file zip hoặc thư mục dataset trong /kaggle/input
zip_candidates = glob.glob("/kaggle/input/**/dataset_yolo_domain_adaptation*.zip", recursive=True)
dir_candidates = glob.glob("/kaggle/input/**/dataset_yolo_domain_adaptation", recursive=True)

if zip_candidates:
    src_zip = zip_candidates[0]
    print(f"📦 Tìm thấy file zip dataset: {src_zip}")
    print("⏳ Đang giải nén tập dữ liệu...")
    with zipfile.ZipFile(src_zip, 'r') as zip_ref:
        zip_ref.extractall("/kaggle/working")
    print("✅ Đã giải nén xong!")
elif dir_candidates:
    src_dir = dir_candidates[0]
    print(f"📂 Tìm thấy thư mục dataset có sẵn: {src_dir}")
    DATASET_ROOT = src_dir
else:
    # Nếu tải trực tiếp từ repository GitHub
    print("🌐 Không tìm thấy zip sẵn, đang kiểm tra clone từ repository...")
    if not os.path.exists("/kaggle/working/TLCN"):
        subprocess.run(["git", "clone", "https://github.com/23110166-tung/TLCN-HKI-2026-2027.git", "/kaggle/working/TLCN"])
    sys.path.insert(0, "/kaggle/working/TLCN")
    from pipeline.prepare_domain_adaptation_dataset import prepare_domain_adaptation_dataset
    DATASET_ROOT, _ = prepare_domain_adaptation_dataset(
        real_dataset_dir="/kaggle/working/TLCN/dataset_yolo_real_world",
        synthetic_dataset_dir="/kaggle/working/TLCN/dataset_yolo",
        output_dir=DATASET_ROOT
    )

# Lưu file cấu hình data_kaggle.yaml vào /kaggle/working (đảm bảo quyền ghi trên Kaggle)
src_yaml = os.path.join(DATASET_ROOT, "data.yaml")
if os.path.exists(src_yaml):
    with open(src_yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
else:
    cfg = {
        'nc': 18,
        'names': ['student_name', 'student_id', 'class_name', 'faculty', 'major', 'dob', 'pob', 'phone', 'email', 'course_code', 'course_name', 'credits', 'score', 'date', 'reason', 'degree_info', 'signature', 'stamp']
    }

cfg['path'] = DATASET_ROOT
cfg['train'] = "images/train"
cfg['val'] = "images/val"
cfg['test'] = "images/test"

data_yaml_path = "/kaggle/working/data_kaggle.yaml"
with open(data_yaml_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"⚙️ File cấu hình data.yaml tại: {data_yaml_path}")
train_imgs = glob.glob(os.path.join(DATASET_ROOT, "images", "train", "*.*"))
val_imgs = glob.glob(os.path.join(DATASET_ROOT, "images", "val", "*.*"))
test_imgs = glob.glob(os.path.join(DATASET_ROOT, "images", "test", "*.*"))

print(f"📊 Thống kê tập dữ liệu Domain Adaptation:")
print(f"   • Train images: {len(train_imgs)} ảnh (Ảnh gốc camera + Augmented + Synthetic Anchors)")
print(f"   • Val images  : {len(val_imgs)} ảnh (100% Ảnh chụp Camera thực tế)")
print(f"   • Test images : {len(test_imgs)} ảnh (100% Ảnh chụp Camera thực tế, Zero Data Leakage)")""")

    # CELL 4: Visual Inspection of Augmented Training Samples
    add_code("""# 3. TRỰC QUAN HÓA CÁC BIẾN THỂ TĂNG CƯỜNG DỮ LIỆU THỰC TẾ (DATA AUGMENTATION)
import cv2
from PIL import Image

train_samples = sorted(glob.glob(os.path.join(DATASET_ROOT, "images", "train", "*REAL*.*")))[:4]
if train_samples:
    fig, axes = plt.subplots(1, min(4, len(train_samples)), figsize=(18, 6), dpi=150)
    if len(train_samples) == 1:
        axes = [axes]
    for ax, img_path in zip(axes, train_samples):
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(os.path.basename(img_path)[:25], fontsize=10, fontweight="bold")
        ax.axis('off')
    plt.suptitle("Các biến thể tăng cường dữ liệu thực tế (Phối cảnh nghiêng, Bóng đổ, Mờ nét)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()""")

    # CELL 5: Locating Pretrained Synthetic Weights
    add_code("""# 4. TÌM KIẾM TRỌNG SỐ TIỀN HUẤN LUYỆN YOLOV11 SYNTHETIC (GIAI ĐOẠN 2)
# Trọng số tốt nhất từ Giai đoạn 2 đã được lưu trong Artifacts hoặc nhúng sẵn trong dataset zip
pretrained_candidates = (
    glob.glob("/kaggle/input/**/pretrained_synthetic_best.pt", recursive=True) +
    glob.glob("/kaggle/working/**/pretrained_synthetic_best.pt", recursive=True) +
    glob.glob("/kaggle/input/**/hcmute_yolo11_layout/**/best.pt", recursive=True) +
    glob.glob("/kaggle/input/**/best.pt", recursive=True) +
    glob.glob("/kaggle/working/**/best.pt", recursive=True)
)

if pretrained_candidates:
    SYNTHETIC_WEIGHTS = pretrained_candidates[0]
    print(f"📦 Tìm thấy trọng số Pretrained Synthetic (Stage 2): {SYNTHETIC_WEIGHTS}")
    print("   -> Sẵn sàng Transfer Learning từ mô hình đạt mAP 98.6% trên biểu mẫu HCMUTE!")
else:
    print("⚠️ Không tìm thấy trọng số Synthetic có sẵn, đang khởi tạo yolo11s.pt (YOLOv11 Small)...")
    SYNTHETIC_WEIGHTS = "yolo11s.pt"

print("✅ Đã sẵn sàng trọng số khởi tạo:", SYNTHETIC_WEIGHTS)""")

    # CELL 6: Step 1 - Baseline Zero-Shot Evaluation on Real-World Test Set
    add_code("""# 5. BƯỚC 1: ĐÁNH GIÁ ĐỊNH LƯỢNG BASELINE ZERO-SHOT TRÊN TẬP TEST CAMERA THỰC TẾ
print("=" * 70)
print("🔍 BẮT ĐẦU ĐÁNH GIÁ BASELINE (SYNTHETIC PRETRAINED MODEL TRÊN ẢNH CAMERA)")
print("=" * 70)

model_baseline = YOLO(SYNTHETIC_WEIGHTS)

metrics_baseline = model_baseline.val(
    data=data_yaml_path,
    imgsz=1280,
    split="test",
    device=0 if cuda_ok else "cpu",
    verbose=False,
    plots=False
)

base_p = float(metrics_baseline.box.mp) * 100.0
base_r = float(metrics_baseline.box.mr) * 100.0
base_map50 = float(metrics_baseline.box.map50) * 100.0
base_map95 = float(metrics_baseline.box.map) * 100.0

print("\\n📊 KẾT QUẢ BASELINE (ZERO-SHOT TRÊN TẬP TEST CAMERA):")
print(f"   • Precision (Độ chính xác) : {base_p:.2f}%")
print(f"   • Recall (Độ bao phủ)      : {base_r:.2f}%")
print(f"   • mAP@0.5 (Độ chuẩn)       : {base_map50:.2f}%")
print(f"   • mAP@0.5:0.95 (Chặt chẽ)  : {base_map95:.2f}%")""")

    # CELL 7: Step 2 - Fine-Tuning YOLOv11 with Domain Adaptation
    add_code("""# 6. BƯỚC 2: TIẾN HÀNH FINE-TUNING YOLOV11 THÍCH NGHI MIỀN (DOMAIN ADAPTATION)
print("=" * 70)
print("🚀 BẮT ĐẦU HUẤN LUYỆN FINE-TUNING YOLOV11 TRÊN DỮ LIỆU DOMAIN ADAPTATION ĐÃ LÀM SẠCH")
print("=" * 70)

# Khởi tạo mô hình từ trọng số Pretrained Synthetic
model_ft = YOLO(SYNTHETIC_WEIGHTS)

# Cấu hình siêu tham số Fine-Tuning tối ưu hóa cho Document Layout Detection:
# - imgsz=1280: Nhận diện sắc nét các ô chữ nhỏ (tín chỉ, điểm số, mã môn học)
# - batch=8: An toàn tuyệt đối với VRAM 16GB GPU T4
# - epochs=50, patience=15: Đủ thời gian hội tụ mà không bị overfitting
train_results = model_ft.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=1280,
    batch=8 if cuda_ok else 4,
    patience=15,
    save=True,
    device=0 if cuda_ok else "cpu",
    project="/kaggle/working/runs/detect",
    name="hcmute_yolo11_domain_adapted",
    exist_ok=True,
    plots=True,
    optimizer="AdamW",
    lr0=0.0008,
    lrf=0.01,
    weight_decay=0.0005,
    degrees=4.0,
    translate=0.04,
    scale=0.08,
    shear=1.5,
    perspective=0.0004,
    hsv_h=0.015,
    hsv_s=0.3,
    hsv_v=0.3,
    mosaic=0.3,
    mixup=0.1,
    verbose=True
)

BEST_FT_WEIGHTS = "/kaggle/working/runs/detect/hcmute_yolo11_domain_adapted/weights/best.pt"
print(f"🎉 HUẤN LUYỆN HOÀN TẤT! Trọng số tối ưu nhất được lưu tại: {BEST_FT_WEIGHTS}")""")

    # CELL 8: Step 3 - Post Fine-Tuning Evaluation on Same Test Set
    add_code("""# 7. BƯỚC 3: ĐÁNH GIÁ MÔ HÌNH FINE-TUNED TRÊN CÙNG TẬP TEST CAMERA THỰC TẾ
print("=" * 70)
print("🔬 ĐÁNH GIÁ HIỆU NĂNG MÔ HÌNH FINE-TUNED TRÊN CÙNG TẬP TEST CAMERA THỰC TẾ (imgsz=1280)")
print("=" * 70)

model_evaluated = YOLO(BEST_FT_WEIGHTS)

metrics_ft = model_evaluated.val(
    data=data_yaml_path,
    imgsz=1280,
    split="test",
    device=0 if cuda_ok else "cpu",
    verbose=True,
    plots=True
)

ft_p = float(metrics_ft.box.mp) * 100.0
ft_r = float(metrics_ft.box.mr) * 100.0
ft_map50 = float(metrics_ft.box.map50) * 100.0
ft_map95 = float(metrics_ft.box.map) * 100.0

print("\\n📊 KẾT QUẢ FINE-TUNED (DOMAIN ADAPTED TRÊN TẬP TEST CAMERA):")
print(f"   • Precision (Độ chính xác) : {ft_p:.2f}%")
print(f"   • Recall (Độ bao phủ)      : {ft_r:.2f}%")
print(f"   • mAP@0.5 (Độ chuẩn)       : {ft_map50:.2f}%")
print(f"   • mAP@0.5:0.95 (Chặt chẽ)  : {ft_map95:.2f}%")

# Thống kê chi tiết trên từng lớp (Per-Class Performance)
per_class_data = []
class_names = list(metrics_ft.names.values())
for i, c_name in enumerate(class_names):
    p_i = float(metrics_ft.box.p[i]) * 100.0 if i < len(metrics_ft.box.p) else 0.0
    r_i = float(metrics_ft.box.r[i]) * 100.0 if i < len(metrics_ft.box.r) else 0.0
    map50_i = float(metrics_ft.box.ap50[i]) * 100.0 if i < len(metrics_ft.box.ap50) else 0.0
    per_class_data.append({
        "Lớp đối tượng": c_name,
        "Precision": f"{p_i:.1f}%",
        "Recall": f"{r_i:.1f}%",
        "mAP@0.5": f"{map50_i:.1f}%"
    })
df_per_class = pd.DataFrame(per_class_data)
df_per_class.to_csv("/kaggle/working/domain_adaptation_per_class.csv", index=False, encoding="utf-8-sig")
print("\\n📋 CHI TIẾT HIỆU NĂNG 18 LỚP HỌC THUẬT (PER-CLASS METRICS):")
print(tabulate(df_per_class, headers="keys", tablefmt="github", showindex=False))""")

    # CELL 9: Scientific Comparison Table answering RQ2
    add_code("""# 8. BƯỚC 4: TỔNG HỢP SO SÁNH KHOA HỌC TRẢ LỜI CÂU HỎI NGHIÊN CỨU 2 (RQ2)
diff_p = ft_p - base_p
diff_r = ft_r - base_r
diff_map50 = ft_map50 - base_map50
diff_map95 = ft_map95 - base_map95

summary_data = [
    {
        "Chỉ số Đánh giá": "Precision (Độ chính xác)",
        "Baseline (Zero-shot)": f"{base_p:.2f}%",
        "Fine-tuned (Domain-Adapted)": f"{ft_p:.2f}%",
        "Chênh lệch (Δ)": f"{'+' if diff_p >= 0 else ''}{diff_p:.2f}%"
    },
    {
        "Chỉ số Đánh giá": "Recall (Độ bao phủ)",
        "Baseline (Zero-shot)": f"{base_r:.2f}%",
        "Fine-tuned (Domain-Adapted)": f"{ft_r:.2f}%",
        "Chênh lệch (Δ)": f"{'+' if diff_r >= 0 else ''}{diff_r:.2f}%"
    },
    {
        "Chỉ số Đánh giá": "mAP@0.5 (Độ chuẩn nhận diện)",
        "Baseline (Zero-shot)": f"{base_map50:.2f}%",
        "Fine-tuned (Domain-Adapted)": f"{ft_map50:.2f}%",
        "Chênh lệch (Δ)": f"{'+' if diff_map50 >= 0 else ''}{diff_map50:.2f}%"
    },
    {
        "Chỉ số Đánh giá": "mAP@0.5:0.95 (Chặt chẽ IoU)",
        "Baseline (Zero-shot)": f"{base_map95:.2f}%",
        "Fine-tuned (Domain-Adapted)": f"{ft_map95:.2f}%",
        "Chênh lệch (Δ)": f"{'+' if diff_map95 >= 0 else ''}{diff_map95:.2f}%"
    }
]

df_rq2 = pd.DataFrame(summary_data)
csv_rq2_path = "/kaggle/working/domain_adaptation_rq2_results.csv"
df_rq2.to_csv(csv_rq2_path, index=False, encoding="utf-8-sig")

print("=" * 80)
print("🏆 BẢNG TỔNG HỢP HIỆU NĂNG TRẢ LỜI CÂU HỎI NGHIÊN CỨU 2 (RQ2):")
print("   (Đánh giá độc lập trên tập Test gồm các ảnh chụp camera thực tế chưa từng thấy)")
print("=" * 80)
print(tabulate(df_rq2, headers="keys", tablefmt="github", showindex=False))
print("=" * 80)
print(f"💾 File bảng kết quả CSV: {csv_rq2_path}")""")

    # CELL 10: Comparison Visualization Chart
    add_code("""# 9. VẼ BIỂU ĐỒ SO SÁNH TRỰC QUAN HÓA KẾT QUẢ ĐỐI CHỨNG
metric_names = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
base_vals = [base_p, base_r, base_map50, base_map95]
ft_vals = [ft_p, ft_r, ft_map50, ft_map95]

x = np.arange(len(metric_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
rects1 = ax.bar(x - width/2, base_vals, width, label='Baseline (Zero-shot Pretrained)', color='#dc3545', alpha=0.85)
rects2 = ax.bar(x + width/2, ft_vals, width, label='Fine-tuned (Domain-Adapted)', color='#28a745', alpha=0.9)

ax.set_ylabel('Tỉ lệ (%)', fontsize=12, fontweight='bold')
ax.set_title('Đánh giá Định lượng Hiệu năng Giải quyết Domain Shift trên Ảnh chụp Thực tế (YOLOv11)\\n'
             'Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=11, fontweight='bold')
ax.set_ylim(0, max(max(ft_vals) * 1.25, 75.0))
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.4)

for idx, rect in enumerate(rects1):
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                fontsize=10, fontweight='bold')

for idx, rect in enumerate(rects2):
    h = rect.get_height()
    diff = h - base_vals[idx]
    diff_str = f" (+{diff:.1f}%)" if diff > 0 else f" ({diff:.1f}%)"
    ax.annotate(f"{h:.1f}%\\n{diff_str}", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#155724')

plt.tight_layout()
chart_path = "/kaggle/working/domain_adaptation_comparison_chart.png"
plt.savefig(chart_path, dpi=200)
plt.show()
print(f"📊 Đã lưu biểu đồ: {chart_path}")""")

    # CELL 11: Visual Detection on Test Samples
    add_code("""# 10. TRỰC QUAN HÓA KẾT QUẢ DỰ ĐOÁN BOUNDING BOX TRÊN ẢNH TEST THỰC TẾ
test_images = sorted(glob.glob(os.path.join(DATASET_ROOT, "images", "test", "*.*")))[:3]
if test_images:
    for idx, t_img in enumerate(test_images, 1):
        res = model_evaluated(t_img, imgsz=1280, conf=0.35, verbose=False)[0]
        res_plotted = res.plot()
        plt.figure(figsize=(10, 14), dpi=150)
        plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
        plt.title(f"Dự đoán YOLOv11 Domain-Adapted trên Ảnh Test thực tế [{idx}]: {os.path.basename(t_img)}", fontsize=12, fontweight="bold")
        plt.axis('off')
        plt.tight_layout()
        plt.show()""")

    # CELL 12: Packaging Artifacts for Download
    add_code("""# 11. ĐÓNG GÓI ARTIFACTS ĐỂ TẢI VỀ MÁY
output_zip = "/kaggle/working/HCMUTE_YOLOv11_Domain_Adaptation_Artifacts.zip"

with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    if os.path.exists(BEST_FT_WEIGHTS):
        zipf.write(BEST_FT_WEIGHTS, arcname="yolo11_domain_adapted_best.pt")
        print("📦 Đã thêm trọng số best.pt vào file zip")
    if os.path.exists(csv_rq2_path):
        zipf.write(csv_rq2_path, arcname="domain_adaptation_rq2_results.csv")
        print("📄 Đã thêm file CSV kết quả RQ2")
    per_class_csv = "/kaggle/working/domain_adaptation_per_class.csv"
    if os.path.exists(per_class_csv):
        zipf.write(per_class_csv, arcname="domain_adaptation_per_class.csv")
        print("📄 Đã thêm file CSV chi tiết 18 lớp học thuật")
    if os.path.exists(chart_path):
        zipf.write(chart_path, arcname="domain_adaptation_comparison_chart.png")
        print("📊 Đã thêm biểu đồ so sánh")
    # Thêm training curves của YOLOv11
    results_png = "/kaggle/working/runs/detect/hcmute_yolo11_domain_adapted/results.png"
    if os.path.exists(results_png):
        zipf.write(results_png, arcname="training_results_curves.png")
        print("📈 Đã thêm đường cong huấn luyện results.png")
    if os.path.exists(chart_path):
        zipf.write(chart_path, arcname="domain_adaptation_comparison_chart.png")
        print("📊 Đã thêm biểu đồ so sánh")
    # Thêm training curves của YOLOv11
    results_png = "/kaggle/working/runs/detect/hcmute_yolo11_domain_adapted/results.png"
    if os.path.exists(results_png):
        zipf.write(results_png, arcname="training_results_curves.png")
        print("📈 Đã thêm đường cong huấn luyện results.png")

zip_size_mb = os.path.getsize(output_zip) / (1024 * 1024)
print("=" * 70)
print("🎉 HOÀN TẤT ĐÓNG GÓI TOÀN BỘ KẾT QUẢ THỰC NGHIỆM DOMAIN ADAPTATION!")
print(f"📁 File ZIP tải về: {output_zip} ({zip_size_mb:.2f} MB)")
print("👉 Bạn hãy tải file ZIP này về và giải nén vào thư mục dự án tại máy local.")
print("=" * 70)""")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 2,
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã tạo thành công Notebook tại: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_domain_adaptation_notebook()
