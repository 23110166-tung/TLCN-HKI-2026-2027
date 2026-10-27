# -*- coding: utf-8 -*-
"""
Generator script for train_vietocr_finetune_kaggle.ipynb
Builds the complete, self-contained Jupyter Notebook for VietOCR Fine-tuning on Kaggle GPU.
"""

import json
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def build_notebook(output_path="train_vietocr_finetune_kaggle.ipynb"):
    cells = []

    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # CELL 1: Header Markdown
    add_md("""# 🎓 TIỂU LUẬN CHUYÊN NGÀNH AI - TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT THÀNH PHỐ HỒ CHÍ MINH
## Đề tài: Xây dựng hệ thống nhận dạng, trích xuất và kiểm tra tính hợp lệ của thông tin từ đơn học vụ và bảng điểm sinh viên bằng học sâu
### 🔬 THỰC NGHIỆM GIAI ĐOẠN 3: FINE-TUNING VIETOCR (VGG-TRANSFORMER) TRÊN DỮ LIỆU BIỂU MẪU HỌC VỤ
- **Giảng viên hướng dẫn**: TS. Bùi Mạnh Quân
- **Sinh viên thực hiện**:
  - Nguyễn Bách Tùng (MSSV: 23110166) - Nhóm trưởng
  - Nguyễn Hưng Nguyên (MSSV: 23110035) - Thành viên
  - Trần Cẩm Long (MSSV: 23110122) - Thành viên

---
### 🎯 Mục tiêu khoa học (Giải quyết Câu hỏi nghiên cứu 1 trong Đề cương):
> *"Fine-tuning VietOCR trên dữ liệu biểu mẫu của trường cải thiện CER (Character Error Rate), WER (Word Error Rate) và Exact Match như thế nào so với mô hình tiền huấn luyện?"*

---
### ⚙️ HƯỚNG DẪN THIẾT LẬP KAGGLENOTEBOOK:
1. Tại bảng bên phải (**Notebook Settings**):
   - **Accelerator**: Chọn **GPU T4 x2** (hoặc GPU P100).
   - **Internet**: Phải gạt sang **Internet on** (để tải thư viện và pretrained weights).
2. Nhấn nút **Run All** hoặc chạy tuần tự từng cell.""")

    # CELL 2: Install dependencies
    add_code("""# 1. Cài đặt thư viện (KHÔNG kéo pillow==10.2.0/imgaug/numpy cũ của vietocr -> tránh lỗi build trên Kaggle)
%pip install -q --no-deps vietocr
%pip install -q einops gdown prefetch-generator lmdb albumentations editdistance tabulate pymupdf

# vietocr import `imgaug` ở đầu module nhưng Trainer mặc định chỉ dùng albumentations (ImgAugTransformV2).
# imgaug 0.4.0 lỗi với numpy>=2 -> nếu không import được thì tạo stub rỗng.
import sys, types
try:
    import imgaug
except Exception:
    ia = types.ModuleType("imgaug"); ia.__path__ = []; ia.ALL = "all"
    iaa = types.ModuleType("imgaug.augmenters")
    ia.augmenters = iaa
    sys.modules["imgaug"] = ia
    sys.modules["imgaug.augmenters"] = iaa
    print("⚠️ imgaug không khả dụng -> dùng stub (không ảnh hưởng huấn luyện)")

from vietocr.tool.config import Cfg
print("✅ Cài đặt & import vietocr thành công")""")


    # CELL 3: GPU check
    add_code("""# 2. Kiểm tra tài nguyên phần cứng GPU trên Kaggle
import torch
print("=" * 60)
print("🚀 PyTorch Version :", torch.__version__)
print("⚡ CUDA Available   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🎮 GPU Device Name :", torch.cuda.get_device_name(0))
    print("💾 VRAM Dung lượng :", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2), "GB")
    device = "cuda:0"
else:
    print("⚠️ CẢNH BÁO: Đang chạy CPU. Hãy bật GPU tại Notebook Settings -> Accelerator!")
    device = "cpu"
print("=" * 60)""")

    # CELL 4: Data preparation markdown
    add_md("""---
## 📦 BƯỚC 1: CHUẨN BỊ BỘ DỮ LIỆU TEXT CROP (dataset_ocr)
Bộ dữ liệu gồm hơn 15.000 text crops chất lượng cao được trích xuất từ 120 ảnh chụp camera thực tế và 100 vector PDF biểu mẫu học vụ HCMUTE, kèm annotation chuẩn theo định dạng VietOCR:
- `train_annotation.txt`: 13.783 mẫu
- `val_annotation.txt`: 857 mẫu
- `test_annotation.txt`: 867 mẫu (tập test độc lập hoàn toàn, không rò rỉ dữ liệu)""")

    # CELL 5: Auto discover / unzip dataset
    add_code("""# 3. Tự động phát hiện và giải nén bộ dữ liệu dataset_ocr
import os
import glob
import zipfile

DATA_ROOT = None

# 1. Quét đệ quy toàn bộ /kaggle/input (nếu đã add dataset bất kỳ trên Kaggle)
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train_annotation.txt" in files and "crops" in dirs:
            DATA_ROOT = root
            print(f"✅ Đã tìm thấy dataset mount từ Kaggle Input: {DATA_ROOT}")
            break
        elif "train_annotation.txt" in files:
            DATA_ROOT = root
            print(f"✅ Đã tìm thấy annotation từ Kaggle Input: {DATA_ROOT}")
            break

# 2. Kiểm tra nếu có file dataset_ocr.zip trong /kaggle/input hoặc /kaggle/working thì tự động giải nén
if DATA_ROOT is None:
    zip_candidates = glob.glob("/kaggle/input/**/dataset_ocr*.zip", recursive=True) + \\
                     glob.glob("/kaggle/input/**/*.zip", recursive=True) + \\
                     glob.glob("*.zip")
    for z in zip_candidates:
        try:
            with zipfile.ZipFile(z, 'r') as zf:
                if any("train_annotation.txt" in name for name in zf.namelist()):
                    print(f"📦 Phát hiện file nén {z}, đang tự động giải nén...")
                    zf.extractall("/kaggle/working")
                    if os.path.exists("/kaggle/working/dataset_ocr/train_annotation.txt"):
                        DATA_ROOT = "/kaggle/working/dataset_ocr"
                    elif os.path.exists("/kaggle/working/train_annotation.txt"):
                        DATA_ROOT = "/kaggle/working"
                    print(f"✅ Giải nén thành công! DATA_ROOT = {DATA_ROOT}")
                    break
        except Exception:
            pass

# 3. Kiểm tra các thư mục local thông thường
if DATA_ROOT is None:
    for candidate in ["./dataset_ocr", "../dataset_ocr", "dataset_ocr", "TLCN/dataset_ocr"]:
        if os.path.exists(candidate) and os.path.exists(os.path.join(candidate, "train_annotation.txt")):
            DATA_ROOT = candidate
            print(f"✅ Đã tìm thấy thư mục local: {DATA_ROOT}")
            break

# 4. Hướng dẫn chi tiết nếu chưa nạp dữ liệu
if DATA_ROOT is None:
    print("=" * 75)
    print("⚠️ CHƯA TÌM THẤY DỮ LIỆU dataset_ocr TRÊN KAGGLE!")
    print("=" * 75)
    print("Bạn vui lòng chọn 1 trong 2 cách sau để nạp dữ liệu:")
    print("👉 CÁCH 1 (Khuyên dùng - Nhanh & Ổn định nhất):")
    print("   1. Tại bảng bên phải Kaggle, nhấn nút '+ Add Input'.")
    print("   2. Chọn dataset của bạn ('database-hcmute' hoặc tạo mới 'dataset-ocr').")
    print("👉 CÁCH 2 (Kéo thả trực tiếp file zip):")
    print("   1. Trên máy tính của bạn đã có sẵn file 'dataset_ocr.zip' (85MB).")
    print("   2. Tại menu của Kaggle Notebook: Chọn File -> Upload Data hoặc kéo thả")
    print("      file 'dataset_ocr.zip' vào cửa sổ làm việc bên trái.")
    print("   3. Chạy lại cell này, notebook sẽ tự động giải nén trong 5 giây!")
    print("=" * 75)
    raise FileNotFoundError("Chưa tìm thấy dataset_ocr. Vui lòng nạp dataset theo hướng dẫn ở trên!")

print("=" * 60)
print(f"📁 Thư mục dữ liệu chính thức: {DATA_ROOT}")
for f in ["train_annotation.txt", "val_annotation.txt", "test_annotation.txt", "vocab.txt"]:
    f_path = os.path.join(DATA_ROOT, f)
    if os.path.exists(f_path):
        with open(f_path, "r", encoding="utf-8") as _f:
            cnt = len([line for line in _f if line.strip()])
        print(f"   📄 {f:<22}: {cnt} dòng")
print("=" * 60)""")

    # CELL 6: Sample preview
    add_code("""# 4. Hiển thị ngẫu nhiên các mẫu ảnh crop và nhãn thực tế
import random
import matplotlib.pyplot as plt
from PIL import Image

train_ann_file = os.path.join(DATA_ROOT, "train_annotation.txt")
with open(train_ann_file, "r", encoding="utf-8") as f:
    sample_lines = [l.strip().split("\t") for l in f if l.strip() and "\t" in l]

random.seed(42)
preview_samples = random.sample(sample_lines, min(8, len(sample_lines)))

fig, axes = plt.subplots(4, 2, figsize=(14, 8))
axes = axes.flatten()

for i, (rel_path, gt_label) in enumerate(preview_samples):
    img_path = os.path.join(DATA_ROOT, rel_path)
    if os.path.exists(img_path):
        im = Image.open(img_path)
        axes[i].imshow(im)
        axes[i].set_title(f"GT: '{gt_label}'", fontsize=11, color="navy", fontweight="bold")
    axes[i].axis("off")

plt.tight_layout()
plt.suptitle("Một số ảnh Crop huấn luyện trong bộ dữ liệu HCMUTE", fontsize=14, y=1.02)
plt.show()""")

    # CELL 7: Baseline Evaluation Markdown
    add_md("""---
## 🔍 BƯỚC 2: ĐÁNH GIÁ MÔ HÌNH BASELINE (VIETOCR PRETRAINED GỐC)
Để trả lời **Câu hỏi nghiên cứu 1**, trước khi huấn luyện ta phải đo lường hiệu năng của **VietOCR tiền huấn luyện gốc (VGG-Transformer)** trên tập test chưa từng thấy (`test_annotation.txt`).
Các chỉ số đo lường học thuật:
- **Exact Match (%)**: Tỷ lệ mẫu văn bản được nhận dạng đúng 100% không sai một ký tự nào.
- **Character Error Rate (CER %)**: Tỷ lệ lỗi ở cấp độ ký tự (khoảng cách Levenshtein).
- **Word Error Rate (WER %)**: Tỷ lệ lỗi ở cấp độ từ.""")

    # CELL 8: Baseline evaluation code
    add_code("""# 5. Hàm tính toán CER, WER, Exact Match và thực hiện suy luận trên tập Test
import time
import editdistance
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

def compute_metrics(predictions, ground_truths):
    total_chars = 0
    total_words = 0
    char_dist = 0
    word_dist = 0
    exact_matches = 0
    n = len(ground_truths)

    for pred, gt in zip(predictions, ground_truths):
        pred_c = pred.strip()
        gt_c = gt.strip()
        if pred_c == gt_c:
            exact_matches += 1

        char_dist += editdistance.eval(pred_c, gt_c)
        total_chars += len(gt_c)

        pw = pred_c.split()
        gw = gt_c.split()
        word_dist += editdistance.eval(pw, gw)
        total_words += len(gw)

    return {
        "exact_match": round((exact_matches / max(1, n)) * 100, 2),
        "cer": round((char_dist / max(1, total_chars)) * 100, 2),
        "wer": round((word_dist / max(1, total_words)) * 100, 2),
        "exact_count": exact_matches,
        "total": n
    }

def evaluate_on_test_set(predictor, test_ann_path, data_root, limit=None):
    with open(test_ann_path, "r", encoding="utf-8") as f:
        lines = [l.strip().split("\t") for l in f if l.strip() and "\t" in l]

    if limit and limit < len(lines):
        lines = lines[:limit]

    preds = []
    gts = []
    sample_comparisons = []
    
    t0 = time.time()
    for idx, (rel_path, gt) in enumerate(lines, 1):
        full_path = os.path.join(data_root, rel_path)
        if not os.path.exists(full_path):
            continue
        try:
            im = Image.open(full_path).convert("RGB")
            p = predictor.predict(im)
        except Exception:
            p = ""
        preds.append(p)
        gts.append(gt)
        sample_comparisons.append({"path": full_path, "gt": gt, "pred": p})
        if idx % 200 == 0 or idx == len(lines):
            print(f"   [{idx:04d}/{len(lines):04d}] Inference...")

    elapsed = time.time() - t0
    m = compute_metrics(preds, gts)
    m["elapsed_sec"] = round(elapsed, 2)
    m["speed_fps"] = round(len(preds) / max(0.001, elapsed), 1)
    return m, sample_comparisons

# --- ĐÁNH GIÁ BASELINE ---
print("⏳ Đang tải mô hình VietOCR VGG-Transformer tiền huấn luyện...")
base_cfg = Cfg.load_config_from_name('vgg_transformer')
base_cfg['device'] = device
base_cfg['prediction_chk'] = None
baseline_predictor = Predictor(base_cfg)

test_ann_path = os.path.join(DATA_ROOT, "test_annotation.txt")
print(f"🚀 Bắt đầu đánh giá Baseline trên tập Test ({test_ann_path})...")
baseline_metrics, baseline_samples = evaluate_on_test_set(baseline_predictor, test_ann_path, DATA_ROOT)

print()
print("=" * 60)
print("📊 KẾT QUẢ ĐO LƯỜNG BASELINE PRETRAINED VIETOCR:")
print(f"   - Exact Match : {baseline_metrics['exact_match']}% ({baseline_metrics['exact_count']}/{baseline_metrics['total']})")
print(f"   - CER         : {baseline_metrics['cer']}%")
print(f"   - WER         : {baseline_metrics['wer']}%")
print(f"   - Tốc độ      : {baseline_metrics['speed_fps']} mẫu/giây")
print("=" * 60)""")


    # CELL 9: Fine-tuning config markdown
    add_md("""---
## 🚀 BƯỚC 3: THIẾT LẬP SIÊU THAM SỐ VÀ HUẤN LUYỆN FINE-TUNING
Cấu hình tối ưu cho VietOCR Fine-tuning trên GPU Kaggle:
- **Kiến trúc**: `VGG-19` CNN Backbone + Sequence-to-Sequence `Transformer` Decoder.
- **Số vòng lặp (Iterations)**: 12.000 iters (~28 epochs với batch_size=32).
- **Learning Rate**: `max_lr = 0.0001` sử dụng `OneCycleLR` scheduler.
- **Tăng cường ảnh (`image_aug = True`)**: Mô phỏng xoay nhẹ, đổi độ tương phản, mờ scanner.
- **Loss Function**: `LabelSmoothingLoss` (smoothing=0.1) chống quá khớp ký tự hiếm.""")

    # CELL 10: Config code
    add_code("""# 6. Thiết lập Cấu hình VietOCR Trainer
from vietocr.model.trainer import Trainer

# Tải cấu hình chuẩn
config = Cfg.load_config_from_name('vgg_transformer')

# Cấu hình thiết bị & Dataset
config['device'] = device
config['dataset']['name'] = 'hcmute_doc'
config['dataset']['data_root'] = DATA_ROOT
config['dataset']['train_annotation'] = 'train_annotation.txt'
config['dataset']['valid_annotation'] = 'val_annotation.txt'
config['dataset']['image_height'] = 32
config['dataset']['image_min_width'] = 32
config['dataset']['image_max_width'] = 512

# Cấu hình Trainer & Hyperparameters
config['trainer']['batch_size'] = 32
config['trainer']['print_every'] = 100
config['trainer']['valid_every'] = 1000
config['trainer']['iters'] = 12000
config['trainer']['export'] = './weights/vietocr_hcmute_finetuned.pth'
config['trainer']['checkpoint'] = './checkpoint/vietocr_checkpoint.pth'
config['trainer']['log'] = './train_vietocr.log'
config['trainer']['metrics'] = None

# Tối ưu hóa Fine-tuning
config['optimizer']['max_lr'] = 0.0001
config['optimizer']['pct_start'] = 0.1
config['aug']['image_aug'] = True
config['aug']['masked_language_model'] = True
config['predictor']['beamsearch'] = False

os.makedirs("./weights", exist_ok=True)
os.makedirs("./checkpoint", exist_ok=True)

print("⚙️ Cấu hình huấn luyện hoàn tất:")
print(f"   - Iterations : {config['trainer']['iters']}")
print(f"   - Batch Size : {config['trainer']['batch_size']}")
print(f"   - Max LR     : {config['optimizer']['max_lr']}")
print(f"   - Export Wts : {config['trainer']['export']}")""")

    # CELL 11: Execute Training
    add_code("""# 7. Khởi tạo Trainer và Bắt đầu Huấn luyện Fine-Tuning
print("⏳ Khởi tạo VietOCR Trainer với pretrained=True...")
trainer = Trainer(config, pretrained=True)

print()
print("=" * 70)
print("🔥 BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN (VUI LÒNG ĐỢI)...")
print("=" * 70)

start_time = time.time()
trainer.train()
total_train_time = time.time() - start_time

print()
print("=" * 70)
print(f"🎉 HUẤN LUYỆN HOÀN TẤT TRONG {total_train_time / 60:.1f} PHÚT!")
print(f"💾 Trọng số tốt nhất đã được xuất ra: {config['trainer']['export']}")
print("=" * 70)""")

    # CELL 12: Evaluate Fine-tuned model markdown
    add_md("""---
## 📈 BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH SAU KHI FINE-TUNING
Tiến hành nạp trọng số `vietocr_hcmute_finetuned.pth` vừa huấn luyện xong và kiểm thử lại trên **chính xác cùng tập Test 867 mẫu** để có kết quả so sánh công bằng tuyệt đối.""")

    # CELL 13: Evaluate fine-tuned model code
    add_code("""# 8. Đánh giá Mô hình sau Fine-tuning
export_weights_path = config['trainer']['export']

print("⏳ Đang tải trọng số VietOCR Fine-tuned vừa huấn luyện...")
ft_cfg = Cfg.load_config_from_name('vgg_transformer')
ft_cfg['device'] = device
ft_cfg['weights'] = export_weights_path
ft_cfg['prediction_chk'] = None
finetuned_predictor = Predictor(ft_cfg)

print(f"🚀 Bắt đầu đánh giá Fine-tuned trên tập Test ({test_ann_path})...")
finetuned_metrics, finetuned_samples = evaluate_on_test_set(finetuned_predictor, test_ann_path, DATA_ROOT)

print()
print("=" * 60)
print("📊 KẾT QUẢ ĐO LƯỜNG SAU FINE-TUNING:")
print(f"   - Exact Match : {finetuned_metrics['exact_match']}% ({finetuned_metrics['exact_count']}/{finetuned_metrics['total']})")
print(f"   - CER         : {finetuned_metrics['cer']}%")
print(f"   - WER         : {finetuned_metrics['wer']}%")
print(f"   - Tốc độ      : {finetuned_metrics['speed_fps']} mẫu/giây")
print("=" * 60)""")


    # CELL 14: Academic Comparison & Charts markdown
    add_md("""---
## 🏆 BƯỚC 5: TỔNG HỢP KẾT QUẢ & TRẢ LỜI CÂU HỎI NGHIÊN CỨU 1
Bảng đối chiếu định lượng và biểu đồ minh chứng khoa học phục vụ trực tiếp cho **Chương 3 của Báo cáo Tiểu luận Chuyên ngành**.""")

    # CELL 15: Table & Plot code
    add_code("""# 9. Bảng so sánh định lượng & Vẽ biểu đồ cải thiện
import pandas as pd
from tabulate import tabulate

delta_em = finetuned_metrics['exact_match'] - baseline_metrics['exact_match']
delta_cer = finetuned_metrics['cer'] - baseline_metrics['cer']
delta_wer = finetuned_metrics['wer'] - baseline_metrics['wer']

results_table = [
    ["Exact Match (%)", f"{baseline_metrics['exact_match']}%", f"{finetuned_metrics['exact_match']}%", f"+{delta_em:.2f}%" if delta_em >= 0 else f"{delta_em:.2f}%"],
    ["CER (%) [Càng thấp càng tốt]", f"{baseline_metrics['cer']}%", f"{finetuned_metrics['cer']}%", f"{delta_cer:.2f}%"],
    ["WER (%) [Càng thấp càng tốt]", f"{baseline_metrics['wer']}%", f"{finetuned_metrics['wer']}%", f"{delta_wer:.2f}%"],
    ["Tốc độ (samples/sec)", f"{baseline_metrics['speed_fps']}", f"{finetuned_metrics['speed_fps']}", f"{finetuned_metrics['speed_fps'] - baseline_metrics['speed_fps']:+.1f}"]
]

print("=" * 85)
print("BẢNG SO SÁNH ĐỊNH LƯỢNG TRƯỚC VÀ SAU FINE-TUNING (TRẢ LỜI CÂU HỎI NGHIÊN CỨU 1):")
print("=" * 85)
print(tabulate(results_table, headers=["Chỉ số", "Baseline (Pretrained)", "VietOCR (Fine-tuned)", "Cải thiện (Δ)"], tablefmt="grid"))
print("=" * 85)

# Lưu bảng kết quả ra file CSV
df_res = pd.DataFrame(results_table, columns=["Metric", "Baseline", "FineTuned", "Delta"])
df_res.to_csv("vietocr_research_question_1_results.csv", index=False)
print("💾 Đã lưu kết quả đối sánh vào: vietocr_research_question_1_results.csv")

# Vẽ biểu đồ so sánh cột
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Exact Match
ax[0].bar(["Baseline", "Fine-tuned"], [baseline_metrics['exact_match'], finetuned_metrics['exact_match']], color=["#4A90E2", "#50E3C2"])
ax[0].set_title("Exact Match (%) - Càng cao càng tốt", fontsize=12, fontweight="bold")
ax[0].set_ylim(0, 105)
for p in ax[0].patches:
    ax[0].annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2.),
                   ha='center', va='center', fontsize=12, color='white', fontweight='bold')

# Plot 2: CER
ax[1].bar(["Baseline", "Fine-tuned"], [baseline_metrics['cer'], finetuned_metrics['cer']], color=["#FF6B6B", "#48CFAD"])
ax[1].set_title("Character Error Rate (CER %) - Càng thấp càng tốt", fontsize=12, fontweight="bold")
for p in ax[1].patches:
    ax[1].annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.3),
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 3: WER
ax[2].bar(["Baseline", "Fine-tuned"], [baseline_metrics['wer'], finetuned_metrics['wer']], color=["#E17055", "#00B894"])
ax[2].set_title("Word Error Rate (WER %) - Càng thấp càng tốt", fontsize=12, fontweight="bold")
for p in ax[2].patches:
    ax[2].annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.3),
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.suptitle("Đánh giá Định lượng Hiệu năng Fine-Tuning VietOCR trên Dữ liệu Biểu mẫu HCMUTE", fontsize=14, y=1.03)
plt.tight_layout()
plt.savefig("vietocr_comparison_chart.png", dpi=300, bbox_inches="tight")
plt.show()
print("🖼️ Đã lưu biểu đồ vào: vietocr_comparison_chart.png")""")

    # CELL 16: Qualitative Case Studies
    add_code("""# 10. Trực quan hóa các ca nhận dạng được sửa lỗi thành công sau Fine-tuning
corrected_cases = []
for b_s, ft_s in zip(baseline_samples, finetuned_samples):
    gt = b_s['gt']
    pred_base = b_s['pred']
    pred_ft = ft_s['pred']
    # Baseline sai nhưng Fine-tuned đọc đúng hoàn toàn
    if pred_base.strip() != gt.strip() and pred_ft.strip() == gt.strip():
        corrected_cases.append((b_s['path'], gt, pred_base, pred_ft))

print(f"✨ Tìm thấy {len(corrected_cases)} trường hợp Baseline đọc SAI nhưng Fine-tuned đã SỬA ĐÚNG hoàn toàn!")

if corrected_cases:
    num_disp = min(5, len(corrected_cases))
    fig, axes = plt.subplots(num_disp, 1, figsize=(12, 2.5 * num_disp))
    if num_disp == 1:
        axes = [axes]

    for idx in range(num_disp):
        c_path, gt, p_base, p_ft = corrected_cases[idx]
        im = Image.open(c_path)
        axes[idx].imshow(im)
        title_txt = f"Ground Truth: '{gt}'\\n❌ Baseline: '{p_base}'\\n✅ Fine-tuned: '{p_ft}'"
        axes[idx].set_title(title_txt, fontsize=10, loc="left", color="darkgreen")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig("vietocr_corrected_samples.png", dpi=300, bbox_inches="tight")
    plt.show()""")

    # CELL 17: Package Artifacts
    add_code("""# 11. Đóng gói kết quả & Trọng số mô hình để tải về
import zipfile

output_zip = "HCMUTE_VietOCR_Finetuned_Artifacts.zip"
with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    # 1. Trọng số fine-tuned
    if os.path.exists("./weights/vietocr_hcmute_finetuned.pth"):
        zipf.write("./weights/vietocr_hcmute_finetuned.pth", arcname="vietocr_hcmute_finetuned.pth")
    # 2. File kết quả đánh giá
    if os.path.exists("vietocr_research_question_1_results.csv"):
        zipf.write("vietocr_research_question_1_results.csv", arcname="vietocr_research_question_1_results.csv")
    # 3. Biểu đồ đánh giá
    if os.path.exists("vietocr_comparison_chart.png"):
        zipf.write("vietocr_comparison_chart.png", arcname="vietocr_comparison_chart.png")
    if os.path.exists("vietocr_corrected_samples.png"):
        zipf.write("vietocr_corrected_samples.png", arcname="vietocr_corrected_samples.png")
    # 4. Training log
    if os.path.exists("./train_vietocr.log"):
        zipf.write("./train_vietocr.log", arcname="train_vietocr.log")

print("=" * 60)
print(f"🎁 ĐÃ ĐÓNG GÓI THÀNH CÔNG: {output_zip} ({round(os.path.getsize(output_zip)/(1024**2), 2)} MB)")
print("Bạn có thể tải file zip này về từ mục 'Output' bên phải giao diện Kaggle!")
print("=" * 60)""")

    notebook_content = {
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
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=1, ensure_ascii=False)

    print(f"✅ Đã tạo thành công Notebook tại: {output_path}")

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "train_vietocr_finetune_kaggle.ipynb"
    build_notebook(out_file)
