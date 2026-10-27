# -*- coding: utf-8 -*-
"""
Fine-Tuning & Evaluation Pipeline for YOLOv11 Domain Adaptation on Camera Documents.
Tiểu luận chuyên ngành AI - Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh.

Giải quyết định lượng Câu hỏi nghiên cứu 2 (RQ2):
- Đo lường hiện tượng Domain Shift khi mô hình huấn luyện trên dữ liệu tổng hợp (Synthetic)
  chuyển sang nhận dạng ảnh chụp camera thực tế (Real-world).
- Thực hiện Fine-tuning (Transfer Learning) mô hình YOLOv11n với siêu tham số tối ưu.
- Đánh giá so sánh trực tiếp Baseline (Zero-shot) vs Fine-tuned (Domain-Adapted)
  trên cùng tập Test thực tế để xác minh mức tăng trưởng mAP@0.5 (mục tiêu: từ 6.89% lên > 50%).
- Xuất bảng kết quả CSV và biểu đồ trực quan phục vụ báo cáo nghiệm thu.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Vui lòng cài đặt PyTorch và Ultralytics: pip install torch torchvision ultralytics")

from pipeline.ontology import CLASS_NAMES

DEFAULT_DATA_YAML = os.path.join(BASE_DIR, "dataset_yolo_domain_adaptation", "data.yaml")
DEFAULT_PRETRAINED = os.path.join(
    BASE_DIR, "HCMUTE_DocAI_YOLOv11_VietOCR_Artifacts", "runs", "detect",
    "hcmute_yolo11_layout", "weights", "best.pt"
)
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts")


# =====================================================================
# 1. HÀM ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST THỰC TẾ
# =====================================================================

def evaluate_on_test_split(model, data_yaml, imgsz=1024, split="test", device=0):
    """
    Đánh giá độ chính xác (P, R, mAP@0.5, mAP@0.5:0.95) trên tập Test của data_yaml.
    """
    print(f"\n🔍 Đang đánh giá mô hình trên tập '{split}' ({data_yaml}) ...")
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        split=split,
        device=device,
        verbose=False,
        save_json=False,
        plots=False
    )

    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    # Lấy mAP@0.5 chi tiết từng lớp
    per_class_map50 = {}
    if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
        for idx, cls_name in enumerate(CLASS_NAMES):
            if idx < len(metrics.box.maps):
                per_class_map50[cls_name] = round(float(metrics.box.maps[idx]) * 100.0, 2)

    return {
        "precision": round(mp * 100.0, 2),
        "recall": round(mr * 100.0, 2),
        "map50": round(map50 * 100.0, 2),
        "map50_95": round(map50_95 * 100.0, 2),
        "per_class_map50": per_class_map50,
        "raw_metrics": metrics
    }


# =====================================================================
# 2. VẼ BIỂU ĐỒ TRỰC QUAN HÓA SO SÁNH BASELINE VS FINE-TUNED
# =====================================================================

def plot_domain_adaptation_comparison(res_base, res_ft, out_chart_path):
    """
    Vẽ biểu đồ cột so sánh định lượng trả lời Câu hỏi nghiên cứu 2.
    """
    metrics_names = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
    base_vals = [res_base["precision"], res_base["recall"], res_base["map50"], res_base["map50_95"]]
    ft_vals = [res_ft["precision"], res_ft["recall"], res_ft["map50"], res_ft["map50_95"]]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    rects1 = ax.bar(x - width / 2, base_vals, width, label="Baseline (Synthetic Pretrained Zero-shot)", color="#dc3545", alpha=0.85)
    rects2 = ax.bar(x + width / 2, ft_vals, width, label="Fine-tuned (Domain-Adapted with Augmentations)", color="#28a745", alpha=0.9)

    ax.set_ylabel("Tỉ lệ (%)", fontsize=12, fontweight="bold")
    ax.set_title("Đánh giá Định lượng Hiệu năng Giải quyết Domain Shift trên Ảnh chụp Thực tế (YOLOv11)\n"
                 "Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(max(ft_vals) * 1.25, 70.0))
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Gắn nhãn số liệu lên đầu từng cột
    def autolabel(rects, is_diff=False, partner_rects=None):
        for idx, rect in enumerate(rects):
            h = rect.get_height()
            txt = f"{h:.1f}%"
            if is_diff and partner_rects:
                diff = h - partner_rects[idx].get_height()
                diff_str = f" (+{diff:.1f}%)" if diff > 0 else f" ({diff:.1f}%)"
                txt += diff_str
            ax.annotate(txt,
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=10, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2, is_diff=True, partner_rects=rects1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_chart_path), exist_ok=True)
    plt.savefig(out_chart_path, dpi=200)
    plt.close()
    print(f"📊 Đã lưu biểu đồ so sánh: {out_chart_path}")


# =====================================================================
# 3. QUY TRÌNH HUẤN LUYỆN & THỰC NGHIỆM CHÍNH
# =====================================================================

def run_domain_adaptation_pipeline(
    data_yaml=DEFAULT_DATA_YAML,
    pretrained_weights=DEFAULT_PRETRAINED,
    epochs=40,
    imgsz=1024,
    batch=8,
    lr0=0.001,
    device=None,
    output_dir=DEFAULT_OUTPUT_DIR
):
    print("=" * 80)
    print("🔬 BẮT ĐẦU THỰC NGHIỆM DOMAIN ADAPTATION YOLOV11 (CÂU HỎI NGHIÊN CỨU 2)")
    print("   Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh")
    print("=" * 80)

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    print(f"💻 Thiết bị sử dụng: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {data_yaml}. Hãy chạy prepare_domain_adaptation_dataset.py trước!")

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # BƯỚC 1: Đánh giá mô hình Baseline (Synthetic Model) trên tập Test thực tế
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 BƯỚC 1: ĐÁNH GIÁ MÔ HÌNH BASELINE (ZERO-SHOT TRÊN ẢNH CAMERA THỰC TẾ)")
    print("=" * 60)

    if not os.path.exists(pretrained_weights):
        print(f"⚠️ Không tìm thấy trọng số fine-tuned tại {pretrained_weights}, nạp yolo11n.pt mặc định...")
        model_baseline = YOLO("yolo11n.pt")
    else:
        print(f"📦 Đang nạp mô hình Synthetic Pretrained từ: {pretrained_weights}")
        model_baseline = YOLO(pretrained_weights)

    res_baseline = evaluate_on_test_split(
        model=model_baseline,
        data_yaml=data_yaml,
        imgsz=imgsz,
        split="test",
        device=device
    )

    print("\n📈 KẾT QUẢ BASELINE (ZERO-SHOT TRÊN TẬP TEST CAMERA):")
    print(f"   • Precision   : {res_baseline['precision']}%")
    print(f"   • Recall      : {res_baseline['recall']}%")
    print(f"   • mAP@0.5     : {res_baseline['map50']}%")
    print(f"   • mAP@0.5:0.95: {res_baseline['map50_95']}%")

    # -------------------------------------------------------------
    # BƯỚC 2: Huấn luyện Fine-Tuning mô hình với Real-World + Augmentations
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🚀 BƯỚC 2: TIẾN HÀNH FINE-TUNING YOLOV11N TRÊN DỮ LIỆU DOMAIN ADAPTATION")
    print("=" * 60)

    # Khởi tạo mô hình fine-tuning từ cùng trọng số ban đầu
    model_ft = YOLO(pretrained_weights if os.path.exists(pretrained_weights) else "yolo11n.pt")

    train_project = os.path.join(output_dir, "runs")
    train_name = "yolo11_domain_adapted"

    # Cấu hình huấn luyện tối ưu cho Fine-Tuning Domain Adaptation:
    # - Học suất nhỏ hơn (lr0=0.001, lrf=0.01) để bảo toàn tri thức biểu mẫu đã học
    # - Augmentations tích hợp của YOLO: degrees, translate, scale, shear, perspective, hsv
    train_results = model_ft.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=10,
        save=True,
        device=device,
        project=train_project,
        name=train_name,
        exist_ok=True,
        plots=True,
        optimizer="AdamW",
        lr0=lr0,
        lrf=0.01,
        weight_decay=0.0005,
        degrees=5.0,
        translate=0.05,
        scale=0.1,
        shear=2.0,
        perspective=0.0005,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        mosaic=0.5,
        verbose=True
    )

    best_ft_weights = os.path.join(train_project, train_name, "weights", "best.pt")
    print(f"\n✅ Đã hoàn tất huấn luyện! Trọng số tối ưu nhất được lưu tại: {best_ft_weights}")

    # Copy trọng số tốt nhất ra thư mục Artifacts chính
    artifact_weights = os.path.join(output_dir, "yolo11_domain_adapted_best.pt")
    if os.path.exists(best_ft_weights):
        import shutil
        shutil.copy2(best_ft_weights, artifact_weights)

    # -------------------------------------------------------------
    # BƯỚC 3: Đánh giá mô hình Fine-Tuned trên chính xác cùng tập Test
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🔬 BƯỚC 3: ĐÁNH GIÁ MÔ HÌNH FINE-TUNED TRÊN CÙNG TẬP TEST CAMERA THỰC TẾ")
    print("=" * 60)

    eval_model_ft = YOLO(best_ft_weights if os.path.exists(best_ft_weights) else artifact_weights)
    res_ft = evaluate_on_test_split(
        model=eval_model_ft,
        data_yaml=data_yaml,
        imgsz=imgsz,
        split="test",
        device=device
    )

    # -------------------------------------------------------------
    # BƯỚC 4: Tổng hợp So sánh Khoa học trả lời Câu hỏi Nghiên cứu 2
    # -------------------------------------------------------------
    diff_p = res_ft["precision"] - res_baseline["precision"]
    diff_r = res_ft["recall"] - res_baseline["recall"]
    diff_map50 = res_ft["map50"] - res_baseline["map50"]
    diff_map95 = res_ft["map50_95"] - res_baseline["map50_95"]

    summary_rows = [
        {
            "Chỉ số": "Precision (Độ chính xác)",
            "Baseline (Zero-shot)": f"{res_baseline['precision']:.2f}%",
            "Fine-tuned (Domain-Adapted)": f"{res_ft['precision']:.2f}%",
            "Chênh lệch (Δ)": f"{'+' if diff_p >= 0 else ''}{diff_p:.2f}%"
        },
        {
            "Chỉ số": "Recall (Độ bao phủ)",
            "Baseline (Zero-shot)": f"{res_baseline['recall']:.2f}%",
            "Fine-tuned (Domain-Adapted)": f"{res_ft['recall']:.2f}%",
            "Chênh lệch (Δ)": f"{'+' if diff_r >= 0 else ''}{diff_r:.2f}%"
        },
        {
            "Chỉ số": "mAP@0.5 (Độ chuẩn nhận diện)",
            "Baseline (Zero-shot)": f"{res_baseline['map50']:.2f}%",
            "Fine-tuned (Domain-Adapted)": f"{res_ft['map50']:.2f}%",
            "Chênh lệch (Δ)": f"{'+' if diff_map50 >= 0 else ''}{diff_map50:.2f}%"
        },
        {
            "Chỉ số": "mAP@0.5:0.95 (Chặt chẽ IoU)",
            "Baseline (Zero-shot)": f"{res_baseline['map50_95']:.2f}%",
            "Fine-tuned (Domain-Adapted)": f"{res_ft['map50_95']:.2f}%",
            "Chênh lệch (Δ)": f"{'+' if diff_map95 >= 0 else ''}{diff_map95:.2f}%"
        }
    ]

    df_summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_dir, "domain_adaptation_rq2_results.csv")
    df_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("🏆 BẢNG TỔNG HỢP KẾT QUẢ ĐỐI CHỨNG CÂU HỎI NGHIÊN CỨU 2 (RQ2):")
    print(f"   (Đánh giá độc lập trên tập Test gồm các ảnh chụp camera thực tế chưa từng thấy)")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)
    print(f"💾 File bảng kết quả CSV: {csv_path}")

    # Vẽ biểu đồ so sánh
    chart_path = os.path.join(output_dir, "domain_adaptation_comparison_chart.png")
    plot_domain_adaptation_comparison(res_baseline, res_ft, chart_path)

    return df_summary, res_baseline, res_ft


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Domain Adaptation Training & Evaluation for HCMUTE Academic Forms")
    parser.add_argument("--data", default=DEFAULT_DATA_YAML, help="Đường dẫn file data.yaml")
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED, help="Trọng số khởi tạo (best.pt synthetic)")
    parser.add_argument("--epochs", type=int, default=40, help="Số epochs fine-tuning")
    parser.add_argument("--batch", type=int, default=8, help="Kích thước batch (khuyên dùng 8 cho 4GB-8GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Kích thước ảnh đầu vào YOLO")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate ban đầu")
    parser.add_argument("--device", default=None, help="Device (0 hoặc cpu)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Thư mục xuất artifacts")
    args = parser.parse_args()

    run_domain_adaptation_pipeline(
        data_yaml=args.data,
        pretrained_weights=args.pretrained,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        device=args.device,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
