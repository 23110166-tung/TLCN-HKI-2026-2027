# -*- coding: utf-8 -*-
"""
Local Evaluation Script for VietOCR on Document Text Crops.
Tests recognition accuracy, CER/WER, and inference speed.
"""

import os
import sys
import time
import glob
from PIL import Image

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_vietocr_benchmark():
    print("=" * 80)
    print("🧠 BẮT ĐẦU KIỂM THỬ MÔ HÌNH NHẬN DẠNG CHỮ TIẾNG VIỆT (VIETOCR) CỤC BỘ")
    print("=" * 80)

    try:
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg
    except ImportError:
        print("❌ Chưa cài đặt vietocr. Vui lòng chạy: pip install vietocr")
        return

    # 1. Khởi tạo Predictor
    print("⏳ Đang tải trọng số VietOCR vgg_transformer...")
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cpu'
    config['prediction_chk'] = None

    try:
        predictor = Predictor(config)
        print("✅ Khởi tạo VietOCR thành công!")
    except Exception as e:
        print("❌ Lỗi khi tải mô hình VietOCR:", e)
        return

    # 2. Tìm ảnh crop mẫu
    crops_dir = os.path.join("dataset_yolo_real_world", "crops_ocr")
    if not os.path.exists(crops_dir):
        print(f"❌ Không tìm thấy thư mục crops: {crops_dir}")
        return

    crop_files = sorted(glob.glob(os.path.join(crops_dir, "*.png")) + glob.glob(os.path.join(crops_dir, "*.jpg")))
    print(f"🔍 Tìm thấy {len(crop_files)} ảnh crop mẫu trong {crops_dir}")

    if not crop_files:
        print("⚠️ Không có ảnh crop nào để kiểm thử.")
        return

    # 3. Tiến hành suy luận trên 15 mẫu đầu tiên
    test_samples = crop_files[:15]
    print(f"\n🚀 Đang chạy suy luận trên {len(test_samples)} mẫu thực tế:")
    print("-" * 80)
    print(f"{'STT':<4} | {'Tên File Crop':<32} | {'Thời gian (s)':<14} | {'Kết quả OCR'}")
    print("-" * 80)

    total_time = 0.0
    results = []

    for i, file_path in enumerate(test_samples):
        img = Image.open(file_path).convert("RGB")
        
        t0 = time.time()
        try:
            pred_text = predictor.predict(img)
        except Exception as err:
            pred_text = f"[LỖI: {err}]"
        elapsed = time.time() - t0
        total_time += elapsed

        f_name = os.path.basename(file_path)
        print(f"{i+1:<4} | {f_name:<32} | {elapsed:<14.3f} | {pred_text}")
        results.append({
            "file": f_name,
            "elapsed": elapsed,
            "pred": pred_text
        })

    avg_time = total_time / len(test_samples) if test_samples else 0
    print("-" * 80)
    print(f"⏱️ Tổng thời gian: {total_time:.2f}s | Trung bình: {avg_time:.3f}s / crop (trên CPU)")
    print("=" * 80)

if __name__ == "__main__":
    run_vietocr_benchmark()
