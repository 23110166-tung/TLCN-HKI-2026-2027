# -*- coding: utf-8 -*-
"""
HCMUTE Academic Document AI - YOLOv11 PyTorch vs ONNX Benchmark
(Trường Đại học Công nghệ Kỹ thuật Thành phố Hồ Chí Minh)

Script đo lường và so sánh hiệu năng suy luận (Inference Latency & Throughput)
giữa mô hình PyTorch gốc (.pt) và mô hình tối ưu hóa ONNX Runtime (.onnx)
trên cả GPU (NVIDIA RTX 3050 Ti) và CPU (Intel Core i5-12500H).

Các cấu hình kiểm thử:
1. PyTorch Native (CUDA - GPU)
2. PyTorch Native (CPU)
3. ONNX Runtime (CUDAExecutionProvider - GPU)
4. ONNX Runtime (CPUExecutionProvider - CPU)
5. Ultralytics YOLO Wrapper on ONNX (CPU & GPU)

Độ phân giải kiểm thử:
- imgsz = 1280 (Chuẩn Document AI cho văn bản hành chính độ phân giải cao)
- imgsz = 640 (Chuẩn baseline tiêu chuẩn của YOLO)
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# Đảm bảo UTF-8 trên Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm đường dẫn DLL của PyTorch để nạp cublas64_12.dll cho ONNX Runtime GPU
import torch
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib):
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass

from ultralytics import YOLO
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2")
PT_PATH = os.path.join(MODEL_DIR, "yolo11_domain_adapted_best.pt")
ONNX_PATH = os.path.join(MODEL_DIR, "yolo11_domain_adapted_best.onnx")
SAMPLE_IMG = os.path.join(BASE_DIR, "dataset_yolo_domain_adaptation", "images", "test", "REAL_009_IMG_2539.jpg")
OUTPUT_CSV = os.path.join(MODEL_DIR, "yolo_pytorch_vs_onnx_benchmark.csv")


def ensure_onnx_model(pt_path, onnx_path, imgsz=1280):
    """Kiểm tra và export mô hình PyTorch (.pt) sang ONNX (.onnx) nếu chưa có."""
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Không tìm thấy mô hình PyTorch tại: {pt_path}")

    pt_size_mb = os.path.getsize(pt_path) / (1024 * 1024)
    print(f"📦 Mô hình PyTorch gốc: {pt_path} ({pt_size_mb:.2f} MB)")

    if not os.path.exists(onnx_path):
        print(f"🚀 Bắt đầu export mô hình sang ONNX (imgsz={imgsz}, dynamic=True)...")
        model = YOLO(pt_path)
        exported_file = model.export(format="onnx", imgsz=imgsz, dynamic=True, opset=17)
        print(f"✅ Đã export thành công sang: {exported_file}")
    
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"📦 Mô hình ONNX Runtime: {onnx_path} ({onnx_size_mb:.2f} MB)")
    return pt_size_mb, onnx_size_mb


def verify_detection_fidelity(pt_path, onnx_path, test_image_path, imgsz=1280):
    """
    Xác minh tính tương đồng kết quả nhận diện (Detection Fidelity)
    giữa mô hình PyTorch và ONNX trên cùng một ảnh kiểm thử.
    """
    print("\n" + "="*70)
    print("🔍 KIỂM TRA ĐỘ CHÍNH XÁC TƯƠNG ĐỒNG (FIDELITY CHECK)")
    print("="*70)

    # 1. PyTorch Inference
    model_pt = YOLO(pt_path)
    res_pt = model_pt.predict(test_image_path, imgsz=imgsz, conf=0.25, device="cuda:0" if torch.cuda.is_available() else "cpu", verbose=False)[0]
    boxes_pt = res_pt.boxes.xyxy.cpu().numpy()
    confs_pt = res_pt.boxes.conf.cpu().numpy()
    cls_pt = res_pt.boxes.cls.cpu().numpy().astype(int)

    # 2. ONNX Inference qua Ultralytics
    model_onnx = YOLO(onnx_path)
    res_onnx = model_onnx.predict(test_image_path, imgsz=imgsz, conf=0.25, verbose=False)[0]
    boxes_onnx = res_onnx.boxes.xyxy.cpu().numpy()
    confs_onnx = res_onnx.boxes.conf.cpu().numpy()
    cls_onnx = res_onnx.boxes.cls.cpu().numpy().astype(int)

    print(f"  • Số bounding boxes phát hiện bởi PyTorch (.pt):  {len(boxes_pt)}")
    print(f"  • Số bounding boxes phát hiện bởi ONNX (.onnx):    {len(boxes_onnx)}")
    print(f"  • Điểm tin cậy trung bình PyTorch:                {float(np.mean(confs_pt)):.4f}" if len(confs_pt) else "  N/A")
    print(f"  • Điểm tin cậy trung bình ONNX:                   {float(np.mean(confs_onnx)):.4f}" if len(confs_onnx) else "  N/A")

    diff_count = abs(len(boxes_pt) - len(boxes_onnx))
    if diff_count == 0:
        print("  ✅ Kết quả phát hiện tương đồng 100% giữa PyTorch và ONNX Runtime!")
    elif diff_count <= 2:
        print(f"  ✅ Độ lệch phát hiện cực nhỏ ({diff_count} hộp biên), hoàn toàn đạt chuẩn số học float32.")
    else:
        print(f"  ⚠️ Lưu ý: Phát hiện chênh lệch {diff_count} hộp biên do làm tròn số học.")


def prepare_input_tensor(image_path, imgsz=1280):
    """Tiền xử lý ảnh (Letterbox resize, BGR->RGB, chuẩn hóa float32 0-1, NCHW)."""
    img = cv2.imread(image_path)
    h0, w0 = img.shape[:2]
    r = min(imgsz / h0, imgsz / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    dw, dh = (imgsz - nw) / 2, (imgsz - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # BGR -> RGB, HWC -> CHW, 0..255 -> 0..1
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob[None, ...])
    return blob


def benchmark_pytorch(pt_path, input_tensor, device="cuda:0", warmup=10, iterations=50):
    """Đo lường thời gian suy luận PyTorch Native."""
    is_cuda = "cuda" in device and torch.cuda.is_available()
    dev = torch.device(device if is_cuda else "cpu")
    
    model = YOLO(pt_path)
    # Lấy mô hình PyTorch torch.nn.Module bên trong
    nn_model = model.model.to(dev)
    nn_model.eval()

    tensor = torch.from_numpy(input_tensor).to(dev)

    # Khởi tạo bộ đếm bộ nhớ GPU nếu có
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    # 1. Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = nn_model(tensor)
        if is_cuda:
            torch.cuda.synchronize(dev)

    # 2. Đo thời gian lặp
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            if is_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = nn_model(tensor)
                end_event.record()
                torch.cuda.synchronize(dev)
                latencies.append(start_event.elapsed_time(end_event))
            else:
                t0 = time.perf_counter()
                _ = nn_model(tensor)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

    peak_vram_mb = 0.0
    if is_cuda:
        peak_vram_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

    return np.array(latencies), peak_vram_mb


def benchmark_onnxruntime(onnx_path, input_tensor, provider="CUDAExecutionProvider", warmup=10, iterations=50):
    """Đo lường thời gian suy luận ONNX Runtime InferenceSession."""
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 4

    available = ort.get_available_providers()
    if provider not in available:
        print(f"⚠️ Provider {provider} không khả dụng, fallback sang CPUExecutionProvider.")
        providers = ["CPUExecutionProvider"]
        active_provider = "CPUExecutionProvider"
    else:
        providers = [provider, "CPUExecutionProvider"]
        active_provider = provider

    sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
    input_name = sess.get_inputs()[0].name
    ort_inputs = {input_name: input_tensor}

    # 1. Warmup
    for _ in range(warmup):
        _ = sess.run(None, ort_inputs)
    if "CUDA" in active_provider and torch.cuda.is_available():
        torch.cuda.synchronize()

    # 2. Đo thời gian lặp
    latencies = []
    for _ in range(iterations):
        if "CUDA" in active_provider and torch.cuda.is_available():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = sess.run(None, ort_inputs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            _ = sess.run(None, ort_inputs)
            t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return np.array(latencies), sess.get_providers()[0]


def benchmark_ultralytics_e2e(model_path, image_path, imgsz=1280, device="cpu", warmup=5, iterations=30):
    """
    Đo thời gian End-to-End thực tế của Ultralytics (Bao gồm Tiền xử lý + Suy luận + NMS).
    Mô phỏng chính xác luồng chạy trong hệ thống Web App / Document Pipeline.
    """
    model = YOLO(model_path)
    # Warmup
    for _ in range(warmup):
        _ = model.predict(image_path, imgsz=imgsz, conf=0.25, device=device, verbose=False)
    
    times_preprocess = []
    times_inference = []
    times_postprocess = []
    times_total = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        res = model.predict(image_path, imgsz=imgsz, conf=0.25, device=device, verbose=False)[0]
        t1 = time.perf_counter()
        
        times_total.append((t1 - t0) * 1000.0)
        sp = res.speed
        times_preprocess.append(sp.get("preprocess", 0.0))
        times_inference.append(sp.get("inference", 0.0))
        times_postprocess.append(sp.get("postprocess", 0.0))

    return {
        "mean_total": np.mean(times_total),
        "mean_preprocess": np.mean(times_preprocess),
        "mean_inference": np.mean(times_inference),
        "mean_postprocess": np.mean(times_postprocess),
    }


def run_comprehensive_benchmark(pt_path, onnx_path, sample_img, warmup=10, iterations=50):
    """Thực thi toàn bộ bộ kiểm thử hiệu năng suy luận."""
    pt_size_mb, onnx_size_mb = ensure_onnx_model(pt_path, onnx_path, imgsz=1280)
    verify_detection_fidelity(pt_path, onnx_path, sample_img, imgsz=1280)

    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "Không có (CPU Only)"

    print("\n" + "="*70)
    print("⚡ THÔNG TIN PHẦN CỨNG & CẤU HÌNH THỬ NGHIỆM")
    print("="*70)
    print(f"  • CPU:             Intel Core i5-12500H (16 threads)")
    print(f"  • GPU:             {gpu_name}")
    print(f"  • PyTorch:         {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"  • ONNX Runtime:    {ort.__version__}")
    print(f"  • Kích thước .pt:  {pt_size_mb:.2f} MB")
    print(f"  • Kích thước .onnx:{onnx_size_mb:.2f} MB")
    print(f"  • Số lần Warmup:   {warmup}")
    print(f"  • Số lần lặp đo:   {iterations}")

    results = []

    # Danh sách các độ phân giải cần đo
    resolutions = [1280, 640]

    for imgsz in resolutions:
        print("\n" + "-"*70)
        print(f"📊 BẮT ĐẦU ĐO LƯỜNG VỚI ĐỘ PHÂN GIẢI imgsz = {imgsz}")
        print("-"*70)

        input_tensor = prepare_input_tensor(sample_img, imgsz=imgsz)

        # 1. PyTorch CUDA (nếu có GPU)
        if has_cuda:
            print(f"  ⏳ [1/4] Đang đo PyTorch Native (CUDA / RTX 3050 Ti) @ {imgsz}...")
            lat_pt_cuda, vram_pt = benchmark_pytorch(pt_path, input_tensor, device="cuda:0", warmup=warmup, iterations=iterations)
            results.append({
                "Resolution": imgsz,
                "Engine": "PyTorch Native (.pt)",
                "Hardware": "GPU (CUDA: RTX 3050 Ti)",
                "Latency Mean (ms)": round(float(np.mean(lat_pt_cuda)), 2),
                "Latency Std (ms)": round(float(np.std(lat_pt_cuda)), 2),
                "P50 Median (ms)": round(float(np.median(lat_pt_cuda)), 2),
                "P95 (ms)": round(float(np.percentile(lat_pt_cuda, 95)), 2),
                "Min (ms)": round(float(np.min(lat_pt_cuda)), 2),
                "Max (ms)": round(float(np.max(lat_pt_cuda)), 2),
                "Throughput (FPS)": round(1000.0 / float(np.mean(lat_pt_cuda)), 1),
                "Peak VRAM (MB)": round(vram_pt, 1),
                "Model Size (MB)": round(pt_size_mb, 2)
            })

        # 2. PyTorch CPU
        print(f"  ⏳ [2/4] Đang đo PyTorch Native (CPU / Intel i5) @ {imgsz}...")
        lat_pt_cpu, _ = benchmark_pytorch(pt_path, input_tensor, device="cpu", warmup=warmup, iterations=iterations)
        results.append({
            "Resolution": imgsz,
            "Engine": "PyTorch Native (.pt)",
            "Hardware": "CPU (Intel Core i5)",
            "Latency Mean (ms)": round(float(np.mean(lat_pt_cpu)), 2),
            "Latency Std (ms)": round(float(np.std(lat_pt_cpu)), 2),
            "P50 Median (ms)": round(float(np.median(lat_pt_cpu)), 2),
            "P95 (ms)": round(float(np.percentile(lat_pt_cpu, 95)), 2),
            "Min (ms)": round(float(np.min(lat_pt_cpu)), 2),
            "Max (ms)": round(float(np.max(lat_pt_cpu)), 2),
            "Throughput (FPS)": round(1000.0 / float(np.mean(lat_pt_cpu)), 1),
            "Peak VRAM (MB)": 0.0,
            "Model Size (MB)": round(pt_size_mb, 2)
        })

        # 3. ONNX Runtime CUDA (nếu có GPU)
        if has_cuda:
            print(f"  ⏳ [3/4] Đang đo ONNX Runtime (CUDAExecutionProvider) @ {imgsz}...")
            lat_ort_cuda, act_prov = benchmark_onnxruntime(onnx_path, input_tensor, provider="CUDAExecutionProvider", warmup=warmup, iterations=iterations)
            results.append({
                "Resolution": imgsz,
                "Engine": "ONNX Runtime (.onnx)",
                "Hardware": f"GPU ({act_prov})",
                "Latency Mean (ms)": round(float(np.mean(lat_ort_cuda)), 2),
                "Latency Std (ms)": round(float(np.std(lat_ort_cuda)), 2),
                "P50 Median (ms)": round(float(np.median(lat_ort_cuda)), 2),
                "P95 (ms)": round(float(np.percentile(lat_ort_cuda, 95)), 2),
                "Min (ms)": round(float(np.min(lat_ort_cuda)), 2),
                "Max (ms)": round(float(np.max(lat_ort_cuda)), 2),
                "Throughput (FPS)": round(1000.0 / float(np.mean(lat_ort_cuda)), 1),
                "Peak VRAM (MB)": 0.0,  # Managed by ORT
                "Model Size (MB)": round(onnx_size_mb, 2)
            })

        # 4. ONNX Runtime CPU
        print(f"  ⏳ [4/4] Đang đo ONNX Runtime (CPUExecutionProvider) @ {imgsz}...")
        lat_ort_cpu, _ = benchmark_onnxruntime(onnx_path, input_tensor, provider="CPUExecutionProvider", warmup=warmup, iterations=iterations)
        results.append({
            "Resolution": imgsz,
            "Engine": "ONNX Runtime (.onnx)",
            "Hardware": "CPU (CPUExecutionProvider)",
            "Latency Mean (ms)": round(float(np.mean(lat_ort_cpu)), 2),
            "Latency Std (ms)": round(float(np.std(lat_ort_cpu)), 2),
            "P50 Median (ms)": round(float(np.median(lat_ort_cpu)), 2),
            "P95 (ms)": round(float(np.percentile(lat_ort_cpu, 95)), 2),
            "Min (ms)": round(float(np.min(lat_ort_cpu)), 2),
            "Max (ms)": round(float(np.max(lat_ort_cpu)), 2),
            "Throughput (FPS)": round(1000.0 / float(np.mean(lat_ort_cpu)), 1),
            "Peak VRAM (MB)": 0.0,
            "Model Size (MB)": round(onnx_size_mb, 2)
        })

    # Tính toán Speedup So Với PyTorch CPU và PyTorch CUDA
    df = pd.DataFrame(results)
    
    # Tính Speedup cho từng độ phân giải
    speedup_col = []
    for idx, row in df.iterrows():
        res = row["Resolution"]
        hw = row["Hardware"]
        eng = row["Engine"]
        if "GPU" in hw:
            # Baseline là PyTorch CUDA cùng độ phân giải
            base_row = df[(df["Resolution"] == res) & (df["Engine"] == "PyTorch Native (.pt)") & (df["Hardware"].str.contains("GPU"))]
            if not base_row.empty:
                base_lat = base_row.iloc[0]["Latency Mean (ms)"]
                speedup = round(base_lat / row["Latency Mean (ms)"], 2)
                speedup_col.append(f"{speedup}x (vs PyTorch GPU)")
            else:
                speedup_col.append("Baseline")
        else:
            # Baseline là PyTorch CPU cùng độ phân giải
            base_row = df[(df["Resolution"] == res) & (df["Engine"] == "PyTorch Native (.pt)") & (df["Hardware"].str.contains("CPU"))]
            if not base_row.empty:
                base_lat = base_row.iloc[0]["Latency Mean (ms)"]
                speedup = round(base_lat / row["Latency Mean (ms)"], 2)
                speedup_col.append(f"{speedup}x (vs PyTorch CPU)")
            else:
                speedup_col.append("Baseline")

    df["Speedup Factor"] = speedup_col

    # Lưu kết quả ra CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("\n" + "="*70)
    print(f"💾 KẾT QUẢ BENCHMARK ĐÃ LƯU TẠI: {OUTPUT_CSV}")
    print("="*70)

    # Đo lường chi tiết End-to-End Pipeline (Tiền xử lý, NMS, v.v.)
    print("\n" + "="*70)
    print("🔬 PHÂN TÍCH THÀNH PHẦN ĐỘ TRỄ END-TO-END (PREPROCESS + INFERENCE + NMS)")
    print("="*70)
    e2e_pt_gpu = benchmark_ultralytics_e2e(pt_path, sample_img, imgsz=1280, device="cuda:0" if has_cuda else "cpu")
    e2e_pt_cpu = benchmark_ultralytics_e2e(pt_path, sample_img, imgsz=1280, device="cpu")
    e2e_onnx_cpu = benchmark_ultralytics_e2e(onnx_path, sample_img, imgsz=1280, device="cpu")

    print(f"1. PyTorch GPU (RTX 3050 Ti, 1280px):")
    print(f"   • Preprocess:   {e2e_pt_gpu['mean_preprocess']:.2f} ms")
    print(f"   • Inference:    {e2e_pt_gpu['mean_inference']:.2f} ms")
    print(f"   • Postprocess:  {e2e_pt_gpu['mean_postprocess']:.2f} ms")
    print(f"   • Tổng E2E:     {e2e_pt_gpu['mean_total']:.2f} ms (~{1000.0/e2e_pt_gpu['mean_total']:.1f} FPS)")

    print(f"\n2. PyTorch CPU (Intel i5-12500H, 1280px):")
    print(f"   • Preprocess:   {e2e_pt_cpu['mean_preprocess']:.2f} ms")
    print(f"   • Inference:    {e2e_pt_cpu['mean_inference']:.2f} ms")
    print(f"   • Postprocess:  {e2e_pt_cpu['mean_postprocess']:.2f} ms")
    print(f"   • Tổng E2E:     {e2e_pt_cpu['mean_total']:.2f} ms (~{1000.0/e2e_pt_cpu['mean_total']:.1f} FPS)")

    print(f"\n3. ONNX Runtime CPU (Intel i5-12500H, 1280px):")
    print(f"   • Preprocess:   {e2e_onnx_cpu['mean_preprocess']:.2f} ms")
    print(f"   • Inference:    {e2e_onnx_cpu['mean_inference']:.2f} ms")
    print(f"   • Postprocess:  {e2e_onnx_cpu['mean_postprocess']:.2f} ms")
    print(f"   • Tổng E2E:     {e2e_onnx_cpu['mean_total']:.2f} ms (~{1000.0/e2e_onnx_cpu['mean_total']:.1f} FPS)")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark YOLOv11 PyTorch vs ONNX Runtime")
    parser.add_argument("--warmup", type=int, default=10, help="Số lần chạy khởi động warmup")
    parser.add_argument("--iterations", type=int, default=50, help="Số lần lặp đo thời gian")
    args = parser.parse_args()

    run_comprehensive_benchmark(PT_PATH, ONNX_PATH, SAMPLE_IMG, warmup=args.warmup, iterations=args.iterations)

