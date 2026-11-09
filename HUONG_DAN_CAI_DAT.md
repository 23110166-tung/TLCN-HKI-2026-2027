# 🚀 HƯỚNG DẪN CÀI ĐẶT & CHẠY DỰ ÁN HCMUTE DOCAI TRÊN MÁY MỚI

Tài liệu này hướng dẫn chi tiết các bước để thành viên trong nhóm (hoặc người kiểm thử) có thể cài đặt và khởi chạy dự án **HCMUTE DocAI** trên một máy tính hoàn toàn mới một cách nhanh chóng và không gặp lỗi.

---

## 📌 1. YÊU CẦU HỆ THỐNG & PHẦN CỨNG

- **Hệ điều hành:** Windows 10/11 (Khuyến nghị 64-bit), Ubuntu Linux hoặc macOS.
- **Python:** Phiên bản **`Python 3.10`** hoặc **`Python 3.11`** *(Lưu ý: Khi cài đặt Python trên Windows, nhớ tích chọn ô `"Add Python to PATH"`)*.
- **Phần cứng:**
  - **CPU:** Mọi máy tính văn phòng phổ thông (Core i3/i5/i7, AMD Ryzen) đều chạy rất mượt nhờ **ONNX Runtime** (độ trễ xử lý ~200ms).
  - **RAM:** Tối thiểu 8GB RAM (khuyến nghị 16GB).
  - **GPU (Không bắt buộc):** Nếu máy có card đồ họa NVIDIA (GTX/RTX) có thể bật `cuda:0` để đạt tốc độ xử lý thời gian thực (93.7 FPS).

---

## ⚡ 2. CÁCH KHỞI CHẠY (CHỌN 1 TRONG 2 CÁCH)

### 🌟 CÁCH 1: Dùng File Tự Động (Khuyên dùng cho thành viên nhóm)
Nếu bạn nhận được toàn bộ thư mục dự án `TLCN/` (đã bao gồm các thư mục trọng số):

1. Mở thư mục dự án `TLCN/`.
2. **Nhấp đúp chuột vào file `run_app.bat`**.
3. Hệ thống sẽ tự động:
   - Tạo môi trường ảo `venv` cô lập (tránh xung đột thư viện với máy).
   - Tự động cài đặt đầy đủ các thư viện từ `requirements.txt`.
   - Khởi động Streamlit và tự động mở trình duyệt web tại địa chỉ: `http://localhost:8501`.

---

### 🛠️ CÁCH 2: Cài Đặt Thủ Công Qua Dòng Lệnh (Command Line)

#### Bước 2.1: Mở Terminal tại thư mục dự án
Mở **PowerShell** hoặc **Command Prompt** (cmd) tại thư mục `TLCN`:
```bash
cd path/to/TLCN
```

#### Bước 2.2: Khởi tạo và kích hoạt môi trường ảo (Khuyến nghị)
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo trên Windows
venv\Scripts\activate

# (Nếu dùng Linux / macOS: source venv/bin/activate)
```

#### Bước 2.3: Cài đặt các thư viện phụ thuộc
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Ghi chú về PyTorch:**
> - Mặc định `requirements.txt` sẽ cài đặt bản PyTorch phù hợp.
> - Nếu máy bạn có card rời NVIDIA và muốn chạy bằng CUDA 12.1:
>   ```bash
>   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
>   ```

#### Bước 2.4: Kiểm tra cấu trúc trọng số mô hình
Do GitHub giới hạn kích thước file tải lên (tối đa 100MB), file trọng số VietOCR (145MB) không thể đẩy trực tiếp lên Git. Hãy đảm bảo thư mục của bạn có đủ các file sau:

```
TLCN/
├── HCMUTE_VietOCR_Finetuned_Artifacts/
│   └── vietocr_hcmute_finetuned.pth          (~145 MB)
├── HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2/
│   ├── yolo11_domain_adapted_best.pt          (~6 MB)
│   ├── yolo11_domain_adapted_best.onnx        (~10 MB)
│   └── yolo_pytorch_vs_onnx_benchmark.csv
├── sample/                                    (Chứa các ảnh test mẫu)
├── pipeline/
├── app.py
├── requirements.txt
└── run_app.bat
```
*(Nếu bạn clone từ GitHub mà thiếu file `vietocr_hcmute_finetuned.pth`, hãy copy file này từ máy của bạn Tùng hoặc tải từ link Google Drive nhóm).*

#### Bước 2.5: Khởi chạy ứng dụng Web
```bash
streamlit run app.py
```
Ứng dụng sẽ khởi chạy tại: `http://localhost:8501`.

---

## 🎯 3. HƯỚNG DẪN TEST CÁC TÍNH NĂNG CHÍNH TRÊN GIAO DIỆN

1. **Chọn mẫu tài liệu kiểm thử:**
   - Ở thanh menu bên trái (Sidebar), mục **Nguồn Dữ liệu Kiểm thử**, chọn **"Chọn mẫu kiểm thử thực tế có sẵn"**.
   - Chọn ảnh chụp smartphone: `REAL_005_IMG_2535.jpg` hoặc `REAL_009_IMG_2539.jpg` (ảnh chụp điện thoại thực tế).
   - Hoặc bạn có thể chuyển sang chế độ **"Tải ảnh từ máy tính (Upload)"** để tải ảnh bất kỳ.
2. **Thử nghiệm chuyển đổi Engine (PyTorch vs ONNX Runtime):**
   - Gạt toggle **⚡ Kích hoạt ONNX Runtime (.onnx)** trên Sidebar.
   - Quan sát thời gian suy luận (`pipeline_latency_ms`) giảm đáng kể so với PyTorch Native.
3. **Thử nghiệm tiền xử lý nắn góc (Auto Doc Scanner):**
   - Bật toggle **✨ Tự động Quét & Nắn phẳng (Auto Doc Scanner)** để quan sát thuật toán nắn phẳng phối cảnh trước khi đưa vào nhận dạng.
4. **Cơ chế Hiệu chỉnh HITL (Human-in-the-Loop):**
   - Nhấn nút **"🚀 BẮT ĐẦU PHÂN TÍCH & THẨM ĐỊNH"**.
   - Xem kết quả trích xuất thực thể, bảng điểm và phán quyết của **Rule Engine**.
   - Khi có ô thông tin cần sửa (ví dụ mã môn, điểm, số điện thoại), chỉnh sửa trực tiếp trên giao diện -> Rule Engine sẽ lập tức tái thẩm định tự động theo thời gian thực.
5. **Xem đối chứng thực nghiệm (Tab 4):**
   - Chuyển sang **Tab 4: 📊 Báo cáo Thẩm định & Đối chứng Thực nghiệm** để xem:
     - Biểu đồ và bảng so sánh mAP@0.5 giữa mô hình Baseline (6.84%) và Thích nghi miền (34.72%).
     - Độ chính xác trích xuất từng trường nghiệp vụ (mã môn 88.1%, điểm 83.3%, ngày thi 82.4%, v.v.).
     - Hiệu quả HITL: Tiết kiệm **78.5% thời gian** (từ 240s xuống 45s) và nâng độ chính xác thẩm định lên **100%**.
     - Bảng đối chứng thông lượng và độ trễ suy luận giữa PyTorch và ONNX.

---

## ❓ 4. XỬ LÝ SỰ CỐ THƯỜNG GẶP (FAQ & TROUBLESHOOTING)

### Q1: Bị lỗi `Port 8501 is already in use`?
- **Nguyên nhân:** Có một phiên bản Streamlit khác đang chạy ngầm trên cổng 8501.
- **Khắc phục:** Mở terminal và chỉ định cổng khác:
  ```bash
  streamlit run app.py --server.port=8502
  ```

### Q2: Bị lỗi `NameError: name 'time' is not defined`?
- **Khắc phục:** Nhấn phím **F5** (hoặc `Ctrl + R`) trên trình duyệt để nạp lại mã nguồn mới nhất.

### Q3: Bị lỗi `AttributeError: module 'numpy' has no attribute ...`?
- **Nguyên nhân:** Phiên bản NumPy 2.x mới phát hành không tương thích ngược với PyTorch 2.x hoặc VietOCR.
- **Khắc phục:** Ép cài đặt NumPy phiên bản 1.x:
  ```bash
  pip install "numpy<2.0.0"
  ```

### Q4: Máy không có card màn hình rời NVIDIA có chạy được không?
- **Hoàn toàn chạy tốt!** Ở mục **Thiết bị** trên thanh Sidebar, chọn **`cpu`** và bật toggle **⚡ Kích hoạt ONNX Runtime (.onnx)**. Mô hình ONNX đã được tối ưu hóa đồ thị tĩnh đặc biệt để chạy cực nhanh trên mọi CPU văn phòng.

