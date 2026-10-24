# HCMUTE DocAI – YOLOv11 Layout Detection & VietOCR

> **Tiểu luận chuyên ngành AI** – Học kỳ 1, Năm học 2026-2027  
> **Trường ĐH Công nghệ Kỹ thuật TP.HCM (HCMUTE)**

## Đề tài

**Xây dựng hệ thống nhận dạng, trích xuất và kiểm tra tính hợp lệ của thông tin từ đơn học vụ và bảng điểm sinh viên bằng học sâu**

---

## Nhóm thực hiện

| MSSV | Họ và Tên | Vai trò |
|---|---|---|
| 23110166 | Nguyễn Bách Tùng | Nhóm trưởng |
| 23110035 | Nguyễn Hưng Nguyên | Thành viên |
| 23110122 | Trần Cẩm Long | Thành viên |

**GVHD**: TS. Bùi Mạnh Quân

---

## Kiến trúc Pipeline

```
Ảnh tài liệu (scan / camera)
        ↓
Perspective Dewarping (nắn thẳng phối cảnh)
        ↓
YOLOv11n Layout Detection (18 lớp học vụ HCMUTE)
        ↓
Crop vùng Bounding Box theo lớp
        ↓
VietOCR (VGG-Transformer) → Trích xuất text tiếng Việt
        ↓
Rule Engine → Kiểm tra tính hợp lệ nghiệp vụ
        ↓
Xuất JSON / Excel / Word / PDF
```

---

## Kết quả Giai đoạn 2 (Tháng 09/2026)

| Mô hình | Tập đánh giá | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| YOLOv8n (Baseline) | Synthetic Test (45 ảnh) | 94.37% | 94.29% | 95.52% | 86.97% |
| **YOLOv11n (Đề xuất)** | Synthetic Test (45 ảnh) | **97.98%** | 91.46% | **95.37%** | 86.10% |
| YOLOv11n Zero-shot | Real-World Camera (120 ảnh) | 23.27% | 12.91% | 6.89% | 1.66% |

**VietOCR Baseline**: 501 trường trích xuất, confidence 90–98% trên ảnh synthetic.

---

## Cấu trúc thư mục

```
TLCN/
├── pipeline/                   # Source code chính
│   ├── mock_data.py            # Sinh dữ liệu sinh viên ngẫu nhiên (Synthetic)
│   ├── ontology.py             # Định nghĩa 18 lớp học vụ HCMUTE
│   ├── generate_crop_ocr_dataset.py    # Trích xuất 15.507 crop OCR và nhãn chuẩn VietOCR
│   ├── train_vietocr_finetune.py       # Script fine-tuning VietOCR & đo CER/WER
│   ├── generate_vietocr_finetune_notebook.py  # Script sinh notebook Kaggle VietOCR
│   ├── process_real_camera_dataset.py  # Perspective Dewarping + gán nhãn ảnh thực
│   ├── inference_engine.py     # Pipeline YOLOv11 + VietOCR end-to-end
│   ├── pdf_converter.py        # Chuyển đổi DOCX → PDF
│   └── ...
├── sample/                     # File mẫu DOCX biểu mẫu gốc (10 loại)
├── train_yolov11_vietocr_kaggle.ipynb     # Notebook YOLOv11 Layout Detection trên Kaggle
├── train_vietocr_finetune_kaggle.ipynb    # Notebook Fine-tuning VietOCR trên Kaggle GPU
├── Bao_Cao_Tien_Do_*.docx      # Báo cáo tiến độ các giai đoạn
└── .gitignore
```

> **Lưu ý**: Dataset ảnh (~12GB) được lưu trữ trên Kaggle:  
> 🔗 https://www.kaggle.com/datasets/nguyenbachtung/database-hcmute

---

## Môi trường & Thư viện

- Python 3.10+
- [ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv11)
- [vietocr](https://github.com/pbcquoc/vietocr)
- opencv-python, Pillow, PyMuPDF, python-docx
- Huấn luyện: Kaggle Notebook – GPU NVIDIA Tesla T4 x2

## Cài đặt

```bash
pip install ultralytics vietocr opencv-python pillow pymupdf python-docx
```

---

## Tiến độ

- [x] **Giai đoạn 1**: Thu thập & xây dựng dataset, gán nhãn 500 ảnh, publish Kaggle
- [x] **Giai đoạn 2**: Huấn luyện YOLOv11n/YOLOv8n, đánh giá định lượng, thử nghiệm VietOCR
- [ ] **Giai đoạn 3**: Fine-tuning VietOCR, Rule Engine đầy đủ, giải quyết Domain Shift
- [ ] **Giai đoạn 4**: Web Demo (Streamlit/FastAPI), đánh giá toàn hệ thống
