import os
import sys
import json
import glob
import time
import torch
import pandas as pd

# Fix encoding on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath('.'))

from pipeline.inference_engine import DocumentLayoutAndOCR

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Khởi tạo DocumentLayoutAndOCR trên {device}...")
    engine = DocumentLayoutAndOCR(device=device)

    test_dir = 'dataset_yolo_domain_adaptation/images/test'
    test_files = sorted(glob.glob(os.path.join(test_dir, '*.jpg')))
    if not test_files:
        test_dir = 'dataset_yolo_real_world/images'
        test_files = sorted(glob.glob(os.path.join(test_dir, '*.jpg')))[:17]

    print(f"📁 Tìm thấy {len(test_files)} ảnh camera kiểm thử thực tế trong: {test_dir}\n")

    records = []
    start_all = time.time()

    for idx, fpath in enumerate(test_files, 1):
        fname = os.path.basename(fpath)
        t0 = time.time()
        try:
            res = engine.process_document(fpath, validate=True)
            elapsed = time.time() - t0

            ents = res.get('entities', {})
            ver = res.get('verification', {})
            val = res.get('validation', {})
            courses = res.get('courses_table', [])

            form_type = val.get('form_type') or res.get('form_type') or 'UNKNOWN'
            status = val.get('status', 'N/A')
            compliance = val.get('compliance_score', 0.0)

            rec = {
                'STT': idx,
                'File': fname,
                'Form': form_type,
                'MSSV': ents.get('student_id', ''),
                'MSSV_Cu': ents.get('old_student_id', ''),
                'Ho_Ten': ents.get('student_name', ''),
                'Lop': ents.get('class_name', ''),
                'SDT': ents.get('phone', ''),
                'Ngay_Lap': ents.get('date', ''),
                'So_Mon': len(courses),
                'Chu_Ky': ver.get('signature_count', 0),
                'Moc_Do': ver.get('stamp_count', 0),
                'Trang_Thai': status,
                'Tuan_Thu_%': compliance,
                'Thoi_Gian_s': round(elapsed, 2)
            }
            records.append(rec)
            print(f"[{idx:02d}/{len(test_files):02d}] {fname:<22} | Form: {form_type:<14} | MSSV: {str(rec['MSSV']):<10} | Tên: {str(rec['Ho_Ten']):<20} | Môn: {rec['So_Mon']:<2} | Ký: {rec['Chu_Ky']:<2} | {status} ({compliance}%) | {elapsed:.2f}s")
        except Exception as e:
            print(f"[{idx:02d}/{len(test_files):02d}] {fname:<22} | LỖI: {e}")

    total_time = time.time() - start_all
    print(f"\n⏱️ Hoàn tất 17 tài liệu trong: {total_time:.2f} giây (Trung bình: {total_time/len(test_files):.2f}s/tài liệu)")

    df = pd.DataFrame(records)
    out_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'HCMUTE_YOLOv11_Domain_Adaptation_Artifacts_v2', 'real_17_docs_evaluation_results.csv')
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"💾 Đã lưu báo cáo chi tiết vào: {out_csv}")

if __name__ == '__main__':
    main()
