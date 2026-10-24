# -*- coding: utf-8 -*-
"""
VietOCR Fine-Tuning Script on Academic Document Crops (HCMUTE DocAI).
Configures VietOCR VGG-Transformer Trainer, evaluates Baseline vs. Fine-tuned
on Unseen Test Set, and calculates CER (Character Error Rate), WER (Word Error Rate),
and Exact Match (%) to answer Research Question 1.
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import torch
except ImportError:
    torch = None

try:
    import editdistance
except ImportError:
    editdistance = None

try:
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor
    from vietocr.model.trainer import Trainer
except ImportError:
    Cfg = None
    Predictor = None
    Trainer = None

def compute_ocr_metrics(predictions, ground_truths):
    """
    Computes Exact Match (%), Character Error Rate (CER %), and Word Error Rate (WER %).
    """
    total_chars = 0
    total_words = 0
    char_edit_dist = 0
    word_edit_dist = 0
    exact_matches = 0
    total_samples = len(ground_truths)

    for pred, gt in zip(predictions, ground_truths):
        pred_clean = pred.strip()
        gt_clean = gt.strip()

        if pred_clean == gt_clean:
            exact_matches += 1

        if editdistance is not None:
            char_edit_dist += editdistance.eval(pred_clean, gt_clean)
            pred_words = pred_clean.split()
            gt_words = gt_clean.split()
            word_edit_dist += editdistance.eval(pred_words, gt_words)
        else:
            # Fallback simple char diff
            char_edit_dist += abs(len(pred_clean) - len(gt_clean))
            word_edit_dist += abs(len(pred_clean.split()) - len(gt_clean.split()))

        total_chars += len(gt_clean)
        total_words += len(gt_clean.split())

    cer = (char_edit_dist / max(1, total_chars)) * 100.0
    wer = (word_edit_dist / max(1, total_words)) * 100.0
    em = (exact_matches / max(1, total_samples)) * 100.0

    return {
        "exact_match": round(em, 2),
        "cer": round(cer, 2),
        "wer": round(wer, 2),
        "total_samples": total_samples,
        "exact_matches_count": exact_matches
    }

def evaluate_test_set(predictor, data_root, test_ann_path, max_samples=None):
    """
    Evaluates a VietOCR Predictor on test_annotation.txt.
    Returns metrics dict and sample predictions list.
    """
    print(f"\n🔍 Đang đánh giá trên tập test: {test_ann_path} ...")
    test_ann_file = os.path.join(data_root, test_ann_path) if not os.path.isabs(test_ann_path) else test_ann_path

    if not os.path.exists(test_ann_file):
        print(f"⚠️ Không tìm thấy file test annotation: {test_ann_file}")
        return None, []

    with open(test_ann_file, "r", encoding="utf-8") as f:
        lines = [l.strip().split("\t") for l in f if l.strip() and "\t" in l]

    if max_samples and max_samples < len(lines):
        lines = lines[:max_samples]

    preds = []
    gts = []
    details = []

    start_eval = time.time()
    for idx, (rel_img_path, gt_text) in enumerate(lines, 1):
        full_img_path = os.path.join(data_root, rel_img_path)
        if not os.path.exists(full_img_path):
            continue

        try:
            with Image.open(full_img_path).convert("RGB") as img:
                pred_text = predictor.predict(img)
        except Exception as e:
            pred_text = ""

        preds.append(pred_text)
        gts.append(gt_text)
        details.append({
            "img": rel_img_path,
            "gt": gt_text,
            "pred": pred_text,
            "is_exact": (pred_text.strip() == gt_text.strip())
        })

        if idx % 200 == 0 or idx == len(lines):
            print(f"   [{idx:04d}/{len(lines):04d}] Đã dự đoán...")

    elapsed = time.time() - start_eval
    metrics = compute_ocr_metrics(preds, gts)
    metrics["eval_time_sec"] = round(elapsed, 2)
    metrics["fps"] = round(len(preds) / max(0.001, elapsed), 1)

    print(f"✅ Hoàn tất đánh giá trong {elapsed:.1f}s ({metrics['fps']} img/s)")
    print(f"   - Exact Match : {metrics['exact_match']}% ({metrics['exact_matches_count']}/{metrics['total_samples']})")
    print(f"   - CER         : {metrics['cer']}%")
    print(f"   - WER         : {metrics['wer']}%")

    return metrics, details

def build_vietocr_config(
    data_root="dataset_ocr",
    train_ann="train_annotation.txt",
    val_ann="val_annotation.txt",
    batch_size=32,
    iters=15000,
    lr=0.0001,
    device="cuda:0",
    export_path="weights/vietocr_hcmute_finetuned.pth",
    checkpoint_path="checkpoint/vietocr_checkpoint.pth",
    log_path="train_vietocr.log"
):
    """
    Creates customized Cfg for VietOCR VGG-Transformer Fine-Tuning.
    """
    if Cfg is None:
        raise ImportError("Chưa cài đặt thư viện vietocr. Vui lòng cài: pip install vietocr")

    config = Cfg.load_config_from_name('vgg_transformer')
    
    # 1. Device configuration
    if device.startswith("cuda") and torch is not None and not torch.cuda.is_available():
        print("⚠️ CUDA không khả dụng, tự động chuyển về device: cpu")
        device = "cpu"
    config['device'] = device

    # 2. Dataset configuration
    config['dataset']['name'] = 'hcmute_doc'
    config['dataset']['data_root'] = str(data_root)
    config['dataset']['train_annotation'] = str(train_ann)
    config['dataset']['valid_annotation'] = str(val_ann)
    config['dataset']['image_height'] = 32
    config['dataset']['image_min_width'] = 32
    config['dataset']['image_max_width'] = 512

    # 3. Training & Optimizer Hyperparameters
    config['trainer']['batch_size'] = batch_size
    config['trainer']['print_every'] = 100
    config['trainer']['valid_every'] = 1000
    config['trainer']['iters'] = iters
    config['trainer']['export'] = str(export_path)
    config['trainer']['checkpoint'] = str(checkpoint_path)
    config['trainer']['log'] = str(log_path)
    config['trainer']['metrics'] = None

    # Fine-tuning uses lower learning rate to preserve pre-trained knowledge
    config['optimizer']['max_lr'] = lr
    config['optimizer']['pct_start'] = 0.1

    # Data augmentation for robust recognition
    config['aug']['image_aug'] = True
    config['aug']['masked_language_model'] = True
    config['predictor']['beamsearch'] = False

    # Ensure parent export dirs exist
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    return config

def run_fine_tuning(config, eval_baseline=True):
    """
    Executes full pipeline:
      1. Baseline evaluation (Pretrained VietOCR)
      2. Fine-tuning with VietOCR Trainer
      3. Post-finetuning evaluation & comparison table
    """
    data_root = config['dataset']['data_root']
    test_ann = "test_annotation.txt"
    export_weights = config['trainer']['export']

    baseline_metrics = None
    # 1. Baseline Evaluation
    if eval_baseline:
        print("\n" + "=" * 80)
        print("📊 [BƯỚC 1/3] ĐÁNH GIÁ BASELINE VIETOCR (PRETRAINED) TRÊN TẬP TEST")
        print("=" * 80)
        try:
            base_cfg = Cfg.load_config_from_name('vgg_transformer')
            base_cfg['device'] = config['device']
            base_cfg['prediction_chk'] = None
            base_predictor = Predictor(base_cfg)
            baseline_metrics, _ = evaluate_test_set(base_predictor, data_root, test_ann)
        except Exception as e:
            print(f"⚠️ Không thể chạy baseline evaluation ({e}), tiếp tục huấn luyện...")

    # 2. Run Fine-Tuning
    print("\n" + "=" * 80)
    print("🚀 [BƯỚC 2/3] BẮT ĐẦU HUẤN LUYỆN FINE-TUNING VIETOCR (VGG-TRANSFORMER)")
    print(f"   - Iterations : {config['trainer']['iters']}")
    print(f"   - Batch Size : {config['trainer']['batch_size']}")
    print(f"   - Max LR     : {config['optimizer']['max_lr']}")
    print(f"   - Device     : {config['device']}")
    print(f"   - Export Wts : {export_weights}")
    print("=" * 80)

    trainer = Trainer(config, pretrained=True)
    start_train = time.time()
    trainer.train()
    train_elapsed = time.time() - start_train
    print(f"\n🎉 HUẤN LUYỆN HOÀN TẤT TRONG {train_elapsed/60:.1f} PHÚT! Trọng số đã lưu tại: {export_weights}")

    # 3. Post-Fine-Tuning Evaluation
    print("\n" + "=" * 80)
    print("📈 [BƯỚC 3/3] ĐÁNH GIÁ MÔ HÌNH SAU KHI FINE-TUNING TRÊN CÙNG TẬP TEST")
    print("=" * 80)

    finetuned_metrics = None
    if os.path.exists(export_weights):
        eval_cfg = Cfg.load_config_from_name('vgg_transformer')
        eval_cfg['device'] = config['device']
        eval_cfg['prediction_chk'] = export_weights
        ft_predictor = Predictor(eval_cfg)
        finetuned_metrics, _ = evaluate_test_set(ft_predictor, data_root, test_ann)

    # 4. Summary & Comparison
    if baseline_metrics and finetuned_metrics:
        print("\n" + "=" * 80)
        print("🏆 BẢNG SO SÁNH ĐỊNH LƯỢNG TRƯỚC VÀ SAU FINE-TUNING (CÂU HỎI NGHIÊN CỨU 1):")
        print("=" * 80)
        delta_em = finetuned_metrics['exact_match'] - baseline_metrics['exact_match']
        delta_cer = finetuned_metrics['cer'] - baseline_metrics['cer']
        delta_wer = finetuned_metrics['wer'] - baseline_metrics['wer']

        print(f"{'Chỉ số':<22} | {'Baseline (Pretrained)':<22} | {'Fine-tuned (HCMUTE)':<22} | {'Chênh lệch (Δ)':<15}")
        print("-" * 88)
        print(f"{'Exact Match (%)':<22} | {baseline_metrics['exact_match']:>20}% | {finetuned_metrics['exact_match']:>20}% | {delta_em:>+14.2f}%")
        print(f"{'CER (%)':<22} | {baseline_metrics['cer']:>20}% | {finetuned_metrics['cer']:>20}% | {delta_cer:>+14.2f}%")
        print(f"{'WER (%)':<22} | {baseline_metrics['wer']:>20}% | {finetuned_metrics['wer']:>20}% | {delta_wer:>+14.2f}%")
        print("=" * 88)

        # Save to JSON for report
        comparison_record = {
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "HCMUTE Academic Document Crops",
            "test_samples": finetuned_metrics["total_samples"],
            "baseline": baseline_metrics,
            "finetuned": finetuned_metrics,
            "improvement": {
                "exact_match_delta": round(delta_em, 2),
                "cer_delta": round(delta_cer, 2),
                "wer_delta": round(delta_wer, 2)
            }
        }
        res_json = os.path.join(os.path.dirname(export_weights), "vietocr_research_question_1_results.json")
        with open(res_json, "w", encoding="utf-8") as f:
            json.dump(comparison_record, f, indent=2, ensure_ascii=False)
        print(f"💾 Đã lưu kết quả đối chiếu vào: {res_json}")

    return {
        "baseline": baseline_metrics,
        "finetuned": finetuned_metrics,
        "export_weights": export_weights
    }

def main():
    parser = argparse.ArgumentParser(description="VietOCR Fine-Tuning Pipeline for HCMUTE Academic Forms")
    parser.add_argument("--data-root", default="dataset_ocr", help="Root folder of OCR crop dataset")
    parser.add_argument("--train-ann", default="train_annotation.txt", help="Train annotation text file")
    parser.add_argument("--val-ann", default="val_annotation.txt", help="Validation annotation text file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--iters", type=int, default=15000, help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=0.0001, help="Max learning rate")
    parser.add_argument("--device", default="cuda:0", help="Computation device (cuda:0 or cpu)")
    parser.add_argument("--export", default="weights/vietocr_hcmute_finetuned.pth", help="Path to export best model weights")
    parser.add_argument("--checkpoint", default="checkpoint/vietocr_checkpoint.pth", help="Checkpoint save path")
    parser.add_argument("--no-eval-baseline", action="store_true", help="Skip pre-training baseline evaluation")

    args = parser.parse_args()

    config = build_vietocr_config(
        data_root=args.data_root,
        train_ann=args.train_ann,
        val_ann=args.val_ann,
        batch_size=args.batch_size,
        iters=args.iters,
        lr=args.lr,
        device=args.device,
        export_path=args.export,
        checkpoint_path=args.checkpoint
    )

    run_fine_tuning(config, eval_baseline=not args.no_eval_baseline)

if __name__ == "__main__":
    main()
