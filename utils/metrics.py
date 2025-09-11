"""
Evaluate predicted masks against ground-truth masks and export metrics to XLSX.

Layout assumption under --base_dir (e.g., ./datas/V60):
  - masks/                 # ground truth
      <subfolder>/filename.tif[f]
  - masks_<modelA>/        # predictions for modelA
      <subfolder>/filename.tif[f]
  - masks_<modelB>/        # predictions for modelB
      ...

Outputs an Excel workbook with per-model summary metrics: accuracy, precision,
recall, F1, along with basic counts.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import tifffile


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eval_masks")


VALID_EXTS = (".tif", ".tiff")


def list_subdirs(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir()])


def list_files(path: Path, exts: Tuple[str, ...] = VALID_EXTS) -> List[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def find_corresponding(pred_dir: Path, gt_file: Path) -> Path | None:
    # Prefer exact name match
    cand = pred_dir / gt_file.name
    if cand.exists():
        return cand
    # Try swapping .tif/.tiff
    alt_suffix = ".tiff" if gt_file.suffix.lower() == ".tif" else ".tif"
    cand2 = pred_dir / (gt_file.stem + alt_suffix)
    if cand2.exists():
        return cand2
    return None


def to_binary(arr: np.ndarray) -> np.ndarray:
    # Accept 2D or 3D masks. Anything > 0 is positive.
    return (arr > 0).astype(np.uint8)


def accumulate_confusion(tp: int, fp: int, fn: int, tn: int, gt: np.ndarray, pr: np.ndarray) -> Tuple[int, int, int, int]:
    tp += int(np.sum((pr == 1) & (gt == 1)))
    fp += int(np.sum((pr == 1) & (gt == 0)))
    fn += int(np.sum((pr == 0) & (gt == 1)))
    tn += int(np.sum((pr == 0) & (gt == 0)))
    return tp, fp, fn, tn


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "total": float(total),
    }


def write_xlsx(rows: List[Dict[str, object]], out_path: Path) -> None:
    try:
        import openpyxl  # type: ignore
    except Exception:
        # Fallback to CSV if openpyxl not available
        csv_path = out_path.with_suffix(".csv")
        logger.warning(f"openpyxl not installed; writing CSV instead: {csv_path}")
        write_csv(rows, csv_path)
        return

    from openpyxl import Workbook  # type: ignore

    wb = Workbook()
    ws = wb.active
    ws.title = "summary"

    # Headers
    headers = [
        "model",
        "files_evaluated",
        "pixels_total",
        "tp",
        "fp",
        "fn",
        "tn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
    ]
    ws.append(headers)

    for r in rows:
        ws.append([
            r.get("model"),
            r.get("files_evaluated"),
            int(r.get("total", 0)),
            int(r.get("tp", 0)),
            int(r.get("fp", 0)),
            int(r.get("fn", 0)),
            int(r.get("tn", 0)),
            float(r.get("accuracy", 0.0)),
            float(r.get("precision", 0.0)),
            float(r.get("recall", 0.0)),
            float(r.get("f1", 0.0)),
            float(r.get("accuracy_mean", 0.0)),
            float(r.get("accuracy_std", 0.0)),
            float(r.get("precision_mean", 0.0)),
            float(r.get("precision_std", 0.0)),
            float(r.get("recall_mean", 0.0)),
            float(r.get("recall_std", 0.0)),
            float(r.get("f1_mean", 0.0)),
            float(r.get("f1_std", 0.0)),
        ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out_path))


def write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    import csv

    headers = [
        "model",
        "files_evaluated",
        "pixels_total",
        "tp",
        "fp",
        "fn",
        "tn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow([
                r.get("model"),
                r.get("files_evaluated"),
                int(r.get("total", 0)),
                int(r.get("tp", 0)),
                int(r.get("fp", 0)),
                int(r.get("fn", 0)),
                int(r.get("tn", 0)),
                float(r.get("accuracy", 0.0)),
                float(r.get("precision", 0.0)),
                float(r.get("recall", 0.0)),
                float(r.get("f1", 0.0)),
                float(r.get("accuracy_mean", 0.0)),
                float(r.get("accuracy_std", 0.0)),
                float(r.get("precision_mean", 0.0)),
                float(r.get("precision_std", 0.0)),
                float(r.get("recall_mean", 0.0)),
                float(r.get("recall_std", 0.0)),
                float(r.get("f1_mean", 0.0)),
                float(r.get("f1_std", 0.0)),
            ])


def evaluate_model(base_dir: Path, gt_dir: Path, pred_dir: Path) -> Tuple[Dict[str, float], int, List[Dict[str, float]]]:
    # Iterate subfolders based on ground truth
    tp = fp = fn = tn = 0
    file_count = 0
    per_file_metrics: List[Dict[str, float]] = []

    for sub in list_subdirs(gt_dir):
        gt_sub = sub
        pr_sub = pred_dir / sub.name
        if not pr_sub.exists():
            logger.warning(f"Missing predicted subfolder: {pr_sub}")
            continue

        gt_files = list_files(gt_sub)
        if not gt_files:
            continue

        for gtf in gt_files:
            prf = find_corresponding(pr_sub, gtf)
            if prf is None:
                logger.warning(f"Missing prediction for {gtf.relative_to(base_dir)}")
                continue
            try:
                g_arr = to_binary(tifffile.imread(str(gtf)))
                p_arr = to_binary(tifffile.imread(str(prf)))
            except Exception as e:
                logger.warning(f"Failed to read pair {gtf} / {prf}: {e}")
                continue

            if g_arr.shape != p_arr.shape:
                logger.warning(f"Shape mismatch {gtf.name}: gt {g_arr.shape} vs pred {p_arr.shape}; skipping")
                continue

            # collect per-file metrics for mean/std
            try:
                tpf = int(np.sum((p_arr == 1) & (g_arr == 1)))
                fpf = int(np.sum((p_arr == 1) & (g_arr == 0)))
                fnf = int(np.sum((p_arr == 0) & (g_arr == 1)))
                tnf = int(np.sum((p_arr == 0) & (g_arr == 0)))
                per_file_metrics.append(compute_metrics(tpf, fpf, fnf, tnf))
            except Exception as e:
                logger.warning(f"Per-file metric calc failed for {gtf}: {e}")

            tp, fp, fn, tn = accumulate_confusion(tp, fp, fn, tn, g_arr, p_arr)
            file_count += 1

    metrics = compute_metrics(tp, fp, fn, tn)
    return metrics, file_count, per_file_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate predicted masks against GT and export XLSX")
    p.add_argument("--base_dir", type=str, required=True, help="Base folder containing masks/ and masks_* folders")
    p.add_argument("--gt_name", type=str, default="masks", help="Ground-truth masks folder name (default: masks)")
    p.add_argument("--pred_prefix", type=str, default="masks_", help="Prefix for prediction folders (default: masks_)")
    p.add_argument("--output", type=str, default="metrics.xlsx", help="Output XLSX filename (relative to base_dir)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base_dir)
    if not base.exists():
        logger.error(f"Base dir not found: {base}")
        return 1

    gt_dir = base / args.gt_name
    if not gt_dir.exists():
        logger.error(f"Ground-truth folder not found: {gt_dir}")
        return 1

    # Discover prediction folders
    pred_dirs = [p for p in list_subdirs(base) if p.name.startswith(args.pred_prefix)]
    if not pred_dirs:
        logger.error(f"No prediction folders starting with '{args.pred_prefix}' found under {base}")
        return 1

    rows: List[Dict[str, object]] = []
    for pred in pred_dirs:
        model_name = pred.name[len(args.pred_prefix):] or pred.name
        logger.info(f"Evaluating model: {model_name}")
        metrics, nfiles, per_file = evaluate_model(base, gt_dir, pred)

        vals_acc = np.array([m.get("accuracy", 0.0) for m in per_file], dtype=float)
        vals_prec = np.array([m.get("precision", 0.0) for m in per_file], dtype=float)
        vals_rec = np.array([m.get("recall", 0.0) for m in per_file], dtype=float)
        vals_f1 = np.array([m.get("f1", 0.0) for m in per_file], dtype=float)

        rows.append({
            "model": model_name,
            "files_evaluated": nfiles,
            **metrics,
            "accuracy_mean": float(vals_acc.mean()) if vals_acc.size else 0.0,
            "accuracy_std": float(vals_acc.std()) if vals_acc.size else 0.0,
            "precision_mean": float(vals_prec.mean()) if vals_prec.size else 0.0,
            "precision_std": float(vals_prec.std()) if vals_prec.size else 0.0,
            "recall_mean": float(vals_rec.mean()) if vals_rec.size else 0.0,
            "recall_std": float(vals_rec.std()) if vals_rec.size else 0.0,
            "f1_mean": float(vals_f1.mean()) if vals_f1.size else 0.0,
            "f1_std": float(vals_f1.std()) if vals_f1.size else 0.0,
        })

    out_path = base / args.output
    write_xlsx(rows, out_path)
    logger.info(f"Wrote metrics: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
