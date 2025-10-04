#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Ezequiel de la Rosa"

"""
Evaluate ISLES predictions against GT labels over the whole dataset.

Per-case metrics:
- Lesion-wise F1 (Panoptica RQ)        -> f1
- Dice coefficient (global_bin_dsc)    -> dice
- Absolute volume difference (mL)      -> avd_ml
- Instance count difference (Panoptica)-> inst_diff
- Absolute Lesion Count Difference     -> alcd  (|#GT - #Pred| via CC)
- GT / Pred volumes (mL), instance counts

Assumptions:
- GT files in --gt-dir named: case_XXXX.nii.gz (binary 0/1)
- Pred files in --pred-dir named: case_XXXX*.nii.gz (binary or soft)
"""

import argparse
import re
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import SimpleITK as sitk

# ---------------------- your evaluation functions (kept) ----------------------

from panoptica import (
    InputType,
    Panoptica_Evaluator,
    ConnectedComponentsInstanceApproximator,
    NaiveThresholdMatching,
)

def compute_absolute_volume_difference(im1, im2, voxel_size):
    """
    Computes the absolute volume difference between two masks (in mL).
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    voxel_size = float(voxel_size)

    if im1.shape != im2.shape:
        warnings.warn(
            "Shape mismatch: ground_truth and prediction have different shapes. "
            "AVD computed on mismatching masks."
        )

    ground_truth_volume = np.sum(im1) * voxel_size
    prediction_volume = np.sum(im2) * voxel_size
    abs_vol_diff = float(np.abs(ground_truth_volume - prediction_volume))
    return abs_vol_diff


def compute_dice_f1_instance_difference(ground_truth, prediction, empty_value=1.0):
    """
    Computes lesion-wise F1 (RQ), instance-count difference, and Dice.
    Returns (f1_score, instance_count_difference, dice_score).
    """
    ground_truth = np.asarray(ground_truth).astype(int)
    prediction   = np.asarray(prediction).astype(int)

    evaluator = Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(),
    )
    out = evaluator.evaluate(prediction, ground_truth, verbose=False)
    res = out.get("ungrouped", None) if isinstance(out, dict) else out

    instance_count_difference = abs(int(res.num_ref_instances) - int(res.num_pred_instances))

    if int(res.num_ref_instances) == 0 and int(res.num_pred_instances) == 0:
        f1_score   = float(empty_value)
        dice_score = float(empty_value)
    else:
        f1_score   = float(res.rq)
        dice_score = float(res.global_bin_dsc)

    return f1_score, instance_count_difference, dice_score

# ----------------------------- helpers ---------------------------------------

CASE_RE = re.compile(r"case_(\d+)\.nii\.gz$")

def case_id_from_name(path: Path):
    m = CASE_RE.search(path.name)
    return m.group(1) if m else None

def load_nifti(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))

def to_np(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img)

def resample_like(img: sitk.Image, ref: sitk.Image,
                  interp=sitk.sitkNearestNeighbor, default=0) -> sitk.Image:
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ref)
    rs.SetInterpolator(interp)
    rs.SetTransform(sitk.Transform())
    rs.SetDefaultPixelValue(default)
    return rs.Execute(img)

def voxel_volume_ml(img: sitk.Image) -> float:
    sx, sy, sz = img.GetSpacing()  # mm
    return float((sx * sy * sz) / 1000.0)  # mL

def binarize(arr: np.ndarray, thr: float | None) -> np.ndarray:
    if arr.dtype.kind in ("b", "i", "u"):
        return (arr > 0).astype(np.uint8)
    if thr is None:
        return (arr > 0).astype(np.uint8)
    return (arr >= thr).astype(np.uint8)

def count_components(mask_bin: np.ndarray) -> int:
    """Return number of connected components in a 3D binary mask."""
    img = sitk.GetImageFromArray(mask_bin.astype(np.uint8))
    cc = sitk.ConnectedComponent(img)
    stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(cc)
    return len(stats.GetLabels())

def compute_alcd(gt_bin: np.ndarray, pred_bin: np.ndarray) -> int:
    """Absolute Lesion Count Difference = |#GT - #Pred| (via CC)."""
    return abs(count_components(gt_bin) - count_components(pred_bin))

def remove_small_components(mask_bin: np.ndarray, min_vox: int) -> np.ndarray:
    """Remove connected components smaller than min_vox (3D)."""
    if min_vox <= 1:
        return mask_bin.astype(np.uint8)
    img = sitk.GetImageFromArray(mask_bin.astype(np.uint8))
    cc = sitk.ConnectedComponent(img)
    stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(cc)
    keep = sitk.Image(cc.GetSize(), sitk.sitkUInt8); keep.CopyInformation(cc)
    for lab in stats.GetLabels():
        if stats.GetNumberOfPixels(lab) >= int(min_vox):
            comp = sitk.BinaryThreshold(cc, lab, lab, 1, 0)
            keep = sitk.Or(keep, comp)
    return sitk.GetArrayFromImage(keep).astype(np.uint8)

# ----------------------------- main ------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir",   required=True, help="Path to labelsTs (GT NIfTI).")
    ap.add_argument("--pred-dir", required=True, help="Path to prediction NIfTI files.")
    ap.add_argument("--out-csv",  default="eval_results.csv")
    ap.add_argument("--thr",      type=float, default=None,
                    help="Threshold for soft predictions; default: >0")
    ap.add_argument("--min-vox",  type=int, default=0,
                    help="Remove predicted components smaller than this many voxels (post-proc).")
    ap.add_argument("--fail-on-missing", action="store_true",
                    help="Stop if a prediction is missing for a GT case.")
    args = ap.parse_args()

    gt_dir   = Path(args.gt_dir).resolve()
    pred_dir = Path(args.pred_dir).resolve()
    out_csv  = Path(args.out_csv).resolve()

    gt_files = sorted(gt_dir.glob("case_*.nii.gz"))
    if not gt_files:
        raise SystemExit(f"No GT files found in {gt_dir}")

    # index predictions by case id (accept case_XXXX*.nii.gz)
    pred_index: dict[str, Path] = {}
    for p in pred_dir.glob("*.nii.gz"):
        m = re.search(r"case_(\d+)", p.name)
        if m:
            pred_index[m.group(1)] = p

    rows = []
    for gt_path in gt_files:
        cid = case_id_from_name(gt_path)
        if cid is None:
            print(f"[SKIP] Unrecognized GT name: {gt_path.name}")
            continue

        pred_path = pred_index.get(cid)
        if pred_path is None:
            msg = f"[MISS] No prediction for case_{cid} in {pred_dir}"
            if args.fail_on_missing:
                raise SystemExit(msg)
            print(msg); continue

        # load & align
        gt_img   = load_nifti(gt_path)
        pred_img = load_nifti(pred_path)
        if (pred_img.GetSize()      != gt_img.GetSize() or
            pred_img.GetSpacing()   != gt_img.GetSpacing() or
            pred_img.GetDirection() != gt_img.GetDirection() or
            pred_img.GetOrigin()    != gt_img.GetOrigin()):
            pred_img = resample_like(pred_img, gt_img, sitk.sitkNearestNeighbor, default=0)

        gt_np   = to_np(gt_img)
        pred_np = to_np(pred_img)

        # binarize GT/pred
        gt_bin   = binarize(gt_np, thr=None)
        pred_bin = binarize(pred_np, thr=args.thr)

        # optional small-component removal on predictions
        if args.min_vox > 0:
            pred_bin = remove_small_components(pred_bin, args.min_vox)

        # metrics
        v_ml  = voxel_volume_ml(gt_img)
        f1, inst_diff, dice = compute_dice_f1_instance_difference(gt_bin, pred_bin, empty_value=1.0)
        avd_ml = compute_absolute_volume_difference(gt_bin, pred_bin, v_ml)

        # counts
        n_gt   = count_components(gt_bin)
        n_pred = count_components(pred_bin)
        alcd   = abs(n_gt - n_pred)   # Absolute Lesion Count Difference

        gt_vol_ml   = float(gt_bin.sum() * v_ml)
        pred_vol_ml = float(pred_bin.sum() * v_ml)

        rows.append({
            "case_id":        cid,
            "f1":             f1,
            "dice":           dice,
            "avd_ml":         avd_ml,
            "inst_diff":      inst_diff,   # from Panoptica
            "alcd":           alcd,        # from CC counts
            "gt_instances":   n_gt,
            "pred_instances": n_pred,
            "gt_vol_ml":      gt_vol_ml,
            "pred_vol_ml":    pred_vol_ml,
            "gt_path":        str(gt_path),
            "pred_path":      str(pred_path),
        })

        print(f"case_{cid}: F1={f1:.3f} | Dice={dice:.3f} | AVD={avd_ml:.2f} mL | "
              f"Δinst(Panoptica)={inst_diff} | ALCD={alcd}")

    if not rows:
        raise SystemExit("No cases evaluated (missing predictions?).")

    df = pd.DataFrame(rows).sort_values("case_id")
    df.to_csv(out_csv, index=False)

    # summary
    print("\n=== Test set summary ===")
    print(f"Cases: {len(df)}")
    print(f"F1 (mean):   {df['f1'].mean():.4f}")
    print(f"Dice (mean): {df['dice'].mean():.4f}")
    print(f"AVD  (mean): {df['avd_ml'].mean():.2f} mL")
    print(f"ALCD (mean): {df['alcd'].mean():.2f}")
    print(f"\nSaved per-case metrics → {out_csv}")

if __name__ == "__main__":
    main()
