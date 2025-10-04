"""
ISLES'24 → nnU-Net v2 (multimodal) with label-aware skull-strip
+ per-modality clipping (single-process), using a final mask that
covers brain plus a small padded band around the lesion.

Inputs (found recursively under --in-roots):
  NCCT (RAW):      .../raw_data/sub-strokeXXXX/ses-01/sub-strokeXXXX_ses-01_ncct.nii.gz
  CTA:             .../derivatives/sub-strokeXXXX/ses-01/sub-strokeXXXX_ses-01_space-ncct_cta.nii.gz
  Perfusion maps (in perfusion-maps/):
    CBF:           .../derivatives/sub-strokeXXXX/ses-01/perfusion-maps/sub-strokeXXXX_ses-01_space-ncct_cbf.nii.gz
    CBV:           .../derivatives/sub-strokeXXXX/ses-01/perfusion-maps/sub-strokeXXXX_ses-01_space-ncct_cbv.nii.gz
    MTT:           .../derivatives/sub-strokeXXXX/ses-01/perfusion-maps/sub-strokeXXXX_ses-01_space-ncct_mtt.nii.gz
    Tmax:          .../derivatives/sub-strokeXXXX/ses-01/perfusion-maps/sub-strokeXXXX_ses-01_space-ncct_tmax.nii.gz
  Label (GT):      .../derivatives/sub-strokeXXXX/ses-02/sub-strokeXXXX_ses-02_lesion-msk.nii.gz

Outputs (to $nnUNet_raw/Dataset{ID}_{NAME} or --out-root):
  imagesTr/case_<ID>_0000.nii.gz   (ncct, float32, brain+lesion-padded mask, clipped 0–90)
  imagesTr/case_<ID>_0001.nii.gz   (cta,  float32, mask, 0–300)
  imagesTr/case_<ID>_0002.nii.gz   (cbv,  float32, mask, 0–10)
  imagesTr/case_<ID>_0003.nii.gz   (cbf,  float32, mask, 0–80)
  imagesTr/case_<ID>_0004.nii.gz   (mtt,  float32, mask, 0–20)
  imagesTr/case_<ID>_0005.nii.gz   (tmax, float32, mask, 0–10)
  labelsTr/case_<ID>.nii.gz        (uint8, background=0, lesion=1)
  imagesTs/...                      (same channels; labelsTs kept only for QA)
  dataset.json

Notes:
- No multiprocessing; purely sequential for reliability.
- Labels are only NN-resampled to NCCT grid if needed (never otherwise altered).
- Final intensity mask = dilated(brain) OR dilated(label) to avoid label-on-zeros.
"""

from __future__ import annotations
import argparse, os, re, sys
from pathlib import Path
from typing import Dict, List, Optional
import SimpleITK as sitk

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

# ---------- path patterns (match your tree exactly) ----------
PATTERNS = {
    "ncct": [
        "**/raw_data/sub-stroke*/ses-01/*_ses-01_ncct.nii.gz",
        "**/raw_data/sub-stroke*/ses-01/*_ses-01_*ncct.nii.gz",
        "**/derivatives/sub-stroke*/ses-01/*space-ncct_ncct.nii.gz",
    ],
    "cta":  ["**/derivatives/sub-stroke*/ses-01/*_ses-01_space-ncct_cta.nii.gz"],
    "cbf":  ["**/derivatives/sub-stroke*/ses-01/perfusion-maps/*_ses-01_space-ncct_cbf.nii.gz"],
    "cbv":  ["**/derivatives/sub-stroke*/ses-01/perfusion-maps/*_ses-01_space-ncct_cbv.nii.gz"],
    "mtt":  ["**/derivatives/sub-stroke*/ses-01/perfusion-maps/*_ses-01_space-ncct_mtt.nii.gz"],
    "tmax": ["**/derivatives/sub-stroke*/ses-01/perfusion-maps/*_ses-01_space-ncct_tmax.nii.gz"],
    "label":["**/derivatives/sub-stroke*/ses-02/*_ses-02_lesion-msk.nii.gz"],
}

# ---------- intensity windows (fixed caps; keep simple/robust) ----------
WINDOWS = {
    "ncct": (0.0, 90.0),
    "cta":  (0.0, 300.0),  # widened to preserve vessels/contrast
    "cbv":  (0.0, 10.0),
    "cbf":  (0.0, 80.0),   # allow normal GM range across vendors
    "mtt":  (0.0, 20.0),
    "tmax": (0.0, 10.0),   # a bit looser than 0–7
}

# nnU-Net channel ordering -> _0000.._0005
CHANNEL_ORDER = ["ncct","cta","cbv","cbf","mtt","tmax"]

# ---------- utilities ----------
SUBJ_PAT = re.compile(r"sub-stroke(\d+)")
ID4 = lambda x: str(int(x)).zfill(4)

def extract_subject_id(p: Path) -> Optional[str]:
    m = SUBJ_PAT.search(p.name)
    if not m:
        for part in p.parts[::-1]:
            m = SUBJ_PAT.search(part)
            if m:
                break
    return ID4(m.group(1)) if m else None

def mm_to_radius(mm: float, spacing: tuple[float,float,float]) -> List[int]:
    return [max(1, int(round(mm / s))) for s in spacing]

def resample_like(img: sitk.Image, ref: sitk.Image, interp=sitk.sitkLinear, default=0.0) -> sitk.Image:
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ref)
    rs.SetInterpolator(interp)
    rs.SetTransform(sitk.Transform())
    rs.SetDefaultPixelValue(default)
    return rs.Execute(img)

def verify_and_align_label(img: sitk.Image, ref: sitk.Image) -> sitk.Image:
    """Verify and ensure label alignment with reference image"""
    needs_resampling = (
        img.GetSize() != ref.GetSize() or
        img.GetSpacing() != ref.GetSpacing() or
        img.GetDirection() != ref.GetDirection() or
        img.GetOrigin() != ref.GetOrigin()
    )
    if needs_resampling:
        print("Warning: Label needs resampling to match reference image")
        img = resample_like(img, ref, sitk.sitkNearestNeighbor, default=0)
    return img

def verify_label_coverage(label_bin: sitk.Image, mask: sitk.Image) -> float:
    """Check what percentage of label is within a given mask"""
    label_binary = sitk.Cast(label_bin > 0, sitk.sitkUInt8)
    intersection = sitk.Multiply(label_binary, mask)
    label_voxels = sitk.GetArrayFromImage(label_binary).sum()
    intersection_voxels = sitk.GetArrayFromImage(intersection).sum()
    if label_voxels == 0:
        return 0.0
    return float(intersection_voxels) / float(label_voxels)

def keep_components_touching_label(mask: sitk.Image, label_bin: sitk.Image) -> sitk.Image:
    """Keep only connected components of mask that touch the lesion (fallback to largest)."""
    cc = sitk.ConnectedComponent(mask)
    stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(cc)
    if not stats.GetLabels():
        return sitk.Cast(mask, sitk.sitkUInt8)
    kept = None
    for lab in stats.GetLabels():
        comp = sitk.BinaryThreshold(cc, lowerThreshold=lab, upperThreshold=lab, insideValue=1, outsideValue=0)
        if sitk.GetArrayFromImage(sitk.And(comp, label_bin)).any():
            kept = comp if kept is None else sitk.Or(kept, comp)
    if kept is None:
        largest = max(stats.GetLabels(), key=lambda l: stats.GetPhysicalSize(l))
        kept = sitk.BinaryThreshold(cc, lowerThreshold=largest, upperThreshold=largest, insideValue=1, outsideValue=0)
    return sitk.Cast(kept, sitk.sitkUInt8)

def skull_strip_from_ncct(ncct: sitk.Image,
                          clamp=(-100, 200), soft=(-15,100),
                          close_mm=3.0, erode_mm=1.0,
                          skull_hu=250, shave_mm=0.8,
                          label_img=None) -> sitk.Image:
    """Gentle, label-aware skull strip on NCCT."""
    sp = ncct.GetSpacing()
    clamped = sitk.Clamp(ncct, lowerBound=clamp[0], upperBound=clamp[1])
    softmask = sitk.BinaryThreshold(clamped, lowerThreshold=soft[0], upperThreshold=soft[1], insideValue=1, outsideValue=0)
    softmask = sitk.BinaryMorphologicalClosing(softmask, mm_to_radius(close_mm, sp))
    softmask = sitk.VotingBinaryIterativeHoleFilling(softmask, radius=mm_to_radius(2.0, sp),
                                                     majorityThreshold=1, backgroundValue=0, foregroundValue=1,
                                                     maximumNumberOfIterations=1)

    skull = sitk.BinaryThreshold(ncct, lowerThreshold=skull_hu, upperThreshold=3000, insideValue=1, outsideValue=0)
    brain0 = sitk.And(softmask, sitk.BinaryNot(sitk.BinaryDilate(skull, mm_to_radius(shave_mm, sp))))
    brain0 = sitk.BinaryErode(brain0, mm_to_radius(erode_mm, sp))
    brain0 = sitk.BinaryMorphologicalClosing(brain0, mm_to_radius(1.0, sp))

    if label_img is not None:
        label_bin = sitk.Cast(label_img > 0, sitk.sitkUInt8)
        brain = keep_components_touching_label(brain0, label_bin)
        brain = sitk.Or(brain, label_bin)  # ensure label retained
    else:
        brain = brain0

    return sitk.Cast(brain, sitk.sitkUInt8)

def clip_and_mask(img: sitk.Image, mask: sitk.Image, vmin: float, vmax: float) -> sitk.Image:
    clamped = sitk.Clamp(img, lowerBound=float(vmin), upperBound=float(vmax))
    masked = sitk.Mask(clamped, mask)   # outside brain -> 0
    return sitk.Cast(masked, sitk.sitkFloat32)

# ---------- indexing ----------
def build_index(roots: List[Path]) -> Dict[str, Dict[str, Path]]:
    index: Dict[str, Dict[str, Path]] = {}
    for key, patterns in PATTERNS.items():
        for root in roots:
            for pat in patterns:
                for p in root.rglob(pat):
                    sid = extract_subject_id(p)
                    if not sid:
                        continue
                    rec = index.setdefault(sid, {})
                    if key not in rec:  # keep first match per key
                        rec[key] = p
    return index

# ---------- per-subject processing ----------
def process_subject(sid: str, rec: Dict[str, Path], out_images: Path, out_labels: Path,
                    mask_dir: Optional[Path]=None) -> bool:
    # sanity: require all channels + label
    for key in CHANNEL_ORDER + ["label"]:
        if key not in rec:
            print(f"[WARN] {sid}: missing '{key}', skipping.")
            return False

    # Read NCCT + label, align label to NCCT grid
    ncct = sitk.ReadImage(str(rec["ncct"]))
    lab = sitk.ReadImage(str(rec["label"]))
    lab = verify_and_align_label(lab, ncct)
    lab_binary = sitk.Cast(lab > 0, sitk.sitkUInt8)

    # Skull-strip (label-aware) and build final masking mask
    brain = skull_strip_from_ncct(ncct, label_img=lab_binary)

    # Final mask for intensities: small padding around both brain and label
    context_mm = 1.0
    label_pad_mm = 2.0
    brain_pad  = sitk.BinaryDilate(brain,      mm_to_radius(context_mm, ncct.GetSpacing()))
    label_pad  = sitk.BinaryDilate(lab_binary, mm_to_radius(label_pad_mm, ncct.GetSpacing()))
    mask_final = sitk.Or(brain_pad, label_pad)

    # QA: lesion coverage and outside count
    coverage = verify_label_coverage(lab_binary, mask_final)
    outside = sitk.And(lab_binary, sitk.BinaryNot(mask_final))
    n_out = int(sitk.GetArrayFromImage(outside).sum())
    if coverage < 0.995 or n_out > 0:
        print(f"[QA] {sid}: lesion coverage {coverage*100:.2f}% | voxels outside mask_final: {n_out}")

    if mask_dir:
        mask_dir.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(brain,      str(mask_dir / f"case_{sid}_brainmask.nii.gz"))
        sitk.WriteImage(mask_final, str(mask_dir / f"case_{sid}_maskfinal.nii.gz"))
        sitk.WriteImage(lab_binary, str(mask_dir / f"case_{sid}_label.nii.gz"))

    # Write channels (aligned to NCCT grid; linear for images)
    for ch, key in enumerate(CHANNEL_ORDER):
        img = sitk.ReadImage(str(rec[key]))
        if (img.GetSize()!=ncct.GetSize() or img.GetSpacing()!=ncct.GetSpacing() or
            img.GetDirection()!=ncct.GetDirection() or img.GetOrigin()!=ncct.GetOrigin()):
            img = resample_like(img, ncct, sitk.sitkLinear, default=0.0)
        vmin, vmax = WINDOWS[key]
        out = clip_and_mask(img, mask_final, vmin, vmax)
        sitk.WriteImage(out, str(out_images / f"case_{sid}_{ch:04d}.nii.gz"))

    # Write label (uint8) using the already-aligned lab_binary
    sitk.WriteImage(lab_binary, str(out_labels / f"case_{sid}.nii.gz"))
    return True

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="ISLES'24 → nnU-Net v2 (ncct+cta+cbv+cbf+mtt+tmax), label-aware skull-strip & clipping (single-process).")
    ap.add_argument("--in-roots", nargs="+", required=True,
                    help="Top-level ISLES roots, e.g. /bhome/test/isles24_train-1 /bhome/test/isles24_train-2/isles24_train_b")
    ap.add_argument("--dataset-id", type=int, required=True, help="e.g., 010")
    ap.add_argument("--dataset-name", default="ISLES24_multi", help="Dataset nickname.")
    ap.add_argument("--out-root", default=None,
                    help="Explicit output root (else uses $nnUNet_raw/Dataset{ID}_{NAME})")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--shuffle", action="store_true", help="Shuffle subjects before split.")
    ap.add_argument("--save-brainmasks", action="store_true", help="Save brain/final masks for QA.")
    args = ap.parse_args()

    # Resolve output base
    if args.out_root:
        out_base = Path(args.out_root).expanduser().resolve()
    else:
        if not nnUNet_raw:
            sys.exit("nnUNet_raw is not set (nnunetv2.paths.nnUNet_raw). Use --out-root or export nnUNet_raw.")
        out_base = Path(nnUNet_raw) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"

    out_imagesTr = out_base / "imagesTr"
    out_labelsTr = out_base / "labelsTr"
    out_imagesTs = out_base / "imagesTs"
    out_labelsTs = out_base / "labelsTs"  # nnU-Net ignores labelsTs; kept for QA only
    for d in (out_imagesTr, out_labelsTr, out_imagesTs, out_labelsTs):
        d.mkdir(parents=True, exist_ok=True)
    mask_dir = (out_base / "brainmasks") if args.save_brainmasks else None

    roots = [Path(r).expanduser().resolve() for r in args.in_roots]
    index = build_index(roots)

    # keep subjects that have all modalities + label
    subjects = sorted([sid for sid, rec in index.items() if all(k in rec for k in CHANNEL_ORDER+["label"])],
                      key=lambda s: int(s))
    if not subjects:
        sys.exit("No complete subjects found. Check your paths/patterns.")

    # split
    if args.shuffle:
        import random; random.seed(42); random.shuffle(subjects)
    n_tr = int(round(len(subjects)*args.train_frac))
    train_sids, test_sids = subjects[:n_tr], subjects[n_tr:]
    print(f"Subjects: {len(subjects)} | train: {len(train_sids)} | test: {len(test_sids)}")

    # process TRAIN sequentially
    for sid in train_sids:
        try:
            ok = process_subject(sid, index[sid], out_imagesTr, out_labelsTr, mask_dir)
            print(f"Train {sid}: {'OK' if ok else 'FAIL'}")
        except Exception as e:
            print(f"Train {sid}: FAIL - {e}")

    # process TEST sequentially (including labels for QA)
    for sid in test_sids:
        try:
            rec = index[sid]

            # Read NCCT + label, align label to NCCT grid
            ncct = sitk.ReadImage(str(rec["ncct"]))
            lab = sitk.ReadImage(str(rec["label"]))
            lab = verify_and_align_label(lab, ncct)
            lab_binary = sitk.Cast(lab > 0, sitk.sitkUInt8)

            # Skull-strip (label-aware) and build final masking mask
            brain = skull_strip_from_ncct(ncct, label_img=lab_binary)
            context_mm = 1.0
            label_pad_mm = 0.3
            brain_pad  = sitk.BinaryDilate(brain,      mm_to_radius(context_mm, ncct.GetSpacing()))
            label_pad  = sitk.BinaryDilate(lab_binary, mm_to_radius(label_pad_mm, ncct.GetSpacing()))
            mask_final = sitk.Or(brain_pad, label_pad)

            if mask_dir:
                sitk.WriteImage(brain,      str(mask_dir / f"case_{sid}_brainmask.nii.gz"))
                sitk.WriteImage(mask_final, str(mask_dir / f"case_{sid}_maskfinal.nii.gz"))
                sitk.WriteImage(lab_binary, str(mask_dir / f"case_{sid}_label.nii.gz"))

            # Process and save images
            for ch, key in enumerate(CHANNEL_ORDER):
                img = sitk.ReadImage(str(rec[key]))
                if (img.GetSize()!=ncct.GetSize() or img.GetSpacing()!=ncct.GetSpacing() or
                    img.GetDirection()!=ncct.GetDirection() or img.GetOrigin()!=ncct.GetOrigin()):
                    img = resample_like(img, ncct, sitk.sitkLinear, default=0.0)
                vmin, vmax = WINDOWS[key]
                out = clip_and_mask(img, mask_final, vmin, vmax)
                sitk.WriteImage(out, str(out_imagesTs / f"case_{sid}_{ch:04d}.nii.gz"))

            # Save test label (QA)
            sitk.WriteImage(lab_binary, str(out_labelsTs / f"case_{sid}.nii.gz"))

            # QA prints
            coverage = verify_label_coverage(lab_binary, mask_final)
            outside = sitk.And(lab_binary, sitk.BinaryNot(mask_final))
            n_out = int(sitk.GetArrayFromImage(outside).sum())
            if coverage < 0.995 or n_out > 0:
                print(f"[QA] {sid}: lesion coverage {coverage*100:.2f}% | voxels outside mask_final: {n_out}")

            print(f"Test  {sid}: OK")
        except Exception as e:
            print(f"Test  {sid}: FAIL - {e}")

    # dataset.json
    channel_names = {i: CHANNEL_ORDER[i] for i in range(len(CHANNEL_ORDER))}
    labels = {"background": 0, "lesion": 1}
    generate_dataset_json(
        output_folder=str(out_base),
        channel_names=channel_names,
        labels=labels,
        num_training_cases=len(train_sids),
        file_ending=".nii.gz",
        regions_class_order=None,
        dataset_name=f"Dataset{args.dataset_id:03d}_{args.dataset_name}",
        reference="ISLES 2024 (NCCT+CTA+CBV+CBF+MTT+Tmax)",
        release="",
        overwrite_image_reader_writer=None,  # avoid unexpected reorientation at I/O
        dataset_description="Label-aware brain mask (+ small padding) & per-modality clipping; labels untouched; single-process.",
        dataset_license="Research; follow ISLES terms.",
        dataset_author=os.environ.get("USER","author")
    )
    print(f"✓ Done. Dataset written to: {out_base}")

if __name__ == "__main__":
    main()
