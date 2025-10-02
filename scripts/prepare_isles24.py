# #!/usr/bin/env python3
# import argparse, os, re, shutil
# from pathlib import Path
# from typing import List, Tuple

# # -------- Patterns (match your screenshot) --------
# TMAX_PATTERN  = "_space-ncct_tmax.nii.gz"   # perfusion-maps/*tmax.nii.gz
# LABEL_PATTERN = "_lesion-msk.nii.gz"        # annotation mask
# # --------------------------------------------------

# TRAIN_FRACTION = 0.8
# SHUFFLE = False
# RANDOM_SEED = 42

# def extract_subject_id(p: Path) -> str | None:
#     """
#     Get the numeric id '0001' from names/dirs like:
#       sub-stroke0001_ses-01_space-ncct_tmax.nii.gz
#       .../sub-stroke0001/ses-02/..._lesion-msk.nii.gz
#     """
#     m = re.search(r"sub-stroke(\d+)", p.name)
#     if not m:
#         for part in p.parts[::-1]:
#             m = re.search(r"sub-stroke(\d+)", part)
#             if m: break
#     return m.group(1).zfill(4) if m else None

# def find_pairs(roots: List[Path]) -> List[Tuple[str, Path, Path]]:
#     """
#     Return (subject_id, tmax_path, label_path). Only keep subjects having both.
#     """
#     idx = {}
#     for root in roots:
#         # search anywhere under derivatives/ for perfusion-maps and labels
#         for p in root.rglob(f"*{TMAX_PATTERN}"):
#             sid = extract_subject_id(p);  
#             if not sid: continue
#             rec = idx.setdefault(sid, {"tmax": None, "label": None})
#             rec["tmax"] = p
#         for p in root.rglob(f"*{LABEL_PATTERN}"):
#             sid = extract_subject_id(p);  
#             if not sid: continue
#             rec = idx.setdefault(sid, {"tmax": None, "label": None})
#             rec["label"] = p

#     out = []
#     for sid, rec in idx.items():
#         if rec["tmax"] is not None and rec["label"] is not None:
#             out.append((sid, rec["tmax"], rec["label"]))
#     return out

# def cp_or_mv(src: Path, dst: Path, move: bool):
#     dst.parent.mkdir(parents=True, exist_ok=True)
#     (shutil.move if move else shutil.copy2)(src, dst)

# def main():
#     ap = argparse.ArgumentParser(description="Create nnU-Netv2 dataset from ISLES24 (Tmax + labels).")
#     ap.add_argument("--in-roots", nargs="+", required=True,
#                     help="Roots like: /data/isles24_train-1 /data/isles24_train-2 (searches recursively)")
#     ap.add_argument("--out-root", required=True,
#                     help="Output dataset root, e.g. $nnUNet_raw/Dataset000_ISLES24_CTA")
#     ap.add_argument("--train-frac", type=float, default=TRAIN_FRACTION)
#     ap.add_argument("--shuffle", action="store_true", default=SHUFFLE)
#     ap.add_argument("--move", action="store_true", help="Move instead of copy")
#     args = ap.parse_args()

#     roots = [Path(p).expanduser().resolve() for p in args.in_roots]
#     out_root = Path(args.out_root).expanduser().resolve()
#     imagesTr = out_root / "imagesTr"; labelsTr = out_root / "labelsTr"; imagesTs = out_root / "imagesTs"
#     for d in (imagesTr, labelsTr, imagesTs): d.mkdir(parents=True, exist_ok=True)

#     pairs = find_pairs(roots)
#     if not pairs:
#         raise SystemExit(f"No (Tmax,label) pairs found. Patterns: TMAX='{TMAX_PATTERN}', LABEL='{LABEL_PATTERN}'")

#     # stable order by subject id; optional shuffle
#     pairs.sort(key=lambda t: int(t[0]))
#     if args.shuffle:
#         import random; random.seed(RANDOM_SEED); random.shuffle(pairs)

#     n = len(pairs); n_tr = int(round(n * args.train_frac))
#     train_pairs, test_pairs = pairs[:n_tr], pairs[n_tr:]

#     # --- write train (Tmax -> channel 0) ---
#     for sid, tmax_p, lab_p in train_pairs:
#         cp_or_mv(tmax_p, imagesTr / f"case_{sid}_0000.nii.gz", args.move)
#         cp_or_mv(lab_p,  labelsTr / f"case_{sid}.nii.gz",      args.move)

#     # --- write test (Tmax only) ---
#     for sid, tmax_p, _ in test_pairs:
#         cp_or_mv(tmax_p, imagesTs / f"case_{sid}_0000.nii.gz", args.move)

#     print(f"Done. Subjects: {n} | train: {len(train_pairs)} | test: {len(test_pairs)}")
#     print(f"Dataset root: {out_root}")
#     print(f" e.g., {imagesTr/'case_0001_0000.nii.gz'} ; {labelsTr/'case_0001.nii.gz'}")

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISLES'24 → nnU-Net v2 (multimodal) with skull-strip + per-modality clipping (single-process).

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
  imagesTr/case_<ID>_0000.nii.gz   (ncct, float32, brain-masked, clipped 0–90)
  imagesTr/case_<ID>_0001.nii.gz   (cta,  float32, mask, 0–90)
  imagesTr/case_<ID>_0002.nii.gz   (cbv,  float32, mask, 0–10)
  imagesTr/case_<ID>_0003.nii.gz   (cbf,  float32, mask, 0–35)
  imagesTr/case_<ID>_0004.nii.gz   (mtt,  float32, mask, 0–20)
  imagesTr/case_<ID>_0005.nii.gz   (tmax, float32, mask, 0–7)
  labelsTr/case_<ID>.nii.gz        (uint8, background=0, lesion=1)
  imagesTs/...                      (same channels; no labels)
  dataset.json

Notes:
- No multiprocessing; purely sequential for reliability.
- Labels are only NN-resampled to NCCT grid if needed (never altered otherwise).
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
    # NCCT from RAW (reference grid)
    "ncct": [
        "**/raw_data/sub-stroke*/ses-01/*_ses-01_ncct.nii.gz",        # e.g. .../raw_data/sub-stroke0001/ses-01/sub-stroke0001_ses-01_ncct.nii.gz
        "**/raw_data/sub-stroke*/ses-01/*_ses-01_*ncct.nii.gz",       # fallback variant
        "**/derivatives/sub-stroke*/ses-01/*space-ncct_ncct.nii.gz",  # rare preprocessed NCCT (fallback)
    ],
    # CTA in derivatives/ses-01
    "cta":  ["**/derivatives/sub-stroke*/ses-01/*_ses-01_space-ncct_cta.nii.gz"],
    # Perfusion maps in derivatives/ses-01/perfusion-maps
    "cbf":  ["**/derivatives/sub-stroke*/ses-01/perfusion-maps/*_ses-01_space-ncct_cbf.nii.gz"],
    "cbv":  ["**/derivatives/sub-stroke*/ses-01/perfusion-maps/*_ses-01_space-ncct_cbv.nii.gz"],
    "mtt":  ["**/derivatives/sub-stroke*/ses-01/perfusion-maps/*_ses-01_space-ncct_mtt.nii.gz"],
    "tmax": ["**/derivatives/sub-stroke*/ses-01/perfusion-maps/*_ses-01_space-ncct_tmax.nii.gz"],
    # Lesion label in derivatives/ses-02
    "label":["**/derivatives/sub-stroke*/ses-02/*_ses-02_lesion-msk.nii.gz"],
}

# ---------- intensity windows ----------
WINDOWS = {
    "ncct": (0.0, 90.0),
    "cta":  (0.0, 90.0),
    "cbv":  (0.0, 10.0),
    "cbf":  (0.0, 35.0),
    "mtt":  (0.0, 20.0),
    "tmax": (0.0, 7.0),
}

# nnU-Net channel ordering
# CHANNEL_ORDER = ["ncct", "cta", "cbv", "cbf", "mtt", "tmax"]  # -> _0000.._0005
CHANNEL_ORDER = ["cta","cbv", "cbf", "mtt", "tmax"]

# ---------- utilities ----------
SUBJ_PAT = re.compile(r"sub-stroke(\d+)")
ID4 = lambda x: str(int(x)).zfill(4)

def extract_subject_id(p: Path) -> Optional[str]:
    m = SUBJ_PAT.search(p.name)
    if not m:
        for part in p.parts[::-1]:
            m = SUBJ_PAT.search(part)
            if m: break
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

def skull_strip_from_ncct(ncct: sitk.Image,
                          clamp=(-100, 200), soft=(-15,100),
                          close_mm=3.0, erode_mm=3.0,
                          skull_hu=250, shave_mm=1.5) -> sitk.Image:
    sp = ncct.GetSpacing()
    clamped = sitk.Clamp(ncct, lowerBound=clamp[0], upperBound=clamp[1])
    softmask = sitk.BinaryThreshold(clamped, lowerThreshold=soft[0], upperThreshold=soft[1],
                                    insideValue=1, outsideValue=0)
    softmask = sitk.BinaryMorphologicalClosing(softmask, mm_to_radius(close_mm, sp))
    softmask = sitk.VotingBinaryIterativeHoleFilling(
        softmask, radius=mm_to_radius(2.0, sp), majorityThreshold=1,
        backgroundValue=0, foregroundValue=1, maximumNumberOfIterations=1
    )
    cc = sitk.ConnectedComponent(softmask)
    stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(cc)
    if not stats.GetLabels():
        raise RuntimeError("Skull-strip: no components found – adjust thresholds.")
    largest = max(stats.GetLabels(), key=lambda l: stats.GetPhysicalSize(l))
    brain = sitk.BinaryThreshold(cc, lowerThreshold=largest, upperThreshold=largest,
                                 insideValue=1, outsideValue=0)
    skull = sitk.BinaryThreshold(ncct, lowerThreshold=skull_hu, upperThreshold=3000,
                                 insideValue=1, outsideValue=0)
    brain = sitk.And(brain, sitk.BinaryNot(sitk.BinaryDilate(skull, mm_to_radius(shave_mm, sp))))
    brain = sitk.BinaryErode(brain, mm_to_radius(erode_mm, sp))
    brain = sitk.BinaryMorphologicalClosing(brain, mm_to_radius(2.0, sp))
    cc2 = sitk.ConnectedComponent(brain)
    stats2 = sitk.LabelShapeStatisticsImageFilter(); stats2.Execute(cc2)
    largest2 = max(stats2.GetLabels(), key=lambda l: stats2.GetPhysicalSize(l))
    brain = sitk.BinaryThreshold(cc2, lowerThreshold=largest2, upperThreshold=largest2,
                                 insideValue=1, outsideValue=0)
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

    # NCCT as reference grid
    ncct = sitk.ReadImage(str(rec["ncct"]))
    brain = skull_strip_from_ncct(ncct)
    if mask_dir:
        mask_dir.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(brain, str(mask_dir / f"case_{sid}_brainmask.nii.gz"))

    # write channels
    for ch, key in enumerate(CHANNEL_ORDER):
        img = sitk.ReadImage(str(rec[key]))
        if (img.GetSize()!=ncct.GetSize() or img.GetSpacing()!=ncct.GetSpacing() or
            img.GetDirection()!=ncct.GetDirection() or img.GetOrigin()!=ncct.GetOrigin()):
            img = resample_like(img, ncct, sitk.sitkLinear, default=0.0)
        vmin, vmax = WINDOWS[key]
        out = clip_and_mask(img, brain, vmin, vmax)
        sitk.WriteImage(out, str(out_images / f"case_{sid}_{ch:04d}.nii.gz"))

    # write label (uint8), NN-resampled if needed
    lab = sitk.ReadImage(str(rec["label"]))
    if (lab.GetSize()!=ncct.GetSize() or lab.GetSpacing()!=ncct.GetSpacing() or
        lab.GetDirection()!=ncct.GetDirection() or lab.GetOrigin()!=ncct.GetOrigin()):
        lab = resample_like(lab, ncct, sitk.sitkNearestNeighbor, default=0)
    lab = sitk.Cast(lab>0, sitk.sitkUInt8)
    sitk.WriteImage(lab, str(out_labels / f"case_{sid}.nii.gz"))
    return True

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="ISLES'24 → nnU-Net v2 (ncct+cta+cbv+cbf+mtt+tmax), skull-strip & clipping (single-process).")
    ap.add_argument("--in-roots", nargs="+", required=True,
                    help="Top-level ISLES roots, e.g. /bhome/test/isles24_train-1 /bhome/test/isles24_train-2/isles24_train_b")
    ap.add_argument("--dataset-id", type=int, required=True, help="e.g., 010")
    ap.add_argument("--dataset-name", default="ISLES24_multi", help="Dataset nickname.")
    ap.add_argument("--out-root", default=None,
                    help="Explicit output root (else uses $nnUNet_raw/Dataset{ID}_{NAME})")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--shuffle", action="store_true", help="Shuffle subjects before split.")
    ap.add_argument("--save-brainmasks", action="store_true", help="Save brain masks for QA.")
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
    for d in (out_imagesTr, out_labelsTr, out_imagesTs): d.mkdir(parents=True, exist_ok=True)
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

    # process TEST sequentially (images only)
    for sid in test_sids:
        try:
            rec = index[sid]
            ncct = sitk.ReadImage(str(rec["ncct"]))
            brain = skull_strip_from_ncct(ncct)
            if mask_dir:
                sitk.WriteImage(brain, str(mask_dir / f"case_{sid}_brainmask.nii.gz"))
            for ch, key in enumerate(CHANNEL_ORDER):
                img = sitk.ReadImage(str(rec[key]))
                if (img.GetSize()!=ncct.GetSize() or img.GetSpacing()!=ncct.GetSpacing() or
                    img.GetDirection()!=ncct.GetDirection() or img.GetOrigin()!=ncct.GetOrigin()):
                    img = resample_like(img, ncct, sitk.sitkLinear, default=0.0)
                vmin, vmax = WINDOWS[key]
                out = clip_and_mask(img, brain, vmin, vmax)
                sitk.WriteImage(out, str(out_imagesTs / f"case_{sid}_{ch:04d}.nii.gz"))
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
        overwrite_image_reader_writer="NibabelIOWithReorient",
        dataset_description="Brain-masked & clipped per modality; labels untouched; single-process.",
        dataset_license="Research; follow ISLES terms.",
        dataset_author=os.environ.get("USER","author")
    )
    print(f"✓ Done. Dataset written to: {out_base}")

if __name__ == "__main__":
    main()