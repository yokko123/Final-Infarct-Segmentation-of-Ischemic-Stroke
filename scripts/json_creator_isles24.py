#!/usr/bin/env python3
"""
ISLES24 → nnU-Net v2 dataset (Tmax) + dataset.json
Accepts source names like:
  imagesTr/ tmax_0113.nii.gz  OR  case_0113_0000.nii.gz  OR anything_*0113[_000k].nii.gz
  labelsTr/ label_0113.nii.gz OR  case_0113.nii.gz       OR tmax_0113.nii.gz
Writes:
  imagesTr/ case_0113_0000.nii.gz
  labelsTr/ case_0113.nii.gz
"""

from __future__ import annotations
import argparse, os, re, shutil
from pathlib import Path

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

# -------- helpers --------
IDX_PAT = re.compile(r"(\d{4})(?=(?:_000\d)?\.nii\.gz$)")  # grab #### before optional _000k + .nii.gz
CHAN_PAT = re.compile(r"(.+?)_000\d$")                     # strip channel suffix from stem

def _copy_or_move(src: Path, dst: Path, move: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    (shutil.move if move else shutil.copy2)(str(src), str(dst))

def _stem_no_channel(p: Path) -> str:
    stem = p.name[:-7]  # drop .nii.gz
    m = CHAN_PAT.match(stem)
    return m.group(1) if m else stem

def _find_idx(name: str) -> str | None:
    m = IDX_PAT.search(name)
    return m.group(1) if m else None

def _normalize_and_copy_split(src_root: Path, dst_root: Path, move: bool = False) -> int:
    imagesTr_src = src_root / "imagesTr"
    labelsTr_src = src_root / "labelsTr"
    imagesTs_src = src_root / "imagesTs"

    imagesTr_dst = dst_root / "imagesTr"
    labelsTr_dst = dst_root / "labelsTr"
    imagesTs_dst = dst_root / "imagesTs"
    for d in (imagesTr_dst, labelsTr_dst, imagesTs_dst):
        d.mkdir(parents=True, exist_ok=True)

    # TRAIN
    n_tr = 0
    for img in sorted(imagesTr_src.glob("*.nii.gz")):
        idx = _find_idx(img.name)
        if not idx:
            print(f"[WARN] Skip (cannot parse case id): {img.name}")
            continue

        # base name we will use consistently
        case_base = f"case_{idx}"

        # candidate labels in order of preference
        cand_labels = [
            labelsTr_src / f"{case_base}.nii.gz",
            labelsTr_src / f"label_{idx}.nii.gz",
            labelsTr_src / f"tmax_{idx}.nii.gz",
        ]
        lab = next((p for p in cand_labels if p.exists()), None)
        if lab is None:
            print(f"[WARN] Missing label for {img.name}; looked for: "
                  f"{', '.join(p.name for p in cand_labels)}")
            continue

        # destination names
        img_dst = imagesTr_dst / f"{case_base}_0000.nii.gz"
        lab_dst = labelsTr_dst / f"{case_base}.nii.gz"

        _copy_or_move(img, img_dst, move)
        _copy_or_move(lab, lab_dst, move)
        n_tr += 1

    # TEST (images only)
    if imagesTs_src.exists():
        for img in sorted(imagesTs_src.glob("*.nii.gz")):
            idx = _find_idx(img.name)
            if not idx:
                print(f"[WARN] Skip test (cannot parse case id): {img.name}")
                continue
            img_dst = imagesTs_dst / f"case_{idx}_0000.nii.gz"
            _copy_or_move(img, img_dst, move)

    return n_tr

def main():
    ap = argparse.ArgumentParser(description="Create nnU-Net v2 dataset (Tmax) + dataset.json.")
    ap.add_argument("--source", required=True, help="Folder with imagesTr/, labelsTr/, imagesTs/")
    ap.add_argument("--dataset-id", type=int, required=True, help="e.g., 000")
    ap.add_argument("--dataset-name", type=str, default="ISLES24_Tmax")
    ap.add_argument("--move", action="store_true", help="Move instead of copy")
    args = ap.parse_args()

    if not nnUNet_raw:
        raise SystemExit("nnUNet_raw env var is not set (nnunetv2.paths.nnUNet_raw).")

    src_root = Path(args.source).expanduser().resolve()
    if not (src_root / "imagesTr").exists() or not (src_root / "labelsTr").exists():
        raise SystemExit(f"{src_root} must contain imagesTr/ and labelsTr/ (and optionally imagesTs/).")

    out_base = Path(nnUNet_raw) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        (out_base / sub).mkdir(parents=True, exist_ok=True)

    print(f"→ Preparing dataset at: {out_base}")
    n_tr = _normalize_and_copy_split(src_root, out_base, move=args.move)
    print(f"→ Training cases prepared: {n_tr}")

    # dataset.json
    generate_dataset_json(
        output_folder=str(out_base),
        channel_names={0: "tmax"},
        labels={"background": 0, "lesion": 1},
        num_training_cases=n_tr,
        file_ending=".nii.gz",
        regions_class_order=None,
        dataset_name=f"Dataset{args.dataset_id:03d}_{args.dataset_name}",
        reference="ISLES 2024 (Tmax lesion)",
        release="",
        overwrite_image_reader_writer="NibabelIOWithReorient",
        dataset_description="ISLES24 Tmax to nnU-Net v2 (single-channel).",
        dataset_license="Research only; respect original terms.",
        dataset_author=os.environ.get("USER", "author")
    )
    print("✓ dataset.json written. nnU-Net v2 raw dataset is ready.")

if __name__ == "__main__":
    main()
