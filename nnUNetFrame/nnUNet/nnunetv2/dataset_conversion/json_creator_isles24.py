"""
Robust nnU-Net v2 JSON creator for ISLES-style data.

This version:
- Prints verbose progress so you SEE output.
- Accepts multiple filename patterns in source:
    imagesTr:  ctp_0001.nii.gz  OR  sub-stroke0001_*ctp*.nii.gz  OR any with a 4-digit id
    labelsTr:  label_0001.nii.gz OR *lesion-msk*.nii.gz OR any with same 4-digit id
- Renames to nnU-Net expected names:
    imagesTr/case_0001_0000.nii.gz
    labelsTr/case_0001.nii.gz
    imagesTs/case_00XX_0000.nii.gz
- Generates dataset.json
"""

from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import re
from typing import Optional, Dict, Tuple

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


IDX_PATTERNS = [
    re.compile(r'.*?(\d{4})\.nii\.gz$'),                 # ...0001.nii.gz
    re.compile(r'sub-stroke(\d+)', re.IGNORECASE),       # sub-stroke0001
]

def extract_idx(p: Path) -> Optional[str]:
    """Return a zero-padded 4-digit index from filename or parents."""
    name = p.name
    for pat in IDX_PATTERNS:
        m = pat.search(name)
        if m:
            return f"{int(m.group(1)):04d}"
    # try parent parts
    for part in p.parts[::-1]:
        m = re.search(r'(\d{4})', part)
        if m:
            return m.group(1)
        m = re.search(r'sub-stroke(\d+)', part, re.IGNORECASE)
        if m:
            return f"{int(m.group(1)):04d}"
    return None


def map_train_pairs(imagesTr: Path, labelsTr: Path) -> Dict[str, Tuple[Path, Path]]:
    """
    Build a dict idx -> (img_path, label_path)
    labels can be label_0001.nii.gz or *lesion-msk*.nii.gz etc.
    """
    print(f"[INFO] Scanning training images in {imagesTr}")
    imgs = list(imagesTr.glob("*.nii.gz"))
    print(f"[INFO] Found {len(imgs)} training images")

    # index images
    img_by_idx: Dict[str, Path] = {}
    for p in imgs:
        idx = extract_idx(p)
        if idx:
            img_by_idx[idx] = p
        else:
            print(f"[WARN] Could not extract index from image: {p.name}")

    print(f"[INFO] Scanning training labels in {labelsTr}")
    labs = list(labelsTr.glob("*.nii.gz"))
    print(f"[INFO] Found {len(labs)} training labels")

    # index labels (prefer lesion-msk or label_ pattern)
    lab_by_idx: Dict[str, Path] = {}
    for p in labs:
        idx = extract_idx(p)
        if idx:
            lab_by_idx[idx] = p
        else:
            print(f"[WARN] Could not extract index from label: {p.name}")

    # pair up
    pairs: Dict[str, Tuple[Path, Path]] = {}
    for idx, img in sorted(img_by_idx.items(), key=lambda t: int(t[0])):
        lab = lab_by_idx.get(idx)
        if not lab:
            print(f"[WARN] No label for image idx {idx} ({img.name})")
            continue
        pairs[idx] = (img, lab)

    print(f"[INFO] Paired {len(pairs)} training cases")
    return pairs


def copy_or_move(src: Path, dst: Path, move: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description="Create nnU-Net v2 dataset.json (robust)")
    ap.add_argument("--source", required=True, help="Folder with imagesTr, labelsTr, imagesTs")
    ap.add_argument("--dataset-id", type=int, required=True)
    ap.add_argument("--dataset-name", type=str, default="ISLES24_CTP")
    ap.add_argument("--move", action="store_true", help="Move instead of copy")
    args = ap.parse_args()

    src = Path(args.source).expanduser().resolve()
    imagesTr_src = src / "imagesTr"
    labelsTr_src = src / "labelsTr"
    imagesTs_src = src / "imagesTs"
    for d in (imagesTr_src, labelsTr_src, imagesTs_src):
        if not d.exists():
            raise SystemExit(f"[ERROR] Missing folder: {d}")

    dataset_folder_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    out_base = Path(nnUNet_raw) / dataset_folder_name
    imagesTr_dst = out_base / "imagesTr"
    labelsTr_dst = out_base / "labelsTr"
    imagesTs_dst = out_base / "imagesTs"
    for d in (imagesTr_dst, labelsTr_dst, imagesTs_dst):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] nnUNet_raw        : {nnUNet_raw}")
    print(f"[INFO] Dataset out folder: {out_base}")
    print(f"[INFO] Source            : {src}")

    # Pair training
    pairs = map_train_pairs(imagesTr_src, labelsTr_src)

    # Copy/move training with nnU-Net naming
    for k, (img, lab) in pairs.items():
        base = f"case_{k}"
        img_dst = imagesTr_dst / f"{base}_0000.nii.gz"
        lab_dst = labelsTr_dst / f"{base}.nii.gz"
        copy_or_move(img, img_dst, args.move)
        copy_or_move(lab, lab_dst, args.move)
    print(f"[INFO] Wrote {len(pairs)} training image/label pairs")

    # Test images
    ts_imgs = list(imagesTs_src.glob("*.nii.gz"))
    print(f"[INFO] Found {len(ts_imgs)} test images")
    wrote_ts = 0
    for p in ts_imgs:
        idx = extract_idx(p)
        if not idx:
            print(f"[WARN] Could not extract index from test image: {p.name}")
            continue
        img_dst = imagesTs_dst / f"case_{idx}_0000.nii.gz"
        copy_or_move(p, img_dst, args.move)
        wrote_ts += 1
    print(f"[INFO] Wrote {wrote_ts} test images")

    # dataset.json
    channel_names = {0: "CTP"}
    labels = {"background": 0, "lesion": 1}
    generate_dataset_json(
        output_folder=str(out_base),
        dataset_output_folder=str(out_base),
        channel_names=channel_names,
        labels=labels,
        num_training_cases=len(pairs),
        file_ending=".nii.gz",
        regions_class_order=None,
        dataset_name=dataset_folder_name,
        overwrite_image_reader_writer="NibabelIOWithReorient",
        reference="ISLES 2024",
        license="for research use; see original dataset terms"
    )
    print("[INFO] dataset.json written")
    print("[DONE] Ready:", out_base)


if __name__ == "__main__":
    main()