"""
prepare_chaos.py
────────────────
Converts raw CHAOS DICOM files to PNGs and organises them into
the flat train/val structure that CHAOSSeg in data.py expects.

Run this ONCE before crop_datasets.py.

Usage:
    python prepare_chaos.py

Expected input structure (what you have):
    pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT/
        1/DICOM_anon/  ← raw .dcm files
        1/Ground/      ← ground truth .png files
        ...
    pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/MR/
        1/T1DUAL/DICOM_anon/InPhase/
        1/T1DUAL/Ground/
        1/T2SPIR/DICOM_anon/
        1/T2SPIR/Ground/
        ...

Output structure (what CHAOSSeg expects):
    pytorch_data_dir/CHAOS/CT/train/images/ & labels/
    pytorch_data_dir/CHAOS/CT/val/images/   & labels/
    pytorch_data_dir/CHAOS/MR/train/images/ & labels/
    pytorch_data_dir/CHAOS/MR/val/images/   & labels/
"""

import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import pydicom
except ImportError:
    os.system("pip install pydicom -q")
    import pydicom

# ── CONFIG ────────────────────────────────────────────────────────────────────
pytorch_data_dir = "/content/drive/MyDrive/STEGO/src/pytorch_data_dir"
val_patient_ids  = [1, 2]   # these patients go to val, rest go to train
# ─────────────────────────────────────────────────────────────────────────────


def dicom_to_png(dcm_path):
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def save_zero_mask(shape, path):
    Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(path)


def process_patient_slices(tag, dicom_dir, ground_dir, output_root, patient_id, split):
    if not dicom_dir.exists():
        print(f"  Skipping patient {patient_id} {tag} — {dicom_dir} not found")
        return

    dicom_files = sorted(dicom_dir.glob("*.dcm"))
    ground_files = sorted(ground_dir.glob("*.png")) if ground_dir.exists() else []

    print(f"  Patient {patient_id} {tag} → {split} | {len(dicom_files)} slices | {len(ground_files)} masks")

    for i, dcm_path in enumerate(dicom_files):
        out_name = "{}_p{:03d}_s{:04d}.png".format(tag, patient_id, i)

        img = dicom_to_png(dcm_path)
        img.save(output_root / split / "images" / out_name)

        if i < len(ground_files):
            shutil.copy(str(ground_files[i]), str(output_root / split / "labels" / out_name))
        else:
            save_zero_mask(np.array(img.convert("L")).shape, output_root / split / "labels" / out_name)


# ── CT ────────────────────────────────────────────────────────────────────────
print("\n=== Processing CT ===")
ct_input  = Path(pytorch_data_dir) / "archive" / "CHAOS_Train_Sets" / "Train_Sets" / "CT"
ct_output = Path(pytorch_data_dir) / "CHAOS" / "CT"

for split in ["train", "val"]:
    (ct_output / split / "images").mkdir(parents=True, exist_ok=True)
    (ct_output / split / "labels").mkdir(parents=True, exist_ok=True)

for patient_dir in sorted([d for d in ct_input.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda d: int(d.name)):
    patient_id = int(patient_dir.name)
    split = "val" if patient_id in val_patient_ids else "train"
    process_patient_slices(
        "CT",
        patient_dir / "DICOM_anon",
        patient_dir / "Ground",
        ct_output, patient_id, split
    )

# ── MR ────────────────────────────────────────────────────────────────────────
print("\n=== Processing MR ===")
mr_root   = Path(pytorch_data_dir) / "archive" / "CHAOS_Train_Sets" / "Train_Sets" / "MR"
mr_output = Path(pytorch_data_dir) / "CHAOS" / "MR"

for split in ["train", "val"]:
    (mr_output / split / "images").mkdir(parents=True, exist_ok=True)
    (mr_output / split / "labels").mkdir(parents=True, exist_ok=True)

for patient_dir in sorted([d for d in mr_root.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda d: int(d.name)):
    patient_id = int(patient_dir.name)
    split = "val" if patient_id in val_patient_ids else "train"

    process_patient_slices(
        "MR_T1",
        patient_dir / "T1DUAL" / "DICOM_anon" / "InPhase",
        patient_dir / "T1DUAL" / "Ground",
        mr_output, patient_id, split
    )
    process_patient_slices(
        "MR_T2",
        patient_dir / "T2SPIR" / "DICOM_anon",
        patient_dir / "T2SPIR" / "Ground",
        mr_output, patient_id, split
    )

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Done! ===")
for modality in ["CT", "MR"]:
    for split in ["train", "val"]:
        p = Path(pytorch_data_dir) / "CHAOS" / modality / split / "images"
        print(f"  {modality}/{split}: {len(list(p.glob('*.png')))} images")