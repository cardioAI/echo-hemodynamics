"""Batch DICOM conversion utilities for the echo-hemodynamics dataset."""

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from .preprocessing import convert_dicom_to_tensor


def batch_convert(dcm_dir, output_dir, views=("FC", "TC", "SA", "LA")):
    """Batch convert DICOM files to PT tensors. Returns (total, successful, failed)."""
    dcm_dir = Path(dcm_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dcm_files = list(dcm_dir.glob("*.dcm"))

    patient_views = {}
    for dcm_file in dcm_files:
        parts = dcm_file.stem.split("_")
        if len(parts) >= 2:
            patient_id = parts[0]
            view = parts[1]
            if view in views:
                patient_views.setdefault(patient_id, {})[view] = dcm_file

    print(f"\nFound {len(patient_views)} patients")
    print(f"Expected views per patient: {list(views)}")

    total_files = 0
    successful = 0

    for patient_idx, (patient_id, view_files) in enumerate(patient_views.items(), 1):
        print(f"\n[{patient_idx}/{len(patient_views)}] Processing {patient_id}...")

        for view in views:
            if view not in view_files:
                print(f"  WARNING: Missing view {view}")
                continue

            dcm_file = view_files[view]
            output_file = output_dir / f"{patient_id}_{view}.pt"

            try:
                tensor = convert_dicom_to_tensor(dcm_file, target_frames=32, target_size=(224, 224))
                torch.save(tensor, output_file)
                total_files += 1
                successful += 1
                print(f"  {view}: {tensor.shape} -> {output_file.name}")
            except Exception as e:
                print(f"  ERROR processing {view}: {e}")
                total_files += 1

    print(f"\n{'=' * 80}")
    print("Conversion complete:")
    print(f"  Total files processed: {total_files}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {total_files - successful}")
    print(f"{'=' * 80}")

    return total_files, successful, total_files - successful


def reconvert_external_patients():
    """Re-convert External patients with artifact removal and update All_PT folder."""
    print("=" * 80)
    print("EXTREME-CLEAN BATCH RE-CONVERSION: REMOVE ALL ARTIFACTS")
    print("=" * 80)

    dcm_dir = r"E:\dataset_cardioAI\EchoCath_cardioAI\External_DCM"
    external_pt_dir = Path(r"E:\dataset_cardioAI\EchoCath_cardioAI\External_PT")
    all_pt_dir = Path(r"E:\dataset_cardioAI\EchoCath_cardioAI\All_PT")
    claude_dir = Path(r"D:\GoogleDrive\Codes\CardioAI\CardioAI\EchoCath_cardioAI\Claude")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(rf"E:\dataset_cardioAI\EchoCath_cardioAI\External_PT_backup_{timestamp}")

    print("\n1. Creating backup of current External_PT...")
    print(f"   From: {external_pt_dir}")
    print(f"   To:   {backup_dir}")

    if external_pt_dir.exists():
        shutil.copytree(external_pt_dir, backup_dir)
        pt_files_backup = list(backup_dir.glob("*.pt"))
        print(f"   Backup complete: {len(pt_files_backup)} files backed up")

        print("\n2. Clearing External_PT directory...")
        for pt_file in external_pt_dir.glob("*.pt"):
            pt_file.unlink()
        print("   External_PT directory cleared")
    else:
        external_pt_dir.mkdir(parents=True, exist_ok=True)
        print("   External_PT directory created (no backup needed)")

    print("\n3. Starting EXTREME-CLEAN batch conversion...")
    print(f"   Input:  {dcm_dir}")
    print(f"   Output: {external_pt_dir}")
    print("   Expected: 73 patients x 4 views = 292 files")

    batch_convert(dcm_dir, external_pt_dir, views=["FC", "TC", "SA", "LA"])

    print("\n4. Verifying conversion results...")
    pt_files = list(external_pt_dir.glob("*.pt"))
    print(f"   Total PT files created: {len(pt_files)}")
    print("   Expected: 292 files")
    if len(pt_files) == 292:
        print("   SUCCESS: All files converted!")
    else:
        print(f"   WARNING: Expected 292 files, got {len(pt_files)}")

    print("\n5. Updating All_PT folder with extreme-clean versions...")
    external_excel = claude_dir / "External.xlsx"
    external_df = pd.read_excel(external_excel)

    views = ["FC", "TC", "SA", "LA"]
    removed_count = 0
    copied_count = 0

    for _, row in external_df.iterrows():
        patient_id = row["E Number"]
        for view in views:
            old_file = all_pt_dir / f"{patient_id}_{view}.pt"
            if old_file.exists():
                old_file.unlink()
                removed_count += 1
    print(f"   Removed {removed_count} old External files from All_PT")

    print("   Copying new extreme-clean files to All_PT...")
    for _, row in external_df.iterrows():
        patient_id = row["E Number"]
        for view in views:
            src = external_pt_dir / f"{patient_id}_{view}.pt"
            if src.exists():
                dst = all_pt_dir / f"{patient_id}_{view}.pt"
                shutil.copy2(src, dst)
                copied_count += 1
    print(f"   Copied {copied_count} new extreme-clean files to All_PT")

    all_pt_files = list(all_pt_dir.glob("*.pt"))
    print("\n6. Verifying All_PT folder...")
    print(f"   Total PT files in All_PT: {len(all_pt_files)}")
    print("   Expected: 308 x 4 = 1232 files")
    if len(all_pt_files) == 1232:
        print("   SUCCESS: All_PT complete!")
    else:
        print(f"   WARNING: Expected 1232 files, got {len(all_pt_files)}")

    print("\n" + "=" * 80)
    print("EXTREME-CLEAN BATCH RE-CONVERSION COMPLETE")
    print("=" * 80)
