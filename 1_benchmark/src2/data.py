"""
Dataset loaders for Experiment 2.

CheXpert Plus benchmark : 1050 samples (500 PTX + 550 normal), images .png
CheXpert v1 benchmark   : 1050 samples (500 PTX + 550 normal), images .jpg

Both metadata CSVs use Windows-style paths (E:\\...) in path_png / path_jpg.
The loaders convert these to server paths using image_root from server.yaml.
"""

from __future__ import annotations

from pathlib import Path, PureWindowsPath

import pandas as pd
from PIL import Image


# ── Image loading ──────────────────────────────────────────────────────────────

def load_image(image_path: str | Path) -> Image.Image:
    p = Path(image_path)
    if p.suffix.lower() in (".dcm", ".dicom"):
        png_path = p.with_suffix(".png")
        if png_path.exists():
            return Image.open(png_path).convert("RGB")
        return _load_dicom(p)
    return Image.open(p).convert("RGB")


def _load_dicom(dcm_path: Path) -> Image.Image:
    import pydicom
    import numpy as np
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(float)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    vmin, vmax = arr.min(), arr.max()
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin) * 255
    return Image.fromarray(arr.astype("uint8")).convert("RGB")


# ── Path conversion ────────────────────────────────────────────────────────────

def _win_to_server(win_path: str, image_root: Path, extension: str) -> str:
    """
    Convert a Windows path like E:\\chexpertplus_benchmark\\images\\label\\file.png
    to {image_root}/chexpertplus_benchmark/images/{label}/{file}.{ext}
    """
    wp = PureWindowsPath(win_path)
    # wp.parts = ('E:\\', 'chexpertplus_benchmark', 'images', 'label', 'file.png')
    rel_parts = wp.parts[1:]  # drop drive letter
    return str(image_root / Path(*rel_parts))


# ── CheXpert Plus benchmark ────────────────────────────────────────────────────

def load_chexpert_plus_benchmark(
    data_root: str | Path,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load CheXpert Plus benchmark subset (1050 samples).

    data_root: directory containing chexpertplus_benchmark/ subfolder.
      Metadata: {data_root}/chexpertplus_benchmark/benchmark_metadata.csv
      Images  : {data_root}/chexpertplus_benchmark/images/{label}/*.png

    Returns DataFrame with columns:
        file_id, image_path, label (0/1), section_impression,
        age, sex, ap_pa, patient_id
    """
    data_root = Path(data_root)
    metadata_csv = data_root / "chexpertplus_benchmark" / "benchmark_metadata.csv"
    df = pd.read_csv(metadata_csv, encoding="utf-8-sig")

    df["file_id"] = df["path_png"].apply(lambda p: PureWindowsPath(p).stem)
    df["image_path"] = df["path_png"].apply(
        lambda p: _win_to_server(p, data_root, "png")
    )
    df["label"] = df["ptx_binary"].astype(int)
    df["section_impression"] = ""  # not available in benchmark metadata

    df = df[["file_id", "image_path", "label", "section_impression",
             "age", "sex", "ap_pa", "patient_id"]].copy()
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ── CheXpert v1 benchmark ──────────────────────────────────────────────────────

def load_chexpert_v1_benchmark(
    data_root: str | Path,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load CheXpert v1 benchmark subset (1050 samples).

    data_root: directory containing chexpert_v1_benchmark/ subfolder.
      Metadata: {data_root}/chexpert_v1_benchmark/benchmark_metadata.csv
      Images  : {data_root}/chexpert_v1_benchmark/images/{label}/*.jpg

    Returns DataFrame with columns:
        file_id, image_path, label (0/1), section_impression,
        age, sex, ap_pa, patient_id
    """
    data_root = Path(data_root)
    metadata_csv = data_root / "chexpert_v1_benchmark" / "benchmark_metadata.csv"
    df = pd.read_csv(metadata_csv, encoding="utf-8-sig")

    df["file_id"] = df["path_jpg"].apply(lambda p: PureWindowsPath(p).stem)
    df["image_path"] = df["path_jpg"].apply(
        lambda p: _win_to_server(p, data_root, "jpg")
    )
    df["label"] = df["ptx_binary"].astype(int)
    df["section_impression"] = ""  # CheXpert v1 has no report text

    df = df[["file_id", "image_path", "label", "section_impression",
             "age", "sex", "ap_pa", "patient_id"]].copy()
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ── Summary ────────────────────────────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame, name: str = "Dataset") -> None:
    print(f"{name}: {len(df)} samples")
    print(f"  PTX positive : {(df.label == 1).sum()}")
    print(f"  PTX negative : {(df.label == 0).sum()}")
    ap = (df.ap_pa == "AP").sum()
    pa = (df.ap_pa == "PA").sum()
    print(f"  AP           : {ap}  |  PA: {pa}")
    male = (df.sex.str.lower() == "male").sum()
    female = (df.sex.str.lower() == "female").sum()
    print(f"  Male         : {male}  |  Female: {female}")
