import math
import re
import time
from pathlib import Path
from time import sleep
from typing import Iterable, Callable, Any

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pydicom import FileDataset
import dicom2nifti as dcm2nifti
from pydicom.misc import is_dicom
import pydicom as dcm


def ensure_dirs(*paths: Path|str):
    for p in paths:
        dir = Path(p) if isinstance(p, str) else p
        dir.mkdir(parents=True, exist_ok=True)


def read_dicom_file(file_path: Path | str) -> FileDataset | None:
    if not is_dicom(file_path):
        print("Not a DICOM file")
        return None
    return dcm.dcmread(file_path)


def load_dicom_series(case_path: Path) -> list[FileDataset]:
    return [
        read_dicom_file(p)
        for p in case_path.iterdir()
    ]


def is_t2_sagittal(dcm_obj: FileDataset) -> bool:
    matches = re.search(r"^t2_.+_sag_384", dcm_obj.SeriesDescription, re.IGNORECASE)
    return bool(matches)


def filter_t2_sagittal(dcm_objects: list[FileDataset]) -> list[FileDataset]:
    return [
        d for d in dcm_objects
        if is_t2_sagittal(d)
    ]


def get_t2_sagittals_only(case_path: Path) -> list[FileDataset]:
    return filter_t2_sagittal(load_dicom_series(case_path))


def convert_case_to_nifti(
        dcm_objects: list[FileDataset],
        output_dir: Path
) -> Path:
    output_file = output_dir / "volumes"

    result = dcm2nifti.convert_dicom.dicom_array_to_nifti(
        dicom_list=dcm_objects,
        output_file=output_file,
        reorient_nifti=True
    )

    return Path(str(output_file) + ".nii")


# def run_total_segmentation(input_nii: Path, output_dir: Path):
#     from totalsegmentator.python_api import totalsegmentator as total_segmentator
#     return total_segmentator(
#         input=input_nii,
#         output=output_dir,
#         task="vertebrae_mr",
#     )


def run_spine_segmentation(input_nii: Path, output_dir: Path) -> Path:
    import subprocess

    start = time.time()

    subprocess.run([
        "totalspineseg",
        str(input_nii),
        str(output_dir),
        "--max-workers", "20",
        "--max-workers-nnunet", "10",
        "-q"
    ], check=True)

    end = time.time()
    print(f"Segmentation took {end - start:.4f} seconds")
    sleep(1)

    return output_dir / "step2_output" / f"{input_nii.name}.gz"


def resolve_segmentation(
        nii_path: Path,
        segmentation_case_path: Path,
        *,
        force: bool = False
) -> Path:
    seg_path = segmentation_case_path / "step2_output" / f"{nii_path.name}.gz"

    if seg_path.exists() and not force:
        return seg_path

    return run_spine_segmentation(nii_path, segmentation_case_path)


def load_volume_and_segmentation(
        volume_path: Path,
        segmentation_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    v = nib.load(volume_path)
    s = nib.load(segmentation_path)

    return v.get_fdata(), s.get_fdata()


def clamp(min_val, max_val, upper_bound):
    return max(min_val, 0), min(max_val, upper_bound - 1)


def resize_keep_aspect(image: np.ndarray, target_size: tuple[int, int] | int):
    target_size = (target_size, target_size) if isinstance(target_size, int) else target_size

    target_w, target_h = target_size
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return resized, new_h, new_w


def resize_with_padding(
        image: np.ndarray,
        target_size: tuple[int, int],
        pad_color=0
):
    target_w, target_h = target_size
    resized, new_h, new_w = resize_keep_aspect(image, target_size)
    # Create canvas
    if image.ndim == 2:
        canvas = np.full((target_h, target_w), pad_color, dtype=image.dtype)
    else:
        canvas = np.full((target_h, target_w, image.shape[2]), pad_color, dtype=image.dtype)

    # Center image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def display_images(
        images: np.ndarray | Iterable[np.ndarray],
        *,
        modifier: Callable[[np.ndarray], Any] | None = None,
        titles: list[str] | None = None,
        cols: int = 4,
        cmap: str | None = None,
        figsize: tuple[int, int] | None = None,
):
    if isinstance(images, np.ndarray):
        images = [images]
    else:
        images = list(images)

    n = len(images)
    rows = math.ceil(n / cols)

    if figsize is None:
        figsize = (4 * cols, 3 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Normalize axes
    axes = np.array(axes).reshape(-1)

    for idx, image in enumerate(images):
        if modifier:
            image = modifier(image)

        ax = axes[idx]

        if image.ndim == 2:
            ax.imshow(image, cmap=cmap or "gray")
        else:
            ax.imshow(image)

        title = titles[idx] if titles and idx < len(titles) else f"img_{idx}"
        ax.set_title(title)
        ax.axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
