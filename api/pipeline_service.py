"""ZIP-of-DICOMs → preprocessed per-disc crops.

Wraps utils.functions and re-implements pipeline.py's crop helpers (we don't
import pipeline.py because it runs main() at module load).
"""
from __future__ import annotations

import hashlib
import inspect
import io
import re
import shutil
import subprocess
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from numpy import ndarray

# on_step(step_id, status, elapsed_ms, cached?) — status ∈ {"start", "done"}.
# Older callers without the `cached` kwarg are supported via the wrapper used
# inside the module (see `_call_on_step`).
StepCallback = Callable[..., None]

CACHE_ROOT = Path(__file__).resolve().parent / "cache"


def _case_workdir(case_id: str) -> Path:
    p = CACHE_ROOT / case_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _call_on_step(
    on_step: StepCallback,
    step_id: str,
    status: str,
    elapsed_ms: Optional[float],
    cached: bool = False,
) -> None:
    """Invoke ``on_step`` with the ``cached`` kwarg when supported, else drop it."""
    try:
        sig = inspect.signature(on_step)
        if "cached" in sig.parameters or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            on_step(step_id, status, elapsed_ms, cached=cached)
            return
    except (TypeError, ValueError):
        pass
    on_step(step_id, status, elapsed_ms)

from utils.functions import (
    convert_case_to_nifti,
    ensure_dirs,
    get_t2_sagittals_only,
    load_volume_and_segmentation,
    resize_with_padding,
    clamp,
)


def _run_totalspineseg(input_nii: Path, output_dir: Path) -> Path:
    """Call totalspineseg with captured output and server-safe flags.

    The version in utils.functions runs with -q and inherits stdout/stderr,
    which makes its failures invisible to the FastAPI request that triggered
    them. We capture both streams and surface them in the exception.
    """
    cmd = [
        "totalspineseg",
        str(input_nii),
        str(output_dir),
        "--no-stalling",
        "--max-workers", "4",
        "--max-workers-nnunet", "2",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-4000:]
        raise RuntimeError(
            f"totalspineseg failed (exit {proc.returncode}). "
            f"cmd: {' '.join(cmd)}\n--- output tail ---\n{tail}"
        )
    return output_dir / "step2_output" / f"{input_nii.name}.gz"


DISC_LABELS: List[int] = [92, 93, 94, 95, 100]
LABEL_TO_LEVEL: Dict[int, str] = {
    92: "L1_L2",
    93: "L2_L3",
    94: "L3_L4",
    95: "L4_L5",
    100: "L5_S",
}

DEFAULT_PADS = {
    "pad_x": (8, 8),
    "pad_y": (5, 5),
    "pad_z": (0, 0),
}


@dataclass
class CaseData:
    case_id: str
    volume: ndarray              # (Z, Y, X) float
    segmentation: ndarray        # (Z, Y, X) int/float, same shape as volume
    disc_crops: Dict[int, ndarray]  # label → 224x224 uint8 mid-slice crop
    workdir: Path                # caller may delete
    patient: Optional[Dict[str, str]] = field(default=None)


def _format_patient_name(name_obj: Any) -> str:
    if name_obj is None:
        return "—"
    family = (getattr(name_obj, "family_name", "") or "").strip()
    given = (getattr(name_obj, "given_name", "") or "").strip()
    if family or given:
        return f"{given} {family}".strip()
    raw = str(name_obj).replace("^", " ").strip()
    return raw or "—"


def _format_patient_age(raw: str) -> str:
    if not raw:
        return "—"
    m = re.match(r"\s*0*(\d+)\s*([YMWD])", raw.upper())
    if not m:
        return raw.strip() or "—"
    n = int(m.group(1))
    unit = {"Y": "y", "M": "mo", "W": "w", "D": "d"}[m.group(2)]
    return f"{n} {unit}"


def _format_patient_sex(raw: str) -> str:
    s = (raw or "").strip().upper()
    return {"M": "Male", "F": "Female", "O": "Other"}.get(s, s or "—")


def extract_patient_info(dcm: Any) -> Dict[str, str]:
    """Pull patient name / age / sex from a pydicom FileDataset, formatted for UI."""
    return {
        "name": _format_patient_name(getattr(dcm, "PatientName", None)),
        "age": _format_patient_age(str(getattr(dcm, "PatientAge", "") or "")),
        "sex": _format_patient_sex(str(getattr(dcm, "PatientSex", "") or "")),
    }


# --------------------------------------------------------------------------- #
# crop helpers — copied verbatim from pipeline.py (lines 20-72) to avoid the
# side-effecting import.
# --------------------------------------------------------------------------- #
def crop_single_label(
    label: int,
    segmentation_data: ndarray,
    volume_data: ndarray,
    *,
    pad_x: Tuple[int, int] = (0, 0),
    pad_y: Tuple[int, int] = (0, 0),
    pad_z: Tuple[int, int] = (0, 0),
):
    assert segmentation_data.shape == volume_data.shape, "Shapes must match"

    coords = np.argwhere(segmentation_data == label)
    if coords.size == 0:
        return None

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    z_min, z_max = clamp(z_min - pad_z[0], z_max + pad_z[1], volume_data.shape[0])
    y_min, y_max = clamp(y_min - pad_y[0], y_max + pad_y[1], volume_data.shape[1])
    x_min, x_max = clamp(x_min - pad_x[0], x_max + pad_x[1], volume_data.shape[2])

    cropped = volume_data[
        z_min:z_max + 1,
        y_min:y_max + 1,
        x_min:x_max + 1,
    ]
    return cropped, cropped.shape


def crop_multiple_labels(
    labels: Iterable[int],
    segmentation_data: ndarray,
    volume_data: ndarray,
    **pads,
) -> Dict[int, Tuple[ndarray, Tuple[int, int, int]]]:
    out: Dict[int, Tuple[ndarray, Tuple[int, int, int]]] = {}
    for label in labels:
        cropped = crop_single_label(label, segmentation_data, volume_data, **pads)
        if cropped is not None:
            out[label] = cropped
    return out


# --------------------------------------------------------------------------- #
# ZIP unpacking
# --------------------------------------------------------------------------- #
def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _find_case_root(extract_root: Path) -> Path:
    """Find the directory that contains the actual DICOM files.

    Accepts both 'caseX/*.dcm' and root-level '*.dcm' layouts. Picks the
    directory with the largest number of files (which will be the DICOM dir).
    """
    candidates: List[Tuple[int, Path]] = []
    for p in extract_root.rglob("*"):
        if p.is_dir():
            n_files = sum(1 for f in p.iterdir() if f.is_file())
            if n_files > 0:
                candidates.append((n_files, p))
    if not candidates:
        if any(p.is_file() for p in extract_root.iterdir()):
            return extract_root
        raise ValueError("ZIP contains no files")
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _make_mid_slice_crop(cropped: ndarray) -> ndarray:
    """Replicates pipeline.save_result lines 209-217: pick mid Z slice,
    rotate to a canonical sagittal view, pad to 224×224.
    """
    shape = cropped.shape
    mid = cropped[shape[0] // 2]
    rotated = np.rot90(np.flipud(mid))
    return resize_with_padding(rotated, target_size=(224, 224), pad_color=200)


@contextmanager
def _step(on_step: StepCallback, step_id: str):
    _call_on_step(on_step, step_id, "start", None)
    t0 = time.time()
    success = False
    try:
        yield
        success = True
    finally:
        if success:
            _call_on_step(on_step, step_id, "done", (time.time() - t0) * 1000.0)


def _emit_cached_step(on_step: StepCallback, step_id: str) -> None:
    """Emit a single done event marking ``step_id`` as cached."""
    _call_on_step(on_step, step_id, "done", 0.0, cached=True)


def _noop_step(*args, **kwargs) -> None:
    pass


def preprocess_case(zip_bytes: bytes, on_step: StepCallback = _noop_step) -> CaseData:
    """Unzip → t2 sagittals → NIfTI → totalspineseg → crops.

    Emits step events via ``on_step(step_id, status, elapsed_ms, cached=)``.
    Step IDs: ``unzip``, ``filter_t2``, ``dicom_to_nifti``, ``segmentation``,
    ``crop_discs``.

    Persists per-case artifacts under ``CACHE_ROOT/<case_id>/`` so that
    re-uploading the same ZIP skips unzip→segmentation entirely.
    """
    case_id = _hash_bytes(zip_bytes)
    workdir = _case_workdir(case_id)
    nii_dir = workdir / "volumes"
    seg_dir = workdir / "segmentation"
    nii_path = nii_dir / "volumes.nii"
    seg_path = seg_dir / "step2_output" / "volumes.nii.gz"
    cache_hit = nii_path.exists() and seg_path.exists()
    patient: Optional[Dict[str, str]] = None

    try:
        if cache_hit:
            for step_id in ("unzip", "filter_t2", "dicom_to_nifti", "segmentation"):
                _emit_cached_step(on_step, step_id)
        else:
            with _step(on_step, "unzip"):
                extract_root = workdir / "extracted"
                if extract_root.exists():
                    shutil.rmtree(extract_root)
                extract_root.mkdir(parents=True)
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                    zf.extractall(extract_root)
                case_root = _find_case_root(extract_root)
                if nii_dir.exists():
                    shutil.rmtree(nii_dir)
                if seg_dir.exists():
                    shutil.rmtree(seg_dir)
                ensure_dirs(nii_dir, seg_dir)

            with _step(on_step, "filter_t2"):
                sag_dcm_objects = get_t2_sagittals_only(case_root)
                if not sag_dcm_objects:
                    raise ValueError(
                        "No T2 sagittal DICOM series found "
                        "(expected SeriesDescription matching r'^t2_.+_sag_384')."
                    )
                # Capture patient demographics once, from the first DICOM. They
                # do not survive the NIfTI conversion, so we persist them
                # alongside the case metadata in case_store.save_case().
                patient = extract_patient_info(sag_dcm_objects[0])

            with _step(on_step, "dicom_to_nifti"):
                nii_path = convert_case_to_nifti(
                    dcm_objects=sag_dcm_objects, output_dir=nii_dir
                )

            with _step(on_step, "segmentation"):
                seg_path = _run_totalspineseg(nii_path, seg_dir)

        with _step(on_step, "crop_discs"):
            volume, segmentation = load_volume_and_segmentation(nii_path, seg_path)

            cropped = crop_multiple_labels(
                DISC_LABELS,
                segmentation,
                volume,
                **DEFAULT_PADS,
            )
            if not cropped:
                raise ValueError(
                    f"Segmentation contained none of the expected disc labels {DISC_LABELS}"
                )

            disc_crops: Dict[int, ndarray] = {}
            for label, (img, _shape) in cropped.items():
                disc_crops[label] = _make_mid_slice_crop(img)

        return CaseData(
            case_id=case_id,
            volume=volume,
            segmentation=segmentation,
            disc_crops=disc_crops,
            workdir=workdir,
            patient=patient,
        )
    except Exception:
        # Keep partially-cached artifacts on disk on failure so retries don't
        # need to redo successful steps; the user can rm cache/<case_id>/ to
        # force a clean re-run.
        raise


def cleanup(case: CaseData) -> None:  # kept for compatibility; no-op now
    return None
