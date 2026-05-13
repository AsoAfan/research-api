"""On-disk persistence for case history.

Each case lives under ``api/cache/<case_id>/`` and the original preprocessing
artifacts (volumes/, segmentation/, extracted/) live alongside the history
metadata:

- ``metadata.json`` — {case_id, filename, uploaded_at}
- ``result.json``   — the full ``AnalyzeResponse`` dict (for instant re-load)
- ``thumbnail.jpg`` — small mid-sagittal preview (~3-5 KB)
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from api.pipeline_service import CACHE_ROOT
from api.schemas import CaseSummary, PatientInfo


METADATA_FILE = "metadata.json"
RESULT_FILE = "result.json"
THUMBNAIL_FILE = "thumbnail.jpg"
THUMBNAIL_MAX = 128  # max dimension in pixels


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _case_dir(case_id: str) -> Path:
    return CACHE_ROOT / case_id


def _make_thumbnail(volume: np.ndarray) -> bytes:
    """Tiny JPEG of the central sagittal slice, suitable for history cards."""
    idx = volume.shape[0] // 2
    slc = np.rot90(np.flipud(volume[idx]))
    arr = slc.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-6:
        u8 = np.zeros_like(arr, dtype=np.uint8)
    else:
        u8 = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = u8.shape
    scale = THUMBNAIL_MAX / max(h, w)
    if scale < 1.0:
        u8 = cv2.resize(u8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", u8, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        raise RuntimeError("thumbnail encode failed")
    return bytes(buf)


def save_case(
    case_id: str,
    filename: str,
    response_dict: dict,
    volume: np.ndarray,
    patient: Optional[dict] = None,
) -> None:
    """Persist metadata + the full AnalyzeResponse + a thumbnail.

    Overwrites any previous files for the same case_id (e.g. on re-inference).
    Preserves the original ``uploaded_at`` and prior ``patient`` info (so a
    re-inference, which has no DICOMs at hand, doesn't blank out the patient).
    """
    d = _case_dir(case_id)
    d.mkdir(parents=True, exist_ok=True)

    meta_path = d / METADATA_FILE
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            uploaded_at = existing.get("uploaded_at", _now_iso())
            # Don't downgrade a real filename to "unknown" on re-inference.
            filename = existing.get("filename") or filename
            if patient is None:
                patient = existing.get("patient")
        except (OSError, json.JSONDecodeError):
            uploaded_at = _now_iso()
    else:
        uploaded_at = _now_iso()

    meta_path.write_text(json.dumps({
        "case_id": case_id,
        "filename": filename,
        "uploaded_at": uploaded_at,
        "patient": patient or {},
    }))

    (d / RESULT_FILE).write_text(json.dumps(response_dict))

    thumb_bytes = _make_thumbnail(volume)
    (d / THUMBNAIL_FILE).write_bytes(thumb_bytes)


def load_result(case_id: str) -> Optional[dict]:
    path = _case_dir(case_id) / RESULT_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_metadata(case_id: str) -> Optional[dict]:
    path = _case_dir(case_id) / METADATA_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _read_thumbnail_b64(case_id: str) -> Optional[str]:
    path = _case_dir(case_id) / THUMBNAIL_FILE
    if not path.exists():
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None


def list_summaries(limit: Optional[int] = None) -> List[CaseSummary]:
    """All analyzed cases, newest first. Cases without metadata/result are skipped."""
    if not CACHE_ROOT.exists():
        return []
    entries: List[CaseSummary] = []
    for d in CACHE_ROOT.iterdir():
        if not d.is_dir():
            continue
        meta = load_metadata(d.name)
        if meta is None:
            continue
        # Skip cases that haven't been fully analyzed yet (no result.json).
        if not (d / RESULT_FILE).exists():
            continue
        thumb = _read_thumbnail_b64(d.name) or ""
        patient_data = meta.get("patient") or {}
        entries.append(CaseSummary(
            case_id=meta.get("case_id", d.name),
            filename=meta.get("filename", "unknown.zip"),
            uploaded_at=meta.get("uploaded_at", _now_iso()),
            thumbnail_b64=thumb,
            patient=PatientInfo(**patient_data) if patient_data else PatientInfo(),
        ))

    entries.sort(key=lambda c: c.uploaded_at, reverse=True)
    if limit is not None:
        return entries[:limit]
    return entries


def total_count() -> int:
    if not CACHE_ROOT.exists():
        return 0
    n = 0
    for d in CACHE_ROOT.iterdir():
        if d.is_dir() and (d / METADATA_FILE).exists() and (d / RESULT_FILE).exists():
            n += 1
    return n
