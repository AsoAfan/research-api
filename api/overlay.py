"""Annotate the central T2 sagittal slice with colored disc-segmentation
overlays — one color per predicted class, per model.
"""
from __future__ import annotations

import base64
from typing import Dict

import cv2
import numpy as np

from api.pipeline_service import DISC_LABELS, LABEL_TO_LEVEL


# BGR colors (matched on the frontend with the equivalent CSS classes).
CLASS_COLOR_BGR: Dict[str, tuple] = {
    "normal":     (0, 200, 0),     # green
    "bulge":      (0, 220, 220),   # yellow
    "protrusion": (0, 140, 255),   # orange
    "extrusion":  (0, 0, 220),     # red
}


def _normalize_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    arr = slice_2d.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _canonical_slice(volume: np.ndarray, slice_idx: int) -> np.ndarray:
    """Match the orientation used for the saved crops (see pipeline.save_result)."""
    return np.rot90(np.flipud(volume[slice_idx]))


def _slice_index(volume: np.ndarray) -> int:
    return volume.shape[0] // 2


def render_sagittal_png(volume: np.ndarray) -> str:
    """Un-annotated central sagittal slice, PNG-base64-encoded."""
    idx = _slice_index(volume)
    slc = _canonical_slice(volume, idx)
    u8 = _normalize_to_uint8(slc)
    bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def render_annotated_png(
    volume: np.ndarray,
    segmentation: np.ndarray,
    disc_predictions: Dict[int, str],
    *,
    with_labels: bool = False,
) -> str:
    """Central sagittal slice with each disc colored by its predicted class.

    disc_predictions maps disc label (e.g. 94) → predicted class name.
    When ``with_labels`` is True, the disc level + predicted class are drawn
    as text near each disc centroid.
    """
    idx = _slice_index(volume)
    vol_slc = _canonical_slice(volume, idx)
    seg_slc = _canonical_slice(segmentation, idx)

    u8 = _normalize_to_uint8(vol_slc)
    out = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR).astype(np.float32)

    for label in DISC_LABELS:
        mask = (np.rint(seg_slc).astype(int) == label)
        if not mask.any():
            continue
        pred = disc_predictions.get(label)
        if pred is None:
            continue
        color = np.array(CLASS_COLOR_BGR.get(pred, (255, 255, 255)), dtype=np.float32)
        out[mask] = out[mask] * 0.45 + color * 0.55

        if with_labels:
            ys, xs = np.where(mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            text = f"{LABEL_TO_LEVEL[label]} {pred[:4]}"
            cv2.putText(
                out,
                text,
                (max(cx - 30, 2), max(cy - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def compute_disc_regions(
    volume: np.ndarray,
    segmentation: np.ndarray,
) -> list[dict]:
    """Bounding box of each disc on the central canonical sagittal slice.

    Returned coordinates are normalized to the rendered PNG dimensions
    (the rotated sagittal slice — same orientation used by
    ``render_annotated_png``), so the frontend can place hover targets with
    CSS percentages regardless of display size.
    """
    idx = _slice_index(volume)
    seg_slc = _canonical_slice(segmentation, idx)
    H, W = seg_slc.shape

    pad_px = 2
    regions: list[dict] = []
    for label in DISC_LABELS:
        mask = (np.rint(seg_slc).astype(int) == label)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        x_min = max(int(xs.min()) - pad_px, 0)
        x_max = min(int(xs.max()) + pad_px, W - 1)
        y_min = max(int(ys.min()) - pad_px, 0)
        y_max = min(int(ys.max()) + pad_px, H - 1)
        regions.append({
            "level": LABEL_TO_LEVEL[label],
            "x": x_min / W,
            "y": y_min / H,
            "w": (x_max - x_min) / W,
            "h": (y_max - y_min) / H,
        })
    return regions


PAD_VALUE = 200.0  # matches _make_mid_slice_crop's resize_with_padding(..., pad_color=200)


def encode_disc_crop_png(crop_uint8: np.ndarray) -> str:
    """PNG-encode a 2D grayscale disc crop (e.g. 224×224) as base64.

    The cached crop is a float array that mixes raw MRI signal (~0-2000) with
    a fixed pad_color region (PAD_VALUE). The padding dominates the histogram,
    so any percentile/min-max normalization computed over the whole image
    crushes the disc tissue. We mask out the padding, normalize the tissue
    using its own 2-98 percentile range, and paint the padding back as a
    fixed light gray so it stays visually distinct.
    """
    arr = crop_uint8
    if arr.dtype != np.uint8:
        flat = arr.astype(np.float32)
        pad_mask = np.isclose(flat, PAD_VALUE, atol=0.5)
        tissue = flat[~pad_mask]
        if tissue.size < 16:
            tissue = flat.ravel()
        lo = float(np.percentile(tissue, 2.0))
        hi = float(np.percentile(tissue, 98.0))
        if hi - lo < 1e-6:
            normed = np.zeros_like(flat, dtype=np.uint8)
        else:
            normed = np.clip((flat - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        normed[pad_mask] = 180  # light gray background
        arr = normed
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
