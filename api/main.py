"""FastAPI server for 4-model spine-disc classification comparison."""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api import case_store, models_registry
from api.overlay import (
    compute_disc_regions,
    encode_disc_crop_png,
    render_annotated_png,
    render_sagittal_png,
)
from api.pipeline_service import (
    CACHE_ROOT,
    DISC_LABELS,
    LABEL_TO_LEVEL,
    CaseData,
    preprocess_case,
)
from api.schemas import (
    AnalyzeResponse,
    CasesListResponse,
    DiscPrediction,
    HealthResponse,
    ModelResult,
    PatientInfo,
    StepTiming,
)


logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)


app = FastAPI(title="Spine Disc Classifier — 4-Model Comparison")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_NAMES: List[str] = ["resnet101", "vgg19", "yolo22", "yolo26"]


@app.on_event("startup")
def _startup() -> None:
    logger.info("loading models on startup...")
    models_registry.load_all()
    logger.info("models ready")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    reg = models_registry.load_all()
    return HealthResponse(
        status="ok",
        device=str(models_registry.DEVICE),
        models_loaded=list(reg.keys()),
        totalspineseg_available=shutil.which("totalspineseg") is not None,
    )


async def _read_zip(file: UploadFile) -> bytes:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, detail="Expected a .zip file containing the case DICOMs.")
    data = await file.read()
    if not data:
        raise HTTPException(400, detail="Empty upload.")
    return data


def _build_response(
    case: CaseData,
    results: List[ModelResult],
    timings: Optional[List[StepTiming]] = None,
    patient: Optional[PatientInfo] = None,
) -> AnalyzeResponse:
    original = render_sagittal_png(case.volume)
    disc_crops_b64 = {
        LABEL_TO_LEVEL[label]: encode_disc_crop_png(crop)
        for label, crop in case.disc_crops.items()
    }
    disc_regions = compute_disc_regions(case.volume, case.segmentation)
    return AnalyzeResponse(
        case_id=case.case_id,
        original_image_b64=original,
        disc_crops_b64=disc_crops_b64,
        disc_regions=disc_regions,
        timings=timings or [],
        patient=patient or PatientInfo(),
        models=results,
    )


def _run_one_model(model_name: str, case: CaseData) -> ModelResult:
    discs: List[DiscPrediction] = []
    disc_predictions: dict[int, str] = {}
    total_ms = 0.0
    for label in DISC_LABELS:
        crop = case.disc_crops.get(label)
        if crop is None:
            continue
        pred, probs, elapsed_ms = models_registry.predict_one(model_name, crop)
        total_ms += elapsed_ms
        disc_predictions[label] = pred
        discs.append(
            DiscPrediction(
                level=LABEL_TO_LEVEL[label],
                prediction=pred,
                probabilities=probs,
            )
        )
    overlay = render_annotated_png(
        case.volume, case.segmentation, disc_predictions, with_labels=False
    )
    overlay_labeled = render_annotated_png(
        case.volume, case.segmentation, disc_predictions, with_labels=True
    )
    return ModelResult(
        model_name=model_name,
        overlay_b64=overlay,
        overlay_labeled_b64=overlay_labeled,
        discs=discs,
        inference_ms=total_ms,
    )


def _per_model_endpoint(model_name: str):
    async def endpoint(file: UploadFile = File(...)) -> ModelResult:
        data = await _read_zip(file)
        case = preprocess_case(data)
        return _run_one_model(model_name, case)
    return endpoint


app.post("/analyze/resnet101", response_model=ModelResult)(_per_model_endpoint("resnet101"))
app.post("/analyze/vgg19", response_model=ModelResult)(_per_model_endpoint("vgg19"))
app.post("/analyze/yolo22", response_model=ModelResult)(_per_model_endpoint("yolo22"))
app.post("/analyze/yolo26", response_model=ModelResult)(_per_model_endpoint("yolo26"))


def _streaming_pipeline(
    *,
    zip_bytes: Optional[bytes],
    existing_case_id: Optional[str],
    filename: str,
) -> StreamingResponse:
    """Shared NDJSON streaming engine used by both ``/analyze`` (zip upload)
    and ``/cases/{id}/reinference`` (existing case, skip preprocessing).

    Exactly one of ``zip_bytes`` or ``existing_case_id`` must be set.
    """
    if (zip_bytes is None) == (existing_case_id is None):
        raise ValueError("provide exactly one of zip_bytes / existing_case_id")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    current_step: dict = {"id": None}
    timings: List[StepTiming] = []

    def emit(payload: dict) -> None:
        queue.put_nowait(json.dumps(payload) + "\n")

    def on_step(
        step_id: str,
        status: str,
        elapsed_ms: Optional[float],
        cached: bool = False,
    ) -> None:
        current_step["id"] = step_id if status == "start" else None
        msg: dict = {"event": "step", "step": step_id, "status": status}
        if elapsed_ms is not None:
            msg["elapsed_ms"] = elapsed_ms
        if cached:
            msg["cached"] = True
        if status == "done":
            timings.append(
                StepTiming(
                    step=step_id,
                    elapsed_ms=float(elapsed_ms or 0.0),
                    cached=cached,
                )
            )
        loop.call_soon_threadsafe(queue.put_nowait, json.dumps(msg) + "\n")

    def _load_existing_case(case_id: str) -> CaseData:
        """Reconstruct a CaseData from a previously-preprocessed cache dir.

        For re-inference we just need the volume, segmentation, and disc crops —
        no need to re-unzip or re-segment. We synthesize the cached step events
        to keep the frontend timeline consistent.

        Also re-derives patient demographics from the cached DICOMs when they
        are still on disk, so re-inferencing an older case backfills patient
        info that wasn't captured the first time.
        """
        import nibabel as nib
        from api.pipeline_service import (
            DISC_LABELS as _DISC_LABELS,
            DEFAULT_PADS,
            _find_case_root,
            _make_mid_slice_crop,
            crop_multiple_labels,
            extract_patient_info,
        )
        from utils.functions import get_t2_sagittals_only

        d = CACHE_ROOT / case_id
        nii_path = d / "volumes" / "volumes.nii"
        seg_path = d / "segmentation" / "step2_output" / "volumes.nii.gz"
        if not (nii_path.exists() and seg_path.exists()):
            raise FileNotFoundError(
                f"Preprocessing artifacts missing for case {case_id}; "
                "upload the case again to populate the cache."
            )

        patient: Optional[dict] = None
        extracted = d / "extracted"
        if extracted.exists():
            try:
                case_root = _find_case_root(extracted)
                sag = get_t2_sagittals_only(case_root)
                if sag:
                    patient = extract_patient_info(sag[0])
            except Exception:
                logger.exception("failed to re-derive patient info for %s", case_id)

        for step_id in ("unzip", "filter_t2", "dicom_to_nifti", "segmentation"):
            on_step(step_id, "done", 0.0, cached=True)
        volume = nib.load(str(nii_path)).get_fdata()
        segmentation = nib.load(str(seg_path)).get_fdata()
        cropped = crop_multiple_labels(_DISC_LABELS, segmentation, volume, **DEFAULT_PADS)
        disc_crops = {
            label: _make_mid_slice_crop(img) for label, (img, _shape) in cropped.items()
        }
        on_step("crop_discs", "done", 0.0, cached=True)
        return CaseData(
            case_id=case_id,
            volume=volume,
            segmentation=segmentation,
            disc_crops=disc_crops,
            workdir=d,
            patient=patient,
        )

    async def runner() -> None:
        case: Optional[CaseData] = None
        try:
            t0 = time.time()
            if zip_bytes is not None:
                case = await asyncio.to_thread(preprocess_case, zip_bytes, on_step)
            else:
                assert existing_case_id is not None
                case = await asyncio.to_thread(_load_existing_case, existing_case_id)
            logger.info(
                "preprocess took %.2fs (case_id=%s)",
                time.time() - t0,
                case.case_id,
            )

            results: List[ModelResult] = []
            for name in MODEL_NAMES:
                current_step["id"] = name
                emit({"event": "step", "step": name, "status": "start"})
                m_t0 = time.time()
                mr = await asyncio.to_thread(_run_one_model, name, case)
                elapsed_ms = (time.time() - m_t0) * 1000.0
                timings.append(StepTiming(step=name, elapsed_ms=elapsed_ms))
                emit({"event": "step", "step": name, "status": "done", "elapsed_ms": elapsed_ms})
                current_step["id"] = None
                results.append(mr)

            # On a fresh upload, case.patient comes from the DICOM headers.
            # On re-inference, it's None — pull it back from disk so the
            # response (and the resaved metadata) preserves it.
            patient_dict = case.patient
            if patient_dict is None:
                stored_meta = case_store.load_metadata(case.case_id) or {}
                patient_dict = stored_meta.get("patient") or None
            patient_model = PatientInfo(**patient_dict) if patient_dict else PatientInfo()

            response = _build_response(
                case, results, timings=timings, patient=patient_model
            )
            try:
                case_store.save_case(
                    case.case_id,
                    filename,
                    response.model_dump(),
                    case.volume,
                    patient=patient_dict,
                )
            except Exception:
                logger.exception("failed to persist case history for %s", case.case_id)

            emit({"event": "result", "data": response.model_dump()})
        except Exception as exc:
            logger.exception("pipeline failed")
            err: dict = {"event": "error", "message": str(exc) or exc.__class__.__name__}
            if current_step["id"]:
                err["step"] = current_step["id"]
            queue.put_nowait(json.dumps(err) + "\n")
        finally:
            queue.put_nowait(None)

    async def gen():
        task = asyncio.create_task(runner())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            await task

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.post("/analyze")
async def analyze_all(file: UploadFile = File(...)) -> StreamingResponse:
    """Run all four models on a fresh ZIP upload, streaming NDJSON progress."""
    data = await _read_zip(file)
    return _streaming_pipeline(
        zip_bytes=data,
        existing_case_id=None,
        filename=file.filename or "unknown.zip",
    )


@app.get("/cases", response_model=CasesListResponse)
def list_cases(limit: Optional[int] = None) -> CasesListResponse:
    """Newest-first list of cases that have a saved AnalyzeResponse on disk."""
    summaries = case_store.list_summaries(limit=limit)
    return CasesListResponse(cases=summaries, total=case_store.total_count())


@app.get("/cases/{case_id}", response_model=AnalyzeResponse)
def get_case(case_id: str) -> AnalyzeResponse:
    """Return the cached AnalyzeResponse for a previously-analyzed case."""
    data = case_store.load_result(case_id)
    if data is None:
        raise HTTPException(404, detail=f"No analyzed result for case_id={case_id!r}.")
    return AnalyzeResponse(**data)


@app.post("/cases/{case_id}/reinference")
async def reinference_case(case_id: str) -> StreamingResponse:
    """Re-run the 4 models against the cached preprocessing for an existing case."""
    if not (CACHE_ROOT / case_id).is_dir():
        raise HTTPException(404, detail=f"Unknown case_id={case_id!r}.")
    meta = case_store.load_metadata(case_id)
    filename = (meta or {}).get("filename", "unknown.zip")
    return _streaming_pipeline(
        zip_bytes=None,
        existing_case_id=case_id,
        filename=filename,
    )
