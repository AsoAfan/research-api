"""Background-job lifecycle for case processing.

The pipeline (preprocess + 4 model inferences) runs as an ``asyncio.Task``
owned by ``JobManager`` — never by the HTTP request that initiated it. Clients
attach to a job by subscribing to its NDJSON event stream via
``GET /jobs/{case_id}/events``; the job keeps running whether or not anyone is
listening.

Each job is keyed by ``case_id`` (sha256 of the upload bytes). Starting an
analyze/reinference for a case that already has a running job returns the
existing job (clients just attach), so refreshing the page or two tabs racing
on the same upload doesn't spawn duplicate work.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
)

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
    DEFAULT_PADS,
    CaseData,
    _find_case_root,
    _hash_bytes,
    _make_mid_slice_crop,
    crop_multiple_labels,
    extract_patient_info,
    preprocess_case,
)
from api.schemas import (
    AnalyzeResponse,
    DiscPrediction,
    ModelResult,
    PatientInfo,
    StepTiming,
)
from utils.functions import get_t2_sagittals_only


logger = logging.getLogger("api.jobs")


MODEL_NAMES: List[str] = ["resnet101", "vgg19", "yolo22", "yolo26"]


# --------------------------------------------------------------------------- #
# Job
# --------------------------------------------------------------------------- #
JobStatus = Literal["running", "done", "error"]
JobKind = Literal["analyze", "reinference"]


@dataclass
class Job:
    case_id: str
    filename: str
    kind: JobKind
    loop: asyncio.AbstractEventLoop
    status: JobStatus = "running"
    events: List[dict] = field(default_factory=list)
    subscribers: List[asyncio.Queue] = field(default_factory=list)
    task: Optional[asyncio.Task] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    current_step: Optional[str] = None

    # ---- event fan-out -----------------------------------------------------
    def emit_sync(self, payload: dict) -> None:
        """Main-loop only. Track step + history, fan out to subscribers."""
        if payload.get("event") == "step":
            if payload.get("status") == "start":
                self.current_step = payload.get("step")
            elif payload.get("status") == "done" and self.current_step == payload.get("step"):
                self.current_step = None
        self.events.append(payload)
        for q in self.subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                logger.warning("subscriber queue full for case_id=%s", self.case_id)

    def emit_threadsafe(self, payload: dict) -> None:
        self.loop.call_soon_threadsafe(self.emit_sync, payload)

    # ---- termination -------------------------------------------------------
    def finish_sync(
        self,
        *,
        status: JobStatus,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        self.status = status
        self.result = result
        self.error = error
        self.finished_at = time.time()
        self.current_step = None
        for q in self.subscribers:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass
        # Clear the subscriber list so the dataclass isn't holding open queues.
        self.subscribers.clear()

    # ---- subscribe ---------------------------------------------------------
    async def subscribe(self) -> AsyncIterator[dict]:
        q: "asyncio.Queue[Optional[dict]]" = asyncio.Queue()
        for ev in list(self.events):
            await q.put(ev)
        if self.status == "running":
            self.subscribers.append(q)
        else:
            await q.put(None)
        try:
            while True:
                ev = await q.get()
                if ev is None:
                    return
                yield ev
        finally:
            if q in self.subscribers:
                self.subscribers.remove(q)


# --------------------------------------------------------------------------- #
# Job runners — moved verbatim from main._streaming_pipeline
# --------------------------------------------------------------------------- #
def _build_response(
    case: CaseData,
    results: List[ModelResult],
    timings: List[StepTiming],
    patient: PatientInfo,
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
        timings=timings,
        patient=patient,
        models=results,
    )


def _run_one_model(model_name: str, case: CaseData) -> ModelResult:
    discs: List[DiscPrediction] = []
    disc_predictions: Dict[int, str] = {}
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


def _load_existing_case(case_id: str, on_step) -> CaseData:
    """Reconstruct a CaseData from a previously-preprocessed cache dir."""
    import nibabel as nib

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
    cropped = crop_multiple_labels(DISC_LABELS, segmentation, volume, **DEFAULT_PADS)
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


async def _run_job(
    job: Job,
    *,
    zip_bytes: Optional[bytes],
    existing_case_id: Optional[str],
) -> None:
    """The detached pipeline coroutine. Emits NDJSON events into ``job``."""
    timings: List[StepTiming] = []

    def on_step(
        step_id: str,
        status: str,
        elapsed_ms: Optional[float],
        cached: bool = False,
    ) -> None:
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
        # Called from the worker thread; hop to the event loop for emit.
        job.emit_threadsafe(msg)

    case: Optional[CaseData] = None
    try:
        t0 = time.time()
        if zip_bytes is not None:
            case = await asyncio.to_thread(preprocess_case, zip_bytes, on_step)
        else:
            assert existing_case_id is not None
            case = await asyncio.to_thread(_load_existing_case, existing_case_id, on_step)
        logger.info(
            "preprocess took %.2fs (case_id=%s)",
            time.time() - t0,
            case.case_id,
        )

        results: List[ModelResult] = []
        for name in MODEL_NAMES:
            job.emit_sync({"event": "step", "step": name, "status": "start"})
            m_t0 = time.time()
            mr = await asyncio.to_thread(_run_one_model, name, case)
            elapsed_ms = (time.time() - m_t0) * 1000.0
            timings.append(StepTiming(step=name, elapsed_ms=elapsed_ms))
            job.emit_sync(
                {"event": "step", "step": name, "status": "done", "elapsed_ms": elapsed_ms}
            )
            results.append(mr)

        # Patient info: fresh upload → case.patient from DICOMs; re-inference
        # → either re-derived from cached DICOMs or loaded from metadata.json.
        patient_dict = case.patient
        if patient_dict is None:
            stored_meta = case_store.load_metadata(case.case_id) or {}
            patient_dict = stored_meta.get("patient") or None
        patient_model = (
            PatientInfo(**patient_dict) if patient_dict else PatientInfo()
        )

        response = _build_response(case, results, timings=timings, patient=patient_model)
        try:
            case_store.save_case(
                case.case_id,
                job.filename,
                response.model_dump(),
                case.volume,
                patient=patient_dict,
            )
        except Exception:
            logger.exception("failed to persist case history for %s", case.case_id)

        result_dict = response.model_dump()
        job.emit_sync({"event": "result", "data": result_dict})
        job.finish_sync(status="done", result=result_dict)
    except asyncio.CancelledError:
        # Server shutdown — record and re-raise so the task is properly torn down.
        job.emit_sync({"event": "error", "message": "Server shutdown"})
        job.finish_sync(status="error", error="Server shutdown")
        raise
    except Exception as exc:
        logger.exception("pipeline failed for case_id=%s", job.case_id)
        msg = str(exc) or exc.__class__.__name__
        err: dict = {"event": "error", "message": msg}
        if job.current_step:
            err["step"] = job.current_step
        job.emit_sync(err)
        job.finish_sync(status="error", error=msg)


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #
class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}

    def get(self, case_id: str) -> Optional[Job]:
        return self._jobs.get(case_id)

    def list_all(self) -> List[Job]:
        """All jobs (running first, then most recently finished)."""
        return sorted(
            self._jobs.values(),
            key=lambda j: (
                0 if j.status == "running" else 1,
                -(j.finished_at or j.created_at),
            ),
        )

    def list_active(self) -> List[Job]:
        return [j for j in self._jobs.values() if j.status == "running"]

    def start_analyze(self, zip_bytes: bytes, filename: str) -> Job:
        case_id = _hash_bytes(zip_bytes)
        existing = self._jobs.get(case_id)
        if existing and existing.status == "running":
            return existing
        loop = asyncio.get_running_loop()
        job = Job(case_id=case_id, filename=filename, kind="analyze", loop=loop)
        self._jobs[case_id] = job
        job.task = asyncio.create_task(
            _run_job(job, zip_bytes=zip_bytes, existing_case_id=None),
            name=f"analyze-{case_id}",
        )
        return job

    def start_reinference(self, case_id: str, filename: str) -> Job:
        existing = self._jobs.get(case_id)
        if existing and existing.status == "running":
            return existing
        loop = asyncio.get_running_loop()
        job = Job(case_id=case_id, filename=filename, kind="reinference", loop=loop)
        self._jobs[case_id] = job
        job.task = asyncio.create_task(
            _run_job(job, zip_bytes=None, existing_case_id=case_id),
            name=f"reinference-{case_id}",
        )
        return job


# Module-level singleton.
manager = JobManager()
