"""FastAPI server for 4-model spine-disc classification comparison."""
from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api import case_store, models_registry
from api.job_manager import Job, manager as job_manager
from api.pipeline_service import CACHE_ROOT
from api.schemas import (
    AnalyzeResponse,
    CasesListResponse,
    HealthResponse,
    JobInfo,
    JobsListResponse,
    ModelResult,
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


# --------------------------------------------------------------------------- #
# Upload validation
# --------------------------------------------------------------------------- #
async def _read_zip(file: UploadFile) -> bytes:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, detail="Expected a .zip file containing the case DICOMs.")
    data = await file.read()
    if not data:
        raise HTTPException(400, detail="Empty upload.")
    return data


# --------------------------------------------------------------------------- #
# Job → JobInfo
# --------------------------------------------------------------------------- #
def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _job_info(job: Job) -> JobInfo:
    end = job.finished_at or time.time()
    return JobInfo(
        case_id=job.case_id,
        filename=job.filename,
        kind=job.kind,
        status=job.status,
        created_at=_iso(job.created_at) or _iso(time.time()),
        finished_at=_iso(job.finished_at),
        current_step=job.current_step,
        elapsed_seconds=max(0.0, end - job.created_at),
        error=job.error,
    )


# --------------------------------------------------------------------------- #
# Background-job endpoints
# --------------------------------------------------------------------------- #
@app.post("/analyze", response_model=JobInfo, status_code=202)
async def analyze_all(file: UploadFile = File(...)) -> JobInfo:
    """Start a background analyze job. Returns immediately with the job info.

    Subscribe to NDJSON progress events via ``GET /jobs/{case_id}/events``.
    """
    data = await _read_zip(file)
    job = job_manager.start_analyze(data, file.filename or "unknown.zip")
    return _job_info(job)


@app.post("/cases/{case_id}/reinference", response_model=JobInfo, status_code=202)
async def reinference_case(case_id: str) -> JobInfo:
    """Start a background re-inference job on cached preprocessing artifacts."""
    if not (CACHE_ROOT / case_id).is_dir():
        raise HTTPException(404, detail=f"Unknown case_id={case_id!r}.")
    meta = case_store.load_metadata(case_id)
    filename = (meta or {}).get("filename", "unknown.zip")
    job = job_manager.start_reinference(case_id, filename)
    return _job_info(job)


@app.get("/jobs", response_model=JobsListResponse)
def list_jobs() -> JobsListResponse:
    """All known jobs (running first, then most recently finished)."""
    return JobsListResponse(jobs=[_job_info(j) for j in job_manager.list_all()])


@app.get("/jobs/{case_id}", response_model=JobInfo)
def get_job(case_id: str) -> JobInfo:
    job = job_manager.get(case_id)
    if job is None:
        raise HTTPException(404, detail=f"No job for case_id={case_id!r}.")
    return _job_info(job)


@app.get("/jobs/{case_id}/events")
async def job_events(case_id: str) -> StreamingResponse:
    """Subscribe to a job's NDJSON event stream.

    Late subscribers get the full event replay first, then live events as they
    arrive, then EOF on job completion. Multiple clients may subscribe to the
    same job concurrently.
    """
    job = job_manager.get(case_id)
    if job is None:
        raise HTTPException(404, detail=f"No job for case_id={case_id!r}.")

    async def gen():
        async for ev in job.subscribe():
            yield json.dumps(ev) + "\n"

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Per-model debugging endpoints (synchronous; not used by the frontend)
# --------------------------------------------------------------------------- #
def _per_model_endpoint(model_name: str):
    from api.job_manager import _run_one_model
    from api.pipeline_service import preprocess_case

    async def endpoint(file: UploadFile = File(...)) -> ModelResult:
        data = await _read_zip(file)
        case = preprocess_case(data)
        return _run_one_model(model_name, case)

    return endpoint


app.post("/analyze/resnet101", response_model=ModelResult)(_per_model_endpoint("resnet101"))
app.post("/analyze/vgg19", response_model=ModelResult)(_per_model_endpoint("vgg19"))
app.post("/analyze/yolo22", response_model=ModelResult)(_per_model_endpoint("yolo22"))
app.post("/analyze/yolo26", response_model=ModelResult)(_per_model_endpoint("yolo26"))
