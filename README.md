# Spine Disc Classifier — Backend

End-to-end pipeline for classifying lumbar intervertebral discs from MRI DICOM
studies. A ZIP of a patient's case is uploaded, the T2 sagittal series is
isolated, the spine is segmented with [TotalSpineSeg], each disc is cropped, and
four classifiers run side-by-side so their predictions can be compared.

Classes per disc: **normal**, **bulge**, **protrusion**, **extrusion**.

[TotalSpineSeg]: https://github.com/neuropoly/totalspineseg

## Architecture

```
┌──────────────┐  ZIP of DICOMs   ┌────────────────────┐   per-disc crops   ┌────────────────────────────┐
│  Frontend /  │ ───────────────▶ │  FastAPI (api/)    │ ─────────────────▶ │  4 classifiers (parallel)  │
│  HTTP client │                  │  job_manager       │                    │  ResNet101 · VGG19         │
└──────────────┘  NDJSON events   │  pipeline_service  │                    │  YOLOv22-cls · YOLOv26-cls │
                  ◀───────────────│  case_store        │                    └────────────────────────────┘
                                  └────────────────────┘
```

Preprocessing (DICOM → NIfTI → segmentation → per-disc crops) is cached on disk
in [api/cache/](api/cache/), keyed by a hash of the uploaded ZIP. Re-inference
on a cached case skips straight to the classifiers.

## Repository layout

| Path | Purpose |
| --- | --- |
| [api/](api/) | FastAPI server, job manager, pipeline service, on-disk case store |
| [api/main.py](api/main.py) | HTTP endpoints (`/analyze`, `/jobs`, `/cases`, …) |
| [api/pipeline_service.py](api/pipeline_service.py) | ZIP → preprocessed crops, with caching |
| [api/models_registry.py](api/models_registry.py) | Lazy singleton loaders + inference for the 4 classifiers |
| [api/job_manager.py](api/job_manager.py) | Background jobs with NDJSON event streams |
| [api/case_store.py](api/case_store.py) | Persisted `AnalyzeResponse` history |
| [pipeline.py](pipeline.py) | Standalone CLI pipeline (no HTTP) |
| [classifier.py](classifier.py), [resnet101.py](resnet101.py), [vgg19.py](vgg19.py) | Training entrypoints |
| [eval_yolo.py](eval_yolo.py), [eval_resnet.py](eval_resnet.py), [eval_vgg.py](eval_vgg.py) | Per-architecture evaluation |
| [utils/](utils/), [custom_types/](custom_types/) | Shared helpers and dataclasses |
| [models/](models/) | Trained weights (`*.pth`, YOLO `best.pt`) — **gitignored** |
| [runs/](runs/), [evaluations/](evaluations/) | Training runs and eval reports — **gitignored** |
| [spine-clarity-view/](spine-clarity-view/) | Frontend (see its own README) |

## Requirements

- Python 3.10+
- A CUDA-capable GPU is strongly recommended (CPU works but is slow)
- [TotalSpineSeg] on `PATH` (the `/health` endpoint reports its availability)
- The four model weights placed in [models/](models/):
  - `models/resnet101/<name>.pth`
  - `models/vgg/<name>.pth`
  - `models/yolo/<name>.pt`
  - `models/yolo_train_22_26/<name>.pt`

  Exact filenames are resolved in [api/models_registry.py](api/models_registry.py).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r api/requirements.txt
# Plus (pre-installed in the project venv): torch, torchvision, ultralytics,
# opencv-python, numpy, pillow, scikit-learn, matplotlib
```

## Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Models load on startup; the first request after boot will not pay the load cost.
CORS is wide-open by default — tighten `allow_origins` in
[api/main.py](api/main.py) before exposing the server.

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET`  | `/health` | Service status, device, loaded models, `totalspineseg` availability |
| `POST` | `/analyze` | Upload a ZIP, start a background job. Returns `JobInfo` (202) |
| `POST` | `/cases/{case_id}/reinference` | Re-run classifiers on a cached case |
| `GET`  | `/jobs` | List all jobs (running first, then most recent) |
| `GET`  | `/jobs/{case_id}` | Single job status |
| `GET`  | `/jobs/{case_id}/events` | NDJSON stream of progress events (replay + live) |
| `GET`  | `/cases` | Newest-first list of analyzed cases (`?limit=` optional) |
| `GET`  | `/cases/{case_id}` | Full cached `AnalyzeResponse` |
| `POST` | `/analyze/{resnet101\|vgg19\|yolo22\|yolo26}` | Synchronous single-model inference (debugging) |

Response schemas: [api/schemas.py](api/schemas.py). Interactive docs are at
`http://localhost:8000/docs`.

### Example

```bash
# Kick off an analysis
curl -s -F "file=@case.zip" http://localhost:8000/analyze

# Stream progress (replays from the start, then tails live)
curl -N http://localhost:8000/jobs/<case_id>/events

# Fetch the final result
curl -s http://localhost:8000/cases/<case_id> | jq .
```

## Standalone pipeline

[pipeline.py](pipeline.py) runs the preprocessing + classification flow from the
command line without the HTTP layer — useful for debugging crops and overlays.

## Training & evaluation

- `python classifier.py` — train YOLO classifiers
- `python resnet101.py` / `python vgg19.py` — train the CNN classifiers
- `python eval_yolo.py` / `eval_resnet.py` / `eval_vgg.py` — evaluation reports
  written to [evaluations/](evaluations/)

These scripts execute on import (they call `main()` at module load), which is
why [api/models_registry.py](api/models_registry.py) reconstructs the model
architectures locally instead of importing them.

## Frontend

A Vite + React UI lives under [spine-clarity-view/](spine-clarity-view/). See
its [README](spine-clarity-view/README.md) for setup and usage.
