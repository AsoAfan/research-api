"""Bundle the production model weights into a portable zip.

The four slots loaded by api/models_registry.py — plus the upgraded
train-22 checkpoint best_2.pt as the new yolo22 weight — are copied into
a single archive together with a manifest of sha256 hashes. The archive
can be moved to another machine and unpacked with scripts/import_models.py.

Usage:
    python scripts/export_models.py [--out PATH] [--repo PATH]

Default output is models_bundle_<YYYYmmdd-HHMMSS>.zip in the repo root.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List

# role -> (source path inside repo, archive path inside the bundle)
# Archive paths mirror the layout under <repo>/models/ that the API expects.
BUNDLE_SPEC: List[Dict[str, str]] = [
    {
        "role": "resnet101",
        "source": "models/resnet101/2_best_resnet101.pth",
        "archive": "resnet101/2_best_resnet101.pth",
    },
    {
        "role": "vgg19",
        "source": "models/vgg/2_best_vgg19.pth",
        "archive": "vgg/2_best_vgg19.pth",
    },
    {
        "role": "yolo22",
        "source": "models/yolo/train-22/weights/best_2.pt",
        "archive": "yolo/train-22/weights/best_2.pt",
    },
    {
        "role": "yolo26",
        "source": "models/yolo/train-26/weights/best.pt",
        "archive": "yolo/train-26/weights/best.pt",
    },
]

MANIFEST_VERSION = 1


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.1f} {u}"
        f /= 1024
    return f"{f:.1f} GB"


def build_bundle(repo: Path, out: Path) -> Path:
    entries = []
    missing = []
    for spec in BUNDLE_SPEC:
        src = repo / spec["source"]
        if not src.is_file():
            missing.append(str(src))
        entries.append((spec, src))

    if missing:
        msg = "missing required weight files:\n  " + "\n  ".join(missing)
        raise FileNotFoundError(msg)

    files_meta = []
    print(f"[export] staging {len(entries)} weights")
    with tempfile.TemporaryDirectory(prefix="models_bundle_") as tmp:
        staging = Path(tmp)
        for spec, src in entries:
            digest = sha256_of(src)
            size = src.stat().st_size
            dst = staging / spec["archive"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            files_meta.append({
                "role": spec["role"],
                "archive": spec["archive"],
                "source": spec["source"],
                "sha256": digest,
                "bytes": size,
            })
            print(f"  + {spec['role']:<10} {spec['archive']}  "
                  f"({human_bytes(size)}, sha256 {digest[:12]}…)")

        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "extract_root": "models",
            "files": files_meta,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))

        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
            for spec in BUNDLE_SPEC:
                zf.write(staging / spec["archive"], spec["archive"])
            zf.write(staging / "manifest.json", "manifest.json")

    total = sum(m["bytes"] for m in files_meta)
    print(f"[export] wrote {out}  ({human_bytes(out.stat().st_size)}, "
          f"weights total {human_bytes(total)})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="repo root (defaults to parent of scripts/)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output zip path (default: <repo>/models_bundle_<ts>.zip)")
    args = ap.parse_args()

    repo: Path = args.repo.resolve()
    out: Path = (args.out or
                 repo / f"models_bundle_{dt.datetime.now():%Y%m%d-%H%M%S}.zip").resolve()

    try:
        build_bundle(repo, out)
    except FileNotFoundError as e:
        print(f"[export] error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
