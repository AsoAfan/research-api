"""Restore a model bundle produced by scripts/export_models.py.

Verifies every file against the manifest's sha256, then extracts under
<repo>/models/ in the layout that api/models_registry.py expects.

Usage:
    python scripts/import_models.py <bundle.zip> [--repo PATH] [--force]

By default an existing destination file is left in place if its sha256
already matches the manifest; pass --force to overwrite mismatches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_of_zipmember(zf: zipfile.ZipFile, name: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with zf.open(name, "r") as f:
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


def restore(bundle: Path, repo: Path, force: bool) -> int:
    if not bundle.is_file():
        print(f"[import] bundle not found: {bundle}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(bundle, "r") as zf:
        if "manifest.json" not in zf.namelist():
            print("[import] bundle is missing manifest.json", file=sys.stderr)
            return 1
        manifest = json.loads(zf.read("manifest.json"))

        extract_root = repo / manifest.get("extract_root", "models")
        print(f"[import] bundle    {bundle.name}")
        print(f"[import] target    {extract_root}")
        print(f"[import] created   {manifest.get('created_at', '?')}")
        print(f"[import] files     {len(manifest['files'])}")

        for entry in manifest["files"]:
            archive = entry["archive"]
            expected = entry["sha256"]

            if archive not in zf.namelist():
                print(f"[import] error: {archive} missing from zip", file=sys.stderr)
                return 1
            actual_zip = sha256_of_zipmember(zf, archive)
            if actual_zip != expected:
                print(f"[import] error: sha256 mismatch inside zip for {archive}\n"
                      f"  expected {expected}\n  got      {actual_zip}",
                      file=sys.stderr)
                return 1

            dst = extract_root / archive
            if dst.is_file() and not force:
                if sha256_of(dst) == expected:
                    print(f"  = {entry['role']:<10} {archive}  (already present)")
                    continue
                print(f"[import] error: {dst} exists with a different sha256; "
                      f"pass --force to overwrite", file=sys.stderr)
                return 1

            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(archive, "r") as src, dst.open("wb") as out:
                while True:
                    block = src.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
            if sha256_of(dst) != expected:
                print(f"[import] error: sha256 mismatch after writing {dst}",
                      file=sys.stderr)
                return 1
            print(f"  + {entry['role']:<10} {archive}  "
                  f"({human_bytes(entry['bytes'])})")

    print("[import] done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=Path, help="path to models_bundle_*.zip")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="repo root (defaults to parent of scripts/)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite destination files whose sha256 does not match")
    args = ap.parse_args()
    return restore(args.bundle.resolve(), args.repo.resolve(), args.force)


if __name__ == "__main__":
    raise SystemExit(main())
