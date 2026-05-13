from ultralytics import YOLO
import sys
import pathlib

# Shim for checkpoints pickled on Windows that reference pathlib._local.WindowsPath.
# On POSIX, pathlib has no _local submodule, and WindowsPath raises NotImplementedError.
sys.modules.setdefault("pathlib._local", pathlib)
if sys.platform != "win32":
    class _PortableWindowsPath(pathlib.PurePosixPath):
        pass
    pathlib.WindowsPath = _PortableWindowsPath
    if not hasattr(pathlib, "_local"):
        pathlib._local = pathlib  # type: ignore[attr-defined]
    pathlib._local.WindowsPath = _PortableWindowsPath  # type: ignore[attr-defined]

    
model = YOLO("yolo_train_22_26/train-26/weights/best.pt")
metrics = model.val(data="augmented_dataset_split", split="val")

print(f"Top-1: {metrics.top1:.4f}")
print(f"Top-5: {metrics.top5:.4f}")
