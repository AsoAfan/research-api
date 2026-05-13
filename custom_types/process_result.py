from typing import TypedDict, Any
from pathlib import Path

from numpy import ndarray


class ProcessResult(TypedDict):
    crops: dict[int, tuple[ndarray, tuple[int, int, int]]]
    nii_path: Path
    seg_path: Path
