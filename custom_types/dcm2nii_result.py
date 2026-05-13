from typing import TypedDict
from pathlib import Path
import numpy as np
import nibabel as nib


class DicomNifiResult(TypedDict):
    NII_FILE: Path|str
    NII: nib.Nifti1Image
    MAX_SLICE_INCREMENT: np.float64