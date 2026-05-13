import random
import shutil
import time

import cv2
from pathlib import Path
import numpy as np
from numpy import ndarray
import os
os.environ['SSL_CERT_FILE'] = r'C:\Users\aso\Desktop\code\venv\Lib\site-packages\certifi\cacert.pem'

# C:\Users\aso\Desktop\code\venv\Lib\site-packages\certifi\cacert.pem

from custom_types.pading_value import Padding3D
from custom_types.process_result import ProcessResult
from utils.functions import ensure_dirs, get_t2_sagittals_only, convert_case_to_nifti, run_spine_segmentation, \
    load_volume_and_segmentation, display_images, clamp, resolve_segmentation, resize_with_padding


def crop_single_label(
        label: int,
        segmentation_data: ndarray,
        volume_data: ndarray,
        *,
        pad_x=(0, 0),
        pad_y=(0, 0),
        pad_z=(0, 0)
) -> tuple[ndarray, tuple[int, int, int]]:
    assert segmentation_data.shape == volume_data.shape, "Shapes must match"

    coords = np.argwhere(segmentation_data == label)

    if coords.size == 0:
        return None

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    z_min, z_max = clamp(z_min - pad_z[0], z_max + pad_z[1], volume_data.shape[0])
    y_min, y_max = clamp(y_min - pad_y[0], y_max + pad_y[1], volume_data.shape[1])
    x_min, x_max = clamp(x_min - pad_x[0], x_max + pad_x[1], volume_data.shape[2])

    cropped = volume_data[
        z_min:z_max + 1,
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    return cropped, cropped.shape


def crop_multiple_labels(
        labels: list[int],
        segmentation_data: ndarray,
        volume_data: ndarray,
        **pads
) -> dict[int, tuple[ndarray, tuple[int, int, int]]]:
    results = {}

    for label in labels:
        cropped = crop_single_label(
            label,
            segmentation_data,
            volume_data,
            **pads
        )

        if cropped is not None:
            # print(cropped)
            results[label] = cropped

    return results


def preprocess(case_path, volumes_case_path):
    # print("=" * 60)
    start_load = time.time()

    print(f"Processing case: {volumes_case_path.name} from class {volumes_case_path.parent.name}")
    print("=" * 60)

    ensure_dirs(volumes_case_path)

    # Step 1: Load DICOM
    # print("=" * 60)
    print("1. Loading data...")

    sag_dcm_objects = get_t2_sagittals_only(case_path)

    end_load = time.time()

    print(f"Done in {(end_load - start_load):.1f} seconds")

    print("=" * 60)

    # Step 2: Convert to NIfTI
    start_convert = time.time()
    # print("=" * 60)
    print("2. Processing data...")
    nii_path = convert_case_to_nifti(
        dcm_objects=sag_dcm_objects,
        output_dir=volumes_case_path
    )
    end_convert = time.time()
    print(f"Done in {(end_convert - start_convert):.1f} seconds")
    print("=" * 60)

    return nii_path


def process_case(
        case_path: Path,
        output_root: Path,
        labels: list[int],
        pads: Padding3D | None = None,
        *,
        segment_path: Path | str = None,
        with_save: bool = True,
        with_display: bool = False
) -> ProcessResult:
    pads = pads or {}
    start = time.time()
    segment_path = Path(segment_path) if isinstance(segment_path, str) else segment_path

    volumes_case_path = output_root / "volumes" / case_path.parent.name / case_path.name
    segmentation_case_path = output_root / "segmentations" / case_path.parent.name / case_path.name
    save_dir = str(volumes_case_path).replace("volumes", "results")
    # print(save_dir)
    # print(Path(save_dir).exists())
    if Path(save_dir).exists():
        print(f"Case {case_path.name} from {case_path.parent.name} allready processed, Skipping...")
        return

    

    nii_path = preprocess(case_path, volumes_case_path)

    # Step 3: Resolve segmentation
    # print("=" * 60)
    print("3. Doing segmentation...")
    start_segment = time.time()

    # seg_path = segment_path.exists() and segment_path or run_spine_segmentation(nii_path, segmentation_case_path)
    if segment_path.exists():
        seg_path = segment_path
        print(f"Segmentation already exist in {segment_path}")
    else:
        seg_path = run_spine_segmentation(nii_path, segmentation_case_path)

    end_segment = time.time()
    print(f"Done in {(end_segment - start_segment):.1f} seconds")
    print("=" * 60)
    


    cropped_results = postprocess(nii_path, seg_path, labels, **pads)

    result: ProcessResult = {
        "crops": cropped_results,
        "nii_path": nii_path,
        "seg_path": seg_path
    }

    if with_save:
        ensure_dirs(save_dir)

        save_result(result, save_dir, display=with_display)
    end = time.time()
    print("=" * 60)
    print(f"Processing for {case_path.parent.name}/{case_path.name} done in {end - start:.4f} seconds")
    print("=" * 60)

    return result


def postprocess(nii_path, seg_path, labels, **pads):
    # Step 4: Load data
    start_load = time.time()
    # print("=" * 60)
    print("4. Loading volume and segmentations...")
    volume_data, segmentation_data = load_volume_and_segmentation(nii_path, seg_path)

    end_load= time.time()
    print(f"Done in {(end_load - start_load):.1f} seconds")
    print("=" * 60)

    # Step 5: Crop multiple labels
    start_crop = time.time()
    # print("=" * 60)
    print("5. Cropping data...")


    cropped_results = crop_multiple_labels(
        labels,
        segmentation_data,
        volume_data,
        **pads
    )
    end_crop = time.time()
    print(f"Done in {(end_crop - start_crop):.1f} seconds")
    print("=" * 60)

    return cropped_results


def save_result(results: ProcessResult, save_dir: Path | str, *, display: bool = False):
    images = []
    save_dir = Path(save_dir) if isinstance(save_dir, str) else save_dir
    for label, cropped_img in results["crops"].items():
        img, shape = cropped_img
        mid_slice = img[shape[0] // 2]
        normal = np.rot90(
            np.flipud(mid_slice),
        )

        resized_normal = resize_with_padding(normal, target_size=(224, 224), pad_color=200)
        images.append(resized_normal)
        save_path = str(save_dir / f"{label_to_id[label]}.jpg")
        is_saved = cv2.imwrite(save_path, resized_normal)
        print("=" * 60)

        print(f"Result for {label} saved in {save_path}" if is_saved else "save failed")
        print("=" * 60)

        if display:
            display_images(images)


data_path = Path("data_v2")
dicom_data_path = data_path / "unzipped"
#
# extrusion_data_path = dicom_data_path / "extrusion"
# bulge_data_path = dicom_data_path / "bulge"
# protrusion_data_path = dicom_data_path / "protrusion"
# normal_data_path = dicom_data_path / "normal"
#
# case_path = extrusion_data_path / "case 3"
#
# volumes_data_path = data_path / "volumes"
# volumes_case_path = volumes_data_path / case_path.parent.name / case_path.name
#
# segmentation_data_path = data_path / "segmentations"
# segmentation_case_path = segmentation_data_path / case_path.parent.name / case_path.name
#
# result_data_path = data_path / "results" / case_path.parent.name / case_path.name
# result_data_path.mkdir(parents=True, exist_ok=True)

label_to_id = {
    41: "L1",
    42: "L2",
    43: "L3",
    44: "L4",
    45: "L5",
    92: "L1_L2",
    93: "L2_L3",
    94: "L3_L4",
    95: "L4_L5",
    100: "L5_S"
}
def main():
    start = time.time()
    try:
        for class_path in dicom_data_path.iterdir():
            cases = list(class_path.iterdir())
            cases_count = len(cases)
            
            for i, case in enumerate(cases):
                case_dir = Path("output_all") / "results" / class_path.name / case.name.replace("_", " ")
                if (case_dir).exists():
                    print(f"Skipping case: {class_path / case.name}")
                    continue

                # print(f"processing {case.name} from {class_path.name}")
                print(f"{i}/{cases_count} from {class_path.name} cases")
                process_case(
                    case_path=case,
                    output_root=Path("output_all"),
                    labels=[92, 93, 94, 95, 100],
                    pads={
                        "pad_x": (8, 8),
                        "pad_y": (5, 5),
                        "pad_z": (0, 0)
                    },
                    segment_path=Path("output_all") / "segmentations" / case.parent.name / case.name/ "step2_output" / "volumes.nii.gz" ,
                    with_save=True
                )
                print("=" * 60)

    except Exception as e:
        with open("error_cases.txt", "a") as err_file:
            line = f"{case} from {case.parent.name} has problem: {e}"
            err_file.write(f"{line}\n")
            err_file.write("=" * 50 + "\n")

        errors_path = data_path / "errors" / class_path.name
        # errors_path.mkdir(exist_ok=True)
        print(f"Problem occured with {case}, Moving it out and try again")
        shutil.move(case, errors_path / case.name)
        # main()


    end = time.time()

    print(f"Done in {(end - start)/60:.2f} minutes")


main()



# vd, sd = load_volume_and_segmentation(
#     Path("/home/aso/projects/python/research/code/data_test/volumes/extrusion/case 3/volumes.nii"),
#     Path("/home/aso/projects/python/research/code/data_test/segmentations/extrusion/case 3/step2_output/volumes.nii.gz"))


# coord1 = np.argwhere(np.isin(sd, [42]))
# z1_min, y1_min, x1_min = coord1.min(axis=0)
# z1_max, y1_max, x1_max = coord1.max(axis=0)

# coord2 = np.argwhere(np.isin(sd, [43]))
# z2_min, y2_min, x2_min = coord2.min(axis=0)
# z2_max, y2_max, x2_max = coord2.max(axis=0)

# crop1 = vd[
#     :,
#     y1_min + 20: y1_max ,
#     x1_min + 5 : x1_max ,
# ]

# crop2 = vd[
#     :,
#     y2_min: y2_max,
#     x2_min : x2_max ,
# ]
# #
# y_max_offset = y2_min * 0.21
# y_min_offset = y2_max * 0.05


# x_max_offset = x2_min * 0.081
# x_min_offset = x1_max * 0.12


# y_max = y2_min + 30  # increasing shifts the right-side to the left
# y_min = y2_max  # increasing shifts the left-side to the left
# x_min = x2_min + 22  # decreasing shifts the lower-side to bottom
# x_max =  x1_min + 25 # decreasing shifts the upper-side to the bottom

# # y_max = y1_min + 0  # increasing shifts the right-side to the left
# # y_min = y1_max  # increasing shifts the left-side to the left
# # x_min = x1_min   # decreasing shifts the lower-side to bottom
# # x_max =  x2_min # decreasing shifts the upper-side to the bottom



# crop3 = vd[
#     :,
#     y_max: y_min,
#     x_min: x_max,
# ]
# display_images([crop3[5], crop1[5], crop2[5], vd[5]], modifier=lambda img: np.rot90(np.flipud(img)),
#                titles=["result", "L3", "L4", "original_volume"]
#                )