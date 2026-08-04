#!/usr/bin/env python3
"""Create binary TotalSegmentator body masks for venous PMPD CT volumes."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_DATASET = Path("/local/radiomics/PMPD_v2_data")
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXECUTABLE = Path(sys.executable).with_name("TotalSegmentator")
DEFAULT_RUNTIME = DEFAULT_PROJECT_ROOT / ".pmpd_body_runtime"
DEFAULT_WEIGHTS = DEFAULT_RUNTIME / "weights"


@dataclass(frozen=True)
class Case:
    cohort: str
    image: str
    output: str


def is_nifti(path: Path) -> bool:
    return path.name.endswith(".nii.gz") or path.suffix == ".nii"


def output_name(image: Path) -> str:
    return image.name


def report_name(image: Path) -> str:
    if image.name.endswith(".nii.gz"):
        return image.name[:-7] + ".json"
    return image.stem + ".json"


def discover_cases(dataset: Path) -> list[Case]:
    cases: list[Case] = []
    for cohort_dir in sorted(dataset.iterdir(), key=lambda path: path.name.casefold()):
        if not cohort_dir.is_dir() or "arterial" in cohort_dir.name.casefold():
            continue
        images_dir = cohort_dir / "images"
        if not images_dir.is_dir():
            continue
        for image in sorted(images_dir.iterdir(), key=lambda path: path.name.casefold()):
            if image.is_file() and is_nifti(image):
                cases.append(
                    Case(
                        cohort=cohort_dir.name,
                        image=str(image),
                        output=str(cohort_dir / "body_mask" / output_name(image)),
                    )
                )
    return cases


def validate_input(path: Path) -> dict[str, object]:
    image = nib.load(path)
    if len(image.shape) != 3 or any(size <= 0 for size in image.shape):
        raise ValueError(f"Expected a non-empty 3D NIfTI, got shape {image.shape}")
    if not np.isfinite(image.affine).all():
        raise ValueError("Input affine contains non-finite values")
    return {
        "shape": tuple(int(value) for value in image.shape),
        "zooms": tuple(float(value) for value in image.header.get_zooms()[:3]),
        "dtype": str(image.get_data_dtype()),
    }


def validate_binary_mask(image_path: Path, mask_path: Path, load_data: bool = True) -> dict[str, object]:
    image = nib.load(image_path)
    mask = nib.load(mask_path)
    if mask.shape != image.shape:
        raise ValueError(f"Shape mismatch: image={image.shape}, mask={mask.shape}")
    if not np.allclose(mask.affine, image.affine, rtol=0.0, atol=1e-4):
        raise ValueError("Affine mismatch between image and mask")
    result: dict[str, object] = {
        "shape": tuple(int(value) for value in mask.shape),
        "dtype": str(mask.get_data_dtype()),
    }
    if load_data:
        data = np.asanyarray(mask.dataobj)
        values = np.unique(data)
        if not set(values.tolist()).issubset({0, 1}):
            raise ValueError(f"Mask is not binary: values={values.tolist()}")
        voxels = int(np.count_nonzero(data))
        if voxels == 0:
            raise ValueError("Mask is empty")
        fraction = float(voxels / data.size)
        if not 0.001 < fraction < 0.95:
            raise ValueError(f"Implausible foreground fraction: {fraction:.6f}")
        result.update(
            {
                "foreground_voxels": voxels,
                "foreground_fraction": fraction,
                "values": values.astype(int).tolist(),
            }
        )
    return result


def case_is_complete(case: Case) -> bool:
    output = Path(case.output)
    if not output.is_file():
        return False
    try:
        validate_binary_mask(Path(case.image), output, load_data=True)
    except Exception:
        return False
    return True


def atomic_json_dump(data: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)


def run_case(
    case: Case,
    gpu: int,
    executable: Path,
    runtime: Path,
    weights: Path,
) -> dict[str, object]:
    started = time.monotonic()
    image_path = Path(case.image)
    output_path = Path(case.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_home = runtime / f"gpu{gpu}"
    gpu_home.mkdir(parents=True, exist_ok=True)
    report_path = runtime / "reports" / case.cohort / report_name(image_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_multilabel = output_path.with_name(f".{output_path.name}.multilabel.tmp.nii.gz")
    temporary_binary = output_path.with_name(f".{output_path.name}.part.nii.gz")
    temporary_report = report_path.with_name(f".{report_path.name}.part")
    log_path = runtime / "logs" / case.cohort / f"{report_name(image_path)[:-5]}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for temporary in (temporary_multilabel, temporary_binary, temporary_report):
        if temporary.exists():
            temporary.unlink()

    input_info = validate_input(image_path)
    environment = os.environ.copy()
    environment["TOTALSEG_HOME_DIR"] = str(gpu_home)
    environment["TOTALSEG_WEIGHTS_PATH"] = str(weights)
    command = [
        str(executable),
        "-i",
        str(image_path),
        "-o",
        str(temporary_multilabel),
        "-ta",
        "body",
        "-ml",
        "-d",
        f"gpu:{gpu}",
        "-rmb",
        "200",
        "-q",
        "-rp",
        str(temporary_report),
    ]

    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"TotalSegmentator exited with {completed.returncode}; see {log_path}")
    if not temporary_multilabel.is_file():
        raise RuntimeError("TotalSegmentator did not create its multilabel output")

    source = nib.load(image_path)
    segmentation = nib.load(temporary_multilabel)
    if segmentation.shape != source.shape:
        raise ValueError(f"TotalSegmentator shape mismatch: {segmentation.shape} != {source.shape}")
    if not np.allclose(segmentation.affine, source.affine, rtol=0.0, atol=1e-4):
        raise ValueError("TotalSegmentator affine does not match the source image")
    segmentation_data = np.asanyarray(segmentation.dataobj)
    source_values = np.unique(segmentation_data)
    if not set(source_values.tolist()).issubset({0, 1, 2}):
        raise ValueError(f"Unexpected TotalSegmentator labels: {source_values.tolist()}")
    body = (segmentation_data > 0).astype(np.uint8, copy=False)

    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    binary_image = nib.Nifti1Image(body, source.affine, header=header)
    qform, qform_code = source.get_qform(coded=True)
    sform, sform_code = source.get_sform(coded=True)
    if qform is not None:
        binary_image.set_qform(qform, int(qform_code))
    if sform is not None:
        binary_image.set_sform(sform, int(sform_code))
    nib.save(binary_image, temporary_binary)
    output_info = validate_binary_mask(image_path, temporary_binary, load_data=True)
    os.replace(temporary_binary, output_path)
    temporary_multilabel.unlink()

    report: dict[str, object] = {}
    if temporary_report.is_file():
        with temporary_report.open(encoding="utf-8") as handle:
            report = json.load(handle)
        temporary_report.unlink()
    report.update(
        {
            "binary_body_mask": str(output_path),
            "binary_definition": "body_trunc OR body_extremities",
            "cohort": case.cohort,
            "gpu_index": gpu,
            "input_validation": input_info,
            "output_validation": output_info,
            "pipeline_runtime_seconds": round(time.monotonic() - started, 3),
        }
    )
    atomic_json_dump(report, report_path)
    return {
        "status": "completed",
        "cohort": case.cohort,
        "image": str(image_path),
        "output": str(output_path),
        "gpu": gpu,
        "seconds": round(time.monotonic() - started, 3),
        "foreground_fraction": output_info["foreground_fraction"],
    }


def worker(
    gpu: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    executable: str,
    runtime: str,
    weights: str,
) -> None:
    while True:
        case = task_queue.get()
        if case is None:
            return
        try:
            result = run_case(
                case,
                gpu,
                Path(executable),
                Path(runtime),
                Path(weights),
            )
        except Exception as error:
            result = {
                "status": "failed",
                "cohort": case.cohort,
                "image": case.image,
                "output": case.output,
                "gpu": gpu,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        result_queue.put(result)


def write_manifest(cases: list[Case], runtime: Path, dataset: Path) -> Path:
    manifest = runtime / "venous_manifest.json"
    cohort_counts: dict[str, int] = {}
    for case in cases:
        cohort_counts[case.cohort] = cohort_counts.get(case.cohort, 0) + 1
    atomic_json_dump(
        {
            "dataset": str(dataset.resolve()),
            "selection_rule": "cohort has images/ and cohort name does not contain 'arterial'",
            "total_cases": len(cases),
            "cohort_counts": cohort_counts,
            "cases": [asdict(case) for case in cases],
        },
        manifest,
    )
    return manifest


def run_parallel(
    cases: list[Case],
    gpus: list[int],
    executable: Path,
    runtime: Path,
    weights: Path,
) -> int:
    pending = [case for case in cases if not case_is_complete(case)]
    skipped = len(cases) - len(pending)
    print(json.dumps({"event": "start", "total": len(cases), "pending": len(pending), "skipped": skipped, "gpus": gpus}), flush=True)
    if not pending:
        return 0

    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=worker,
            args=(gpu, task_queue, result_queue, str(executable), str(runtime), str(weights)),
        )
        for gpu in gpus
    ]
    for process in processes:
        process.start()
    for case in pending:
        task_queue.put(case)
    for _ in processes:
        task_queue.put(None)

    failed = 0
    status_path = runtime / "batch_status.jsonl"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("a", encoding="utf-8") as status_handle:
        for index in range(1, len(pending) + 1):
            result = result_queue.get()
            if result["status"] == "failed":
                failed += 1
            result["progress"] = f"{index}/{len(pending)}"
            line = json.dumps(result, sort_keys=True)
            print(line, flush=True)
            status_handle.write(line + "\n")
            status_handle.flush()

    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed += 1
    print(json.dumps({"event": "finished", "completed": len(pending) - failed, "failed": failed, "skipped": skipped}), flush=True)
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--case", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = discover_cases(args.dataset)
    manifest = write_manifest(cases, args.runtime, args.dataset)
    cohort_counts: dict[str, int] = {}
    for case in cases:
        cohort_counts[case.cohort] = cohort_counts.get(case.cohort, 0) + 1
    if args.inventory:
        input_errors: list[dict[str, str]] = []
        shapes: dict[str, int] = {}
        for case in cases:
            try:
                info = validate_input(Path(case.image))
                shape_key = "x".join(str(value) for value in info["shape"])
                shapes[shape_key] = shapes.get(shape_key, 0) + 1
            except Exception as error:
                input_errors.append({"image": case.image, "error": str(error)})
        print(
            json.dumps(
                {
                    "manifest": str(manifest),
                    "total": len(cases),
                    "cohorts": cohort_counts,
                    "shape_counts": shapes,
                    "input_errors": input_errors,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1 if input_errors else 0

    if args.case is not None:
        resolved = args.case.resolve()
        matches = [case for case in cases if Path(case.image).resolve() == resolved]
        if len(matches) != 1:
            raise SystemExit(f"Case is not in the venous manifest: {resolved}")
        if case_is_complete(matches[0]):
            print(json.dumps({"status": "skipped", "reason": "valid output already exists", "output": matches[0].output}))
            return 0
        result = run_case(matches[0], args.gpu, args.executable, args.runtime, args.weights)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.all:
        return run_parallel(cases, args.gpus, args.executable, args.runtime, args.weights)

    raise SystemExit("Choose one of --inventory, --case, or --all")


if __name__ == "__main__":
    sys.exit(main())
