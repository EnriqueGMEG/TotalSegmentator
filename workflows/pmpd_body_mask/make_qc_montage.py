#!/usr/bin/env python3
"""Render orthogonal CT slices with an optional body-mask overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mask", type=Path)
    return parser.parse_args()


def oriented_slice(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return np.rot90(data[index, :, :])
    if axis == 1:
        return np.rot90(data[:, index, :])
    return np.rot90(data[:, :, index])


def main() -> None:
    args = parse_args()
    image = nib.as_closest_canonical(nib.load(args.image))
    data = image.get_fdata(dtype=np.float32)
    mask_data = None
    if args.mask is not None:
        mask = nib.as_closest_canonical(nib.load(args.mask))
        if mask.shape != image.shape or not np.allclose(mask.affine, image.affine, atol=1e-4, rtol=0.0):
            raise ValueError("Mask geometry does not match the CT")
        mask_data = np.asanyarray(mask.dataobj) > 0

    views = []
    for axis, label in ((2, "axial"), (1, "coronal"), (0, "sagittal")):
        for fraction in (0.3, 0.5, 0.7):
            index = min(data.shape[axis] - 1, max(0, round((data.shape[axis] - 1) * fraction)))
            views.append((axis, index, f"{label} {fraction:.0%}"))

    figure, axes = plt.subplots(3, 3, figsize=(12, 12), facecolor="black")
    for axis_object, (axis, index, title) in zip(axes.flat, views):
        ct_slice = oriented_slice(data, axis, index)
        axis_object.imshow(ct_slice, cmap="gray", vmin=-300, vmax=500, interpolation="nearest")
        if mask_data is not None:
            mask_slice = oriented_slice(mask_data, axis, index)
            overlay = np.ma.masked_where(~mask_slice, mask_slice)
            axis_object.imshow(overlay, cmap="autumn", alpha=0.28, interpolation="nearest", vmin=0, vmax=1)
            if mask_slice.any() and not mask_slice.all():
                axis_object.contour(mask_slice.astype(float), levels=[0.5], colors=["#00ffff"], linewidths=0.7)
        axis_object.set_title(title, color="white")
        axis_object.axis("off")
    figure.suptitle(args.image.name, color="white", fontsize=14)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=140, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
