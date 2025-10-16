"""Command‑line interface for the EduPIV project.

Run this module as a script to compute a velocity field from two images
and optionally save the results to a text file.  The script accepts
parameters for the interrogation window size, overlap, frame interval
and pixel scaling to convert displacements into velocity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.image as mpimg

from .edupiv import compute_piv, plot_vector_field


def _load_image(path: str) -> np.ndarray:
    """Load an image file and convert to a 2D grayscale array.

    The function uses matplotlib.image to read various image formats.
    If the image has multiple colour channels, it computes the mean
    across the last axis to obtain a single channel.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    img : ndarray
        2D array of floats with pixel intensities.
    """
    img = mpimg.imread(path)
    # Convert images with dtype uint8 or float to floats in [0, 1]
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img.astype(np.float32)


def parse_args(args: Tuple[str, ...]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PIV displacement/velocity field between two images"
    )
    parser.add_argument(
        "--image1", required=True, help="Path to the first (reference) frame"
    )
    parser.add_argument(
        "--image2", required=True, help="Path to the second frame"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=32,
        help="Size of the square interrogation window in pixels",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=16,
        help="Number of pixels by which adjacent windows overlap",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time between frames (seconds) for velocity computation",
    )
    parser.add_argument(
        "--pixel-scale",
        type=float,
        default=1.0,
        help="Physical length per pixel (e.g., mm/pixel) to convert displacement to velocity",
    )
    parser.add_argument(
        "--subpixel",
        action="store_true",
        help="Enable sub‑pixel peak estimation for higher accuracy",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the resulting vectors as a text file",
    )
    return parser.parse_args(args)


def main(argv: Tuple[str, ...] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    # Load images
    img1 = _load_image(args.image1)
    img2 = _load_image(args.image2)

    # Compute displacement field
    xs, ys, u, v = compute_piv(
        img1,
        img2,
        window_size=args.window_size,
        overlap=args.overlap,
        subpixel=args.subpixel,
    )

    # Convert displacement to velocity
    u_vel = (u * args.pixel_scale) / args.dt
    v_vel = (v * args.pixel_scale) / args.dt

    # Visualize
    plot_vector_field(xs, ys, u_vel, v_vel, scale=1.0, title="Velocity field")

    # Save to file if requested
    if args.output:
        out_path = Path(args.output)
        data = np.column_stack(
            [xs.ravel(), ys.ravel(), u_vel.ravel(), v_vel.ravel()]
        )
        header = "x y u v"
        np.savetxt(out_path, data, header=header)
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()