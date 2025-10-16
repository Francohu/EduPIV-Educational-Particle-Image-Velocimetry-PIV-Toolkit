"""Core algorithms for EduPIV.

This module implements synthetic PIV image generation, a simple
cross‑correlation‐based PIV algorithm and tools for visualizing the
resulting velocity fields.  The algorithms are designed for education
and prototyping rather than maximum performance.  They rely on
NumPy and SciPy for array manipulation and 2D correlation.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def generate_synthetic_images(
    image_size: Tuple[int, int] = (256, 256),
    shift: Tuple[int, int] = (2, 0),
    particle_density: float = 0.01,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a pair of synthetic PIV frames with a uniform translation.

    The first frame contains randomly distributed point particles.  The
    second frame is created by shifting the first frame by ``shift`` pixels
    without wrap–around (regions outside the original image are
    zero‑padded).

    Parameters
    ----------
    image_size : tuple of int, optional
        Size of the generated images ``(height, width)``.
    shift : tuple of int, optional
        Translation applied to the second frame given as ``(dx, dy)`` in
        pixels.  Positive ``dx`` shifts particles to the right and
        positive ``dy`` shifts particles downward.
    particle_density : float, optional
        Fraction of pixels that contain a particle in the first frame.
    seed : int, optional
        Seed for NumPy’s random number generator for reproducibility.

    Returns
    -------
    img1 : ndarray
        The first frame as a float32 array of shape ``image_size``.
    img2 : ndarray
        The second frame after applying the translation.  Pixels moved
        outside the image boundaries are discarded; empty regions are
        filled with zeros.
    true_shift : ndarray of shape (2,)
        The true displacement vector ``(dx, dy)`` for reference.
    """
    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)

    height, width = image_size
    num_pixels = height * width
    num_particles = int(num_pixels * particle_density)

    # Create an empty image and randomly place particles
    img1 = np.zeros((height, width), dtype=np.float32)
    # Select random indices without replacement
    indices = rng.choice(num_pixels, size=num_particles, replace=False)
    ys = indices // width
    xs = indices % width
    img1[ys, xs] = 1.0

    # Create second frame by translating the first frame
    dx, dy = shift  # positive dx shifts to the right, dy shifts downwards
    img2 = np.zeros_like(img1)

    # Compute destination coordinates for the shifted image
    y_start_dst = max(0, dy)
    y_end_dst = min(height, height + dy)
    x_start_dst = max(0, dx)
    x_end_dst = min(width, width + dx)

    # Compute source region in img1
    y_start_src = y_start_dst - dy
    y_end_src = y_end_dst - dy
    x_start_src = x_start_dst - dx
    x_end_src = x_end_dst - dx

    img2[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = img1[
        y_start_src:y_end_src, x_start_src:x_end_src
    ]

    true_shift = np.array([dx, dy], dtype=np.float32)
    return img1, img2, true_shift


def _quadratic_subpixel(corr: np.ndarray, peak_y: int, peak_x: int) -> Tuple[float, float]:
    """Estimate sub‑pixel peak location using 1D quadratic fit.

    This helper function fits a 1D parabola to the correlation peak
    and its immediate neighbors along x and y directions to estimate
    sub‑pixel displacements.  The method returns fractional offsets
    relative to the integer peak position.

    Parameters
    ----------
    corr : ndarray
        2D correlation map.
    peak_y : int
        Row index of the peak.
    peak_x : int
        Column index of the peak.

    Returns
    -------
    (dy_sub, dx_sub) : tuple of float
        Fractional offsets in y and x directions.  These values are in
        pixels and should be added to the integer displacement.
    """
    dy_sub = 0.0
    dx_sub = 0.0
    h, w = corr.shape

    # Sub‑pixel estimation along y
    if 1 <= peak_y < h - 1:
        c0 = corr[peak_y, peak_x]
        c_up = corr[peak_y - 1, peak_x]
        c_down = corr[peak_y + 1, peak_x]
        denom = c_up - 2 * c0 + c_down
        if denom != 0:
            dy_sub = 0.5 * (c_up - c_down) / denom
    # Sub‑pixel estimation along x
    if 1 <= peak_x < w - 1:
        c0 = corr[peak_y, peak_x]
        c_left = corr[peak_y, peak_x - 1]
        c_right = corr[peak_y, peak_x + 1]
        denom = c_left - 2 * c0 + c_right
        if denom != 0:
            dx_sub = 0.5 * (c_left - c_right) / denom

    return dy_sub, dx_sub


def compute_piv(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int = 32,
    overlap: int = 16,
    subpixel: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute displacement field between two frames using cross‑correlation.

    Both input images must be 2D arrays of the same shape.  The images
    are divided into square interrogation windows of size ``window_size``.
    Windows overlap by ``overlap`` pixels.  For each window, the
    displacement that maximizes the cross‑correlation between the two
    images is computed.  Optionally, a sub‑pixel refinement of the
    displacement can be performed using a quadratic fit.

    Parameters
    ----------
    img1, img2 : ndarray
        Two consecutive frames containing particle images.  They must have
        identical shapes and should be pre‑processed (e.g., converted to
        grayscale).
    window_size : int, optional
        Size of the square interrogation window in pixels.  Typical values
        are 16, 32 or 64.
    overlap : int, optional
        Number of pixels by which adjacent interrogation windows overlap.
        Overlap increases spatial resolution at the cost of computation.
    subpixel : bool, optional
        Whether to perform sub‑pixel peak estimation.

    Returns
    -------
    xs : ndarray of shape (n_rows, n_cols)
        x‑coordinates of the centres of the interrogation windows.
    ys : ndarray of shape (n_rows, n_cols)
        y‑coordinates of the centres of the interrogation windows.
    u : ndarray of shape (n_rows, n_cols)
        Displacement in x direction (pixels) for each window.
    v : ndarray of shape (n_rows, n_cols)
        Displacement in y direction (pixels) for each window.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same shape")

    # Convert to float for correlation
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    height, width = img1.shape
    step = window_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than window_size")

    # Determine window starting indices
    y_starts = np.arange(0, height - window_size + 1, step)
    x_starts = np.arange(0, width - window_size + 1, step)

    # Preallocate displacement arrays
    u = np.zeros((len(y_starts), len(x_starts)), dtype=np.float32)
    v = np.zeros_like(u)

    # Compute centre coordinates
    ys_centres = y_starts + window_size / 2.0
    xs_centres = x_starts + window_size / 2.0
    xs_grid, ys_grid = np.meshgrid(xs_centres, ys_centres)

    # Loop over interrogation windows
    for i, y in enumerate(y_starts):
        for j, x in enumerate(x_starts):
            win1 = img1[y : y + window_size, x : x + window_size]
            win2 = img2[y : y + window_size, x : x + window_size]

            # Normalize windows by removing mean to reduce bias
            win1m = win1 - np.mean(win1)
            win2m = win2 - np.mean(win2)

            # Compute full correlation map via FFT-based convolution
            corr = correlate2d(win2m, win1m, mode="full", boundary='fill')

            # Find integer peak location
            peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
            # Convert to displacement relative to zero-lag at centre
            # In 'full' mode, correlation shape is (2*win_h-1, 2*win_w-1)
            dy_int = peak_y - (window_size - 1)
            dx_int = peak_x - (window_size - 1)

            if subpixel:
                dy_sub, dx_sub = _quadratic_subpixel(corr, peak_y, peak_x)
                v[i, j] = dy_int + dy_sub
                u[i, j] = dx_int + dx_sub
            else:
                v[i, j] = dy_int
                u[i, j] = dx_int

    return xs_grid, ys_grid, u, v


def plot_vector_field(
    xs: np.ndarray,
    ys: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    scale: float = 1.0,
    title: Optional[str] = None,
) -> None:
    """Plot a displacement or velocity field using quiver.

    Parameters
    ----------
    xs, ys : ndarray
        Coordinate grids of shape ``(n_rows, n_cols)`` containing the x and y
        positions of the vectors.
    u, v : ndarray
        Components of the vectors (displacement or velocity) in x and y
        directions.  Must have the same shape as ``xs`` and ``ys``.
    scale : float, optional
        Scaling factor for the arrow lengths.  Larger values shrink the
        arrows.  The default of 1.0 leaves arrow lengths proportional to
        the vector magnitude.
    title : str, optional
        Title for the plot.
    """
    magnitude = np.sqrt(u ** 2 + v ** 2)
    fig, ax = plt.subplots(figsize=(6, 6))
    quiver = ax.quiver(
        xs,
        ys,
        u,
        v,
        magnitude,
        cmap="plasma",
        angles="xy",
        scale_units="xy",
        scale=1.0 / scale,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Match image coordinate system
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(quiver, ax=ax)
    cbar.set_label("Vector magnitude")
    plt.tight_layout()
    plt.show()
