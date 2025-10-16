"""Template for running EduPIV on real experimental images.

This example loads two user‑supplied grayscale images and computes the
velocity field using EduPIV.  Replace `frame1_path` and `frame2_path`
with the paths to your own images.  The script demonstrates how to
convert displacements into physical velocities using the time step and
pixel scaling.
"""

import numpy as np
import matplotlib.image as mpimg

from edupiv import compute_piv, plot_vector_field


def load_grayscale(path: str) -> np.ndarray:
    img = mpimg.imread(path)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img.astype(np.float32)


def main() -> None:
    # Specify paths to your experimental images (replace with actual files)
    frame1_path = "path/to/your/frame1.png"
    frame2_path = "path/to/your/frame2.png"

    img1 = load_grayscale(frame1_path)
    img2 = load_grayscale(frame2_path)

    # Compute displacement field
    xs, ys, u, v = compute_piv(
        img1, img2, window_size=32, overlap=16, subpixel=True
    )

    # Convert displacement to physical velocity
    dt = 0.01  # seconds between frames
    pixel_scale = 0.1  # mm per pixel (example)
    u_vel = u * pixel_scale / dt
    v_vel = v * pixel_scale / dt

    # Visualize velocity field
    plot_vector_field(xs, ys, u_vel, v_vel, scale=1.0, title="Experimental PIV velocity")


if __name__ == "__main__":
    main()