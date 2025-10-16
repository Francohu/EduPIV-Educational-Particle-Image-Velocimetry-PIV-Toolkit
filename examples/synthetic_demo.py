"""Demonstrate EduPIV on synthetic images.

This script generates a pair of synthetic particle images with a
prescribed translation, computes the displacement field and visualizes
the result.  It also prints the mean estimated displacement to
illustrate accuracy.
"""

import numpy as np

from edupiv import generate_synthetic_images, compute_piv, plot_vector_field


def main() -> None:
    # Generate synthetic images with known shift
    image_size = (256, 256)
    true_shift = (3, -2)  # (dx, dy)
    img1, img2, shift = generate_synthetic_images(
        image_size=image_size, shift=true_shift, particle_density=0.02, seed=0
    )

    # Compute PIV displacement field
    xs, ys, u, v = compute_piv(
        img1, img2, window_size=32, overlap=16, subpixel=True
    )

    # Visualize the displacement field
    plot_vector_field(xs, ys, u, v, scale=1.0, title="Synthetic PIV displacement")

    # Print true and mean estimated shift for comparison
    mean_dx = np.mean(u)
    mean_dy = np.mean(v)
    print(f"True shift: dx={shift[0]}, dy={shift[1]}")
    print(f"Estimated mean shift: dx={mean_dx:.3f}, dy={mean_dy:.3f}")


if __name__ == "__main__":
    main()