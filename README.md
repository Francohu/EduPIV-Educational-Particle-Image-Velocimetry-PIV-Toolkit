# EduPIV – Educational Particle Image Velocimetry (PIV) Toolkit

EduPIV is a Python‑based educational toolkit for **Particle Image Velocimetry** (PIV). It helps students and researchers understand the fundamentals of PIV and quickly process experimental or synthetic data. The project includes synthetic PIV image generation, cross‑correlation‑based velocity field computation and vector field visualization; it is implemented using NumPy, SciPy and Matplotlib.

## Background and key features

Particle Image Velocimetry is a non‑intrusive method for measuring instantaneous flow velocities. Two consecutive images of seeded particles are divided into small interrogation windows, and a cross‑correlation is performed on each pair of windows to find the displacement that maximizes the correlation【875893234148448†L280-L349】. This produces two‑dimensional or three‑dimensional vector fields for the entire flow region instead of a single‑point measurement【962836915831531†L156-L170】.

Key features of EduPIV include:

* **Synthetic data generator** – generate pairs of random particle images with specified particle density, image size and translation for testing and teaching.
* **Pure Python cross‑correlation algorithm** – use SciPy’s `correlate2d` to compute correlation maps for each window and determine displacements; window size and overlap are configurable. The approach follows the sliding dot‑product method commonly used in PIV【875893234148448†L342-L349】.
* **Visualization functions** – plot velocity vectors at the centre of each interrogation window with colour encoding for magnitude.
* **Modular design** – simple structure makes it easy to extend the code (e.g., multi‑grid window deformation, multi‑pass schemes or sub‑pixel peak fitting).

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Francohu/EduPIV.git
cd EduPIV
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes `numpy`, `scipy` and `matplotlib`.

## Usage

### 1. Generate synthetic data

Use `generate_synthetic_images` to create two frames of random particles with a known translation:

```python
from edupiv import generate_synthetic_images, compute_piv, plot_vector_field

# generate two 256×256 images with 1 % particle density and a shift of (2, -1)
img1, img2, true_shift = generate_synthetic_images(
    image_size=(256, 256), shift=(2, -1), particle_density=0.01, seed=42
)
```

### 2. Compute velocity field

Call `compute_piv` to divide the images into interrogation windows and determine the displacement of each region:

```python
xs, ys, u, v = compute_piv(
    img1, img2,
    window_size=32,
    overlap=16,
    subpixel=False
)
```

`xs` and `ys` are coordinate grids of window centres; `u` and `v` are the corresponding displacements in pixels. Dividing displacements by the frame interval and multiplying by pixel‑to‑length scaling converts to physical velocity.

### 3. Visualize results

Use `plot_vector_field` to display the velocity vectors:

```python
plot_vector_field(xs, ys, u, v, scale=1.0, title="PIV Velocity Field")
```

### 4. Command‑line interface

EduPIV provides a CLI (`edupiv/cli.py`) for processing image pairs from the terminal:

```bash
python -m edupiv.cli --image1 path/to/frame1.png --image2 path/to/frame2.png \
    --window-size 32 --overlap 16 --dt 0.02 --pixel-scale 1.0 \
    --output vectors.txt
```

Parameters:

* `--image1`, `--image2` – paths to the two grayscale images.
* `--window-size` – interrogation window size (pixels).
* `--overlap` – overlapping pixels between adjacent windows.
* `--dt` – time between frames (seconds).
* `--pixel-scale` – physical size per pixel (e.g., mm/pixel).
* `--output` – optional path to save the computed vectors (columns `x`, `y`, `u`, `v`).

The script will display the velocity field and save results if requested.

## Directory structure

```
EduPIV/
├── README.md         # project description
├── requirements.txt  # dependencies list
├── edupiv/
│   ├── __init__.py
│   ├── edupiv.py     # core algorithms
│   └── cli.py        # command-line interface
└── examples/
    ├── synthetic_demo.py  # synthetic data example
    └── real_data_demo.py  # template for real data
```

## License

This project is released under the MIT License. Contributions are welcome.