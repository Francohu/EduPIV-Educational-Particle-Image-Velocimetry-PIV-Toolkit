"""EduPIV package exports key functions.

This module makes it convenient to import the primary functions from
``edupiv.edupiv`` directly from the top-level package.
"""

from .edupiv import generate_synthetic_images, compute_piv, plot_vector_field

__all__ = [
    "generate_synthetic_images",
    "compute_piv",
    "plot_vector_field",
]