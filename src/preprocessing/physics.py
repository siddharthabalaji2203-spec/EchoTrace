"""
EchoTrace - Physics-Based Sonar Preprocessing

Physically motivated preprocessing for Side-Scan Sonar imagery.
"""
from .histogram import calculate_statistics
from pathlib import Path

import cv2
import numpy as np


def load_grayscale(image_path: str | Path) -> np.ndarray:
    """
    Load a Side-Scan Sonar image as an 8-bit grayscale array.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    return image
def apply_range_compensation(
    image: np.ndarray,
    strength: float = 0.5
) -> np.ndarray:
    """
    Apply a simple range-dependent intensity compensation.

    Assumes image columns represent increasing sonar range.

    Parameters
    ----------
    image : np.ndarray
        8-bit grayscale sonar image.
    strength : float
        Compensation strength from 0.0 to 1.0.

    Returns
    -------
    np.ndarray
        Range-compensated 8-bit image.
    """

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0.0 and 1.0")

    image_float = image.astype(np.float32)

    height, width = image.shape

    # Normalized range: 0 = near sonar, 1 = far range
    range_axis = np.linspace(0.0, 1.0, width)

    # Gradual compensation factor
    compensation = 1.0 + (strength * range_axis)

    corrected = image_float * compensation[np.newaxis, :]

    corrected = np.clip(corrected, 0, 255)

    return corrected.astype(np.uint8)
def preprocess_image(
    image: np.ndarray,
    compensation_strength: float = 0.15
) -> np.ndarray:
    """
    Complete physics-based preprocessing pipeline.
    """

    corrected = apply_range_compensation(
        image,
        strength=compensation_strength
    )

    return corrected
def adaptive_preprocess(
    image: np.ndarray,
    low_threshold: float = 60.0,
    high_threshold: float = 140.0,
    compensation_strength: float = 0.15
) -> np.ndarray:
    """
    Apply range compensation adaptively based on image intensity.

    Dark images receive compensation.
    Normal-bright images receive minimal/no compensation.
    """

    mean_intensity = float(np.mean(image))

    if mean_intensity < low_threshold:
        strength = compensation_strength

    elif mean_intensity > high_threshold:
        strength = 0.0

    else:
        # Gradually scale compensation within the normal range
        strength = (
            compensation_strength
            * (high_threshold - mean_intensity)
            / (high_threshold - low_threshold)
        )

    return apply_range_compensation(
        image,
        strength=strength
    )
def adaptive_preprocess(
    image: np.ndarray,
    base_strength: float = 0.15
) -> np.ndarray:
    """
    Adapt range compensation to the intensity characteristics
    of the sonar image.
    """

    stats = calculate_statistics(image)

    mean = stats["mean"]
    p10 = stats["p10"]
    p90 = stats["p90"]
    entropy = stats["entropy"]

    dynamic_range = p90 - p10

    # Very dark frame: stronger compensation
    if mean < 60:
        strength = base_strength

    # Normal frame: moderate compensation
    elif mean < 100:
        strength = base_strength * 0.5

    # Bright frame: avoid unnecessary amplification
    else:
        strength = 0.0

    # Low dynamic range means limited useful intensity variation.
    # Reduce compensation rather than amplifying a flat/noisy image.
    if dynamic_range < 50:
        strength *= 0.5

    # Very low entropy indicates a relatively uniform image.
    if entropy < 6.0:
        strength *= 0.5

    return apply_range_compensation(
        image,
        strength=strength
    )
