"""
EchoTrace - Histogram Analysis

Basic intensity analysis for Side-Scan Sonar imagery.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np


def load_grayscale(image_path: str | Path) -> np.ndarray:
    """
    Load a sonar image as an 8-bit grayscale NumPy array.

    Parameters
    ----------
    image_path : str | Path
        Path to the input image.

    Returns
    -------
    np.ndarray
        Grayscale image.

    Raises
    ------
    FileNotFoundError
        If the image cannot be loaded.
    """

    image_path = Path(image_path)

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    return image

def calculate_histogram(image: np.ndarray) -> np.ndarray:
    """
    Calculate the intensity histogram of a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        8-bit grayscale image.

    Returns
    -------
    np.ndarray
        Histogram containing the pixel count for each intensity
        value from 0 to 255.
    """

    histogram = cv2.calcHist(
        [image],
        [0],
        None,
        [256],
        [0, 256]
    )

    return histogram.flatten()

def calculate_statistics(image: np.ndarray) -> dict:
    """
    Calculate basic intensity statistics for a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        8-bit grayscale image.

    Returns
    -------
    dict
        Statistical descriptors of the image intensity.
    """

    pixels = image.astype(np.float32).ravel()

    statistics = {
        "mean": float(np.mean(pixels)),
        "median": float(np.median(pixels)),
        "std": float(np.std(pixels)),
        "min": float(np.min(pixels)),
        "max": float(np.max(pixels)),
        "p10": float(np.percentile(pixels, 10)),
        "p25": float(np.percentile(pixels, 25)),
        "p75": float(np.percentile(pixels, 75)),
        "p90": float(np.percentile(pixels, 90)),
    }

    return statistics

def calculate_entropy(image: np.ndarray) -> float:
    """
    Calculate Shannon entropy of a grayscale image.
    """

    histogram = calculate_histogram(image)

    probabilities = histogram / histogram.sum()

    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return float(entropy)

def analyze_image(image_path: str | Path) -> dict:
    """
    Perform complete histogram and intensity analysis.

    Parameters
    ----------
    image_path : str | Path
        Path to the grayscale-compatible sonar image.

    Returns
    -------
    dict
        Image metadata, histogram, and intensity statistics.
    """

    image = load_grayscale(image_path)

    histogram = calculate_histogram(image)
    statistics = calculate_statistics(image)

    statistics["entropy"] = calculate_entropy(image)

    return {
        "image_path": str(image_path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "statistics": statistics,
        "histogram": histogram,
    }

def save_histogram_plot(
    image: np.ndarray,
    output_path: str | Path
) -> None:
    """
    Save the image intensity histogram as a plot.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    output_path : str | Path
        Where to save the histogram plot.
    """

    histogram = calculate_histogram(image)

    plt.figure(figsize=(8, 4))

    plt.plot(histogram)

    plt.xlabel("Intensity")
    plt.ylabel("Pixel Count")
    plt.title("EchoTrace Sonar Intensity Histogram")
    plt.xlim(0, 255)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def calculate_local_histograms(
    image: np.ndarray,
    grid_size: tuple[int, int] = (4, 4)
) -> list[np.ndarray]:
    """
    Calculate intensity histograms for local regions of an image.

    Parameters
    ----------
    image : np.ndarray
        8-bit grayscale image.
    grid_size : tuple[int, int]
        Number of rows and columns to divide the image into.

    Returns
    -------
    list[np.ndarray]
        A list containing one 256-bin histogram per region.
    """

    rows, cols = grid_size
    height, width = image.shape

    region_height = height // rows
    region_width = width // cols

    histograms = []

    for row in range(rows):
        for col in range(cols):
            y_start = row * region_height
            y_end = (
                (row + 1) * region_height
                if row < rows - 1
                else height
            )

            x_start = col * region_width
            x_end = (
                (col + 1) * region_width
                if col < cols - 1
                else width
            )

            region = image[y_start:y_end, x_start:x_end]

            histogram = calculate_histogram(region)
            histograms.append(histogram)

    return histograms

def calculate_local_anomalies(
    image: np.ndarray,
    grid_size: tuple[int, int] = (4, 4)
) -> list[dict]:
    """
    Calculate local intensity anomaly scores.

    Each region is compared against the global image mean
    and standard deviation.

    Returns
    -------
    list[dict]
        Statistics and anomaly score for each region.
    """

    global_mean = float(np.mean(image))
    global_std = float(np.std(image))

    if global_std == 0:
        global_std = 1.0

    rows, cols = grid_size
    height, width = image.shape

    region_height = height // rows
    region_width = width // cols

    anomalies = []

    for row in range(rows):
        for col in range(cols):

            y_start = row * region_height
            y_end = (row + 1) * region_height if row < rows - 1 else height

            x_start = col * region_width
            x_end = (col + 1) * region_width if col < cols - 1 else width

            region = image[y_start:y_end, x_start:x_end]

            local_mean = float(np.mean(region))
            local_std = float(np.std(region))

            anomaly_score = abs(local_mean - global_mean) / global_std

            anomalies.append({
                "row": row,
                "col": col,
                "mean": local_mean,
                "std": local_std,
                "anomaly_score": float(anomaly_score),
            })

    return anomalies
