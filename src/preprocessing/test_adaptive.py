from pathlib import Path
import cv2
import numpy as np

from physics import adaptive_preprocess
from histogram import calculate_statistics, calculate_entropy


IMAGE_PATH = Path(
    r"C:\Users\Siddhartha\Desktop\EchoTrace\data\KLSG\archive\valid\images\000010_jpg.rf.efb6cf8c3257d078e4fffc1451283b5b.jpg"
)


images = list(IMAGE_PATH.glob("*.jpg"))[:10]

for image_path in images:

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    stats = calculate_statistics(image)

    mean = stats["mean"]
    p10 = stats["p10"]
    p90 = stats["p90"]
    entropy = calculate_entropy(image)

    if mean < 60:
        strength = 0.15
    elif mean < 100:
        strength = 0.075
    else:
        strength = 0.0

    if (p90 - p10) < 50:
        strength *= 0.5

    if entropy < 6.0:
        strength *= 0.5

    print(
        f"{image_path.name}\n"
        f"  Mean: {mean:.2f}\n"
        f"  P10-P90: {p10:.2f} - {p90:.2f}\n"
        f"  Entropy: {entropy:.2f}\n"
        f"  Selected strength: {strength:.4f}\n"
    )
