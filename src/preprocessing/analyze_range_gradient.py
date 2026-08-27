from pathlib import Path

import cv2
import numpy as np


INPUT_DIR = Path(
    r"C:\Users\Siddhartha\Desktop\EchoTrace\data\KLSG\archive"
)

SPLITS = ["train", "valid", "test"]


def calculate_gradient(image):
    height, width = image.shape

    quarter = width // 4

    left = image[:, :quarter]
    right = image[:, -quarter:]

    left_mean = float(np.mean(left))
    right_mean = float(np.mean(right))

    gradient = left_mean - right_mean

    return left_mean, right_mean, gradient


def analyze_split(split):

    image_dir = INPUT_DIR / split / "images"

    gradients = []
    left_means = []
    right_means = []

    images = sorted(
        image_dir.glob("*.jpg")
    )

    for image_path in images:

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        left, right, gradient = calculate_gradient(image)

        left_means.append(left)
        right_means.append(right)
        gradients.append(gradient)

    gradients = np.array(gradients)

    print()
    print("=" * 60)
    print(f"{split.upper()} RANGE GRADIENT ANALYSIS")
    print("=" * 60)

    print(f"Images analyzed : {len(gradients)}")

    if len(gradients) == 0:
        return

    print(
        f"Mean gradient   : {np.mean(gradients):.2f}"
    )

    print(
        f"Median gradient : {np.median(gradients):.2f}"
    )

    print(
        f"Minimum         : {np.min(gradients):.2f}"
    )

    print(
        f"Maximum         : {np.max(gradients):.2f}"
    )

    print(
        f"Std deviation   : {np.std(gradients):.2f}"
    )

    left_dominant = np.sum(gradients > 0)
    right_dominant = np.sum(gradients < 0)
    approximately_equal = np.sum(gradients == 0)

    total = len(gradients)

    print()
    print(
        f"Left brighter   : "
        f"{left_dominant} "
        f"({left_dominant / total * 100:.1f}%)"
    )

    print(
        f"Right brighter  : "
        f"{right_dominant} "
        f"({right_dominant / total * 100:.1f}%)"
    )

    print(
        f"Equal           : "
        f"{approximately_equal} "
        f"({approximately_equal / total * 100:.1f}%)"
    )


def main():

    print("=" * 60)
    print(" EchoTrace Dataset Range-Gradient Investigation")
    print("=" * 60)

    for split in SPLITS:
        analyze_split(split)


if __name__ == "__main__":
    main()
