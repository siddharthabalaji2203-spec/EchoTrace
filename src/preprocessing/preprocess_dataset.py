"""
EchoTrace - Dataset Preprocessing Pipeline

Applies physics-based range compensation to the KLSG
Side-Scan Sonar dataset while preserving YOLO annotations.

Original dataset is never modified.
"""

from pathlib import Path
import shutil

import cv2

from physics import load_grayscale, preprocess_image
from histogram import calculate_statistics


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DATASET = PROJECT_ROOT / "data" / "KLSG" / "archive"
OUTPUT_DATASET = PROJECT_ROOT / "data" / "KLSG_processed"

COMPENSATION_STRENGTH = 0.15


# --------------------------------------------------
# Dataset preprocessing
# --------------------------------------------------

def process_split(split: str) -> None:

    input_images = INPUT_DATASET / split / "images"
    input_labels = INPUT_DATASET / split / "labels"

    output_images = OUTPUT_DATASET / split / "images"
    output_labels = OUTPUT_DATASET / split / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_files = list(input_images.glob("*"))

    print(f"\nProcessing {split}: {len(image_files)} images")

    for index, image_path in enumerate(image_files, start=1):

        if not image_path.is_file():
            continue

        try:
            # Load original sonar image
            image = load_grayscale(image_path)

            # Histogram/intensity analysis
            original_stats = calculate_statistics(image)

            # Physics-based range compensation
            processed = preprocess_image(
                image,
                compensation_strength=COMPENSATION_STRENGTH
            )

            # Analyze processed image
            processed_stats = calculate_statistics(processed)

            # Save processed image
            output_path = output_images / image_path.name
            cv2.imwrite(str(output_path), processed)

            # Preserve YOLO annotation
            label_path = input_labels / f"{image_path.stem}.txt"

            if label_path.exists():
                shutil.copy2(
                    label_path,
                    output_labels / label_path.name
                )

            if index % 100 == 0 or index == len(image_files):
                print(
                    f"[{index}/{len(image_files)}] "
                    f"{image_path.name} | "
                    f"mean: {original_stats['mean']:.2f} → "
                    f"{processed_stats['mean']:.2f}"
                )

        except Exception as error:
            print(f"ERROR processing {image_path}: {error}")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("======================================")
    print(" EchoTrace Dataset Preprocessing")
    print("======================================")

    print(f"Input : {INPUT_DATASET}")
    print(f"Output: {OUTPUT_DATASET}")
    print(f"Compensation strength: {COMPENSATION_STRENGTH}")

    # Copy data.yaml
    source_yaml = INPUT_DATASET / "data.yaml"
    output_yaml = OUTPUT_DATASET / "data.yaml"

    if source_yaml.exists():
        shutil.copy2(source_yaml, output_yaml)
        print("data.yaml copied successfully.")

    # Process all YOLO splits
    for split in ["train", "valid", "test"]:
        process_split(split)

    print("\n======================================")
    print(" Preprocessing complete.")
    print("======================================")
