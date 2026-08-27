from pathlib import Path
import shutil

from histogram import analyze_image
from physics import apply_range_compensation


INPUT_DIR = Path(r"C:\Users\Siddhartha\Desktop\EchoTrace\data\KLSG\archive")
OUTPUT_DIR = Path(r"C:\Users\Siddhartha\Desktop\EchoTrace\data\KLSG_adaptive")


STRENGTHS = [0.0, 0.075, 0.15]


def select_strength(image_path):
    stats = analyze_image(image_path)["statistics"]

    mean = stats["mean"]
    p10 = stats["p10"]
    p90 = stats["p90"]
    entropy = stats["entropy"]

    # Adaptive decision rules
    if mean < 60 and entropy < 7.0:
        return 0.15

    elif mean < 100 and (p90 - p10) > 100:
        return 0.075

    else:
        return 0.0


def process_split(split):
    input_images = INPUT_DIR / split / "images"
    input_labels = INPUT_DIR / split / "labels"

    output_images = OUTPUT_DIR / split / "images"
    output_labels = OUTPUT_DIR / split / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    images = list(input_images.glob("*"))

    print(f"\nProcessing {split}: {len(images)} images")

    for i, image_path in enumerate(images, 1):

        strength = select_strength(image_path)

        image = analyze_image(image_path)

        # Load actual image
        import cv2
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        processed = apply_range_compensation(
            img,
            strength=strength
        )

        output_path = output_images / image_path.name

        cv2.imwrite(str(output_path), processed)

        # Copy corresponding YOLO label
        label_path = input_labels / f"{image_path.stem}.txt"

        if label_path.exists():
            shutil.copy2(
                label_path,
                output_labels / label_path.name
            )

        if i % 100 == 0 or i == len(images):
            print(
                f"[{i}/{len(images)}] "
                f"{image_path.name} | "
                f"strength: {strength:.3f}"
            )


def main():

    print("======================================")
    print(" EchoTrace Adaptive Preprocessing")
    print("======================================")

    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(
        INPUT_DIR / "data.yaml",
        OUTPUT_DIR / "data.yaml"
    )

    for split in ["train", "valid", "test"]:
        process_split(split)

    print("\n======================================")
    print(" Adaptive preprocessing complete.")
    print("======================================")


if __name__ == "__main__":
    main()
