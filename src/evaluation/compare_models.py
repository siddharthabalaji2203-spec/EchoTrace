from pathlib import Path
import csv

from ultralytics import YOLO


# ============================================================
# EchoTrace - Three Model Comparison
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Siddhartha\Desktop\EchoTrace")

TEST_DIR = PROJECT_ROOT / "data" / "KLSG" / "archive" / "test" / "images"

MODEL_DIR = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "echo_trace"
)

OUTPUT_DIR = PROJECT_ROOT / "runs" / "comparison"

MODELS = {
    "baseline": MODEL_DIR / "baseline" / "weights" / "best.pt",
    "physics_histogram": MODEL_DIR / "physics_histogram" / "weights" / "best.pt",
    "adaptive_physics": MODEL_DIR / "adaptive_physics" / "weights" / "best.pt",
}

CONFIDENCE = 0.25
IMAGE_SIZE = 640


def check_paths():
    print("\nChecking paths...")

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found:\n{TEST_DIR}"
        )

    print(f"Test images: {TEST_DIR}")

    for name, path in MODELS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} model not found:\n{path}"
            )

        print(f"{name}: {path}")

    print("All paths verified.\n")


def run_model(model_name, model_path):
    print("=" * 60)
    print(f"Running: {model_name}")
    print("=" * 60)

    model = YOLO(str(model_path))

    output_name = model_name

    results = model.predict(
        source=str(TEST_DIR),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(OUTPUT_DIR),
        name=output_name,
        exist_ok=True,
        verbose=False,
    )

    csv_path = OUTPUT_DIR / f"{model_name}_detections.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "image",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ])

        detection_count = 0

        for result in results:

            image_name = Path(result.path).name

            if result.boxes is None:
                continue

            boxes = result.boxes

            for i in range(len(boxes)):

                class_id = int(
                    boxes.cls[i].item()
                )

                class_name = result.names[class_id]

                confidence = float(
                    boxes.conf[i].item()
                )

                x1, y1, x2, y2 = (
                    boxes.xyxy[i].tolist()
                )

                writer.writerow([
                    image_name,
                    class_id,
                    class_name,
                    round(confidence, 4),
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                ])

                detection_count += 1

    print(f"Images processed: {len(results)}")
    print(f"Detections: {detection_count}")
    print(f"CSV saved: {csv_path}")
    print(f"Images saved: {OUTPUT_DIR / output_name}")
    print()


def main():

    print("=" * 60)
    print(" EchoTrace Three-Model Comparison")
    print("=" * 60)

    print(f"Test set : {TEST_DIR}")
    print(f"Output   : {OUTPUT_DIR}")
    print(f"Confidence threshold: {CONFIDENCE}")
    print(f"Image size: {IMAGE_SIZE}")

    check_paths()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for model_name, model_path in MODELS.items():
        run_model(model_name, model_path)

    print("=" * 60)
    print(" Comparison complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
