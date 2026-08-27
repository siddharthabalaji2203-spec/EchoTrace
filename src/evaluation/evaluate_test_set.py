from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(r"C:\Users\Siddhartha\Desktop\EchoTrace")

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "KLSG"
    / "archive"
    / "test"
)

MODELS = {
    "baseline": (
        PROJECT_ROOT
        / "runs"
        / "detect"
        / "runs"
        / "echo_trace"
        / "baseline"
        / "weights"
        / "best.pt"
    ),

    "physics_histogram": (
        PROJECT_ROOT
        / "runs"
        / "detect"
        / "runs"
        / "echo_trace"
        / "physics_histogram"
        / "weights"
        / "best.pt"
    ),

    "adaptive_physics": (
        PROJECT_ROOT
        / "runs"
        / "detect"
        / "runs"
        / "echo_trace"
        / "adaptive_physics"
        / "weights"
        / "best.pt"
    ),
}


def main():

    print("=" * 60)
    print(" EchoTrace Test-Set Evaluation")
    print("=" * 60)

    print(f"Test set: {TEST_DIR}")
    print()

    for name, model_path in MODELS.items():

        print("=" * 60)
        print(f"Evaluating: {name}")
        print("=" * 60)

        if not model_path.exists():
            print(f"ERROR: Model not found:")
            print(model_path)
            continue

        model = YOLO(str(model_path))

        metrics = model.val(
            data=str(TEST_DIR.parent / "data.yaml"),
            split="test",
            imgsz=640,
            conf=0.001,
            iou=0.7,
            verbose=True,
        )

        print()
        print(f"{name} RESULTS")
        print("-" * 40)

        print(f"Precision : {metrics.box.mp:.4f}")
        print(f"Recall    : {metrics.box.mr:.4f}")
        print(f"mAP50     : {metrics.box.map50:.4f}")
        print(f"mAP50-95  : {metrics.box.map:.4f}")

        print()


if __name__ == "__main__":
    main()
