from pathlib import Path

from detector import EchoTraceDetector


PROJECT_ROOT = Path(
    r"C:\Users\Siddhartha\Desktop\EchoTrace"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "echo_trace"
    / "baseline"
    / "weights"
    / "best.pt"
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "KLSG"
    / "archive"
    / "test"
    / "images"
    / "000686_jpg.rf.2607951aa62c8eae9a9ce7a12f59f21a.jpg"
)


def main():

    print("=" * 60)
    print(" EchoTrace Inference Test")
    print("=" * 60)

    detector = EchoTraceDetector(
        model_path=MODEL_PATH,
        confidence=0.25
    )

    detections = detector.get_detections(
        IMAGE_PATH
    )

    print(f"\nImage: {IMAGE_PATH.name}")
    print(f"Detections: {len(detections)}")

    print()

    for index, detection in enumerate(
        detections,
        start=1
    ):

        print(
            f"Detection {index}"
        )

        print(
            f"  Class      : "
            f"{detection['class_name']}"
        )

        print(
            f"  Confidence : "
            f"{detection['confidence']:.4f}"
        )

        print(
            f"  Bounding Box: "
            f"{detection['bbox']}"
        )

        print()

    print("=" * 60)
    print(" Inference test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
