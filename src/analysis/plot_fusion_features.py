from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(
    r"C:\Users\Siddhartha\Desktop\EchoTrace"
)

CSV_PATH = (
    PROJECT_ROOT
    / "runs"
    / "fusion_analysis"
    / "sonar_detection_features_validation.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "runs"
    / "fusion_analysis"
    / "confidence_entropy.png"
)


def main():

    df = pd.read_csv(CSV_PATH)

    tp = df[df["status"] == "TP"]
    fp = df[df["status"] == "FP"]

    plt.figure(figsize=(8, 6))

    plt.scatter(
        fp["confidence"],
        fp["roi_entropy"],
        label="False Positive",
        alpha=0.7
    )

    plt.scatter(
        tp["confidence"],
        tp["roi_entropy"],
        label="True Positive",
        alpha=0.7
    )

    plt.xlabel("YOLO Confidence")
    plt.ylabel("ROI Entropy")
    plt.title(
        "EchoTrace: YOLO Confidence vs ROI Entropy"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=200
    )

    plt.close()

    print("=" * 60)
    print("Plot generated successfully.")
    print("=" * 60)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
