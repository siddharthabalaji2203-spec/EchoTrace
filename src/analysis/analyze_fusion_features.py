from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(
    r"C:\Users\Siddhartha\Desktop\EchoTrace"
)

CSV_PATH = (
    PROJECT_ROOT
    / "runs"
    / "fusion_analysis"
    / "sonar_detection_features_validation.csv"
)


FEATURES = [
    "confidence",
    "roi_mean",
    "roi_std",
    "roi_p10",
    "roi_p90",
    "roi_entropy",
    "roi_anomaly",
]


def main():

    print("=" * 60)
    print(" EchoTrace Fusion Feature Analysis")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)

    print(f"Rows loaded: {len(df)}")

    print()

    tp = df[df["status"] == "TP"]
    fp = df[df["status"] == "FP"]

    print(f"True positives : {len(tp)}")
    print(f"False positives: {len(fp)}")

    print()
    print("=" * 60)
    print(" TP vs FP FEATURE COMPARISON")
    print("=" * 60)

    print(
        f"{'Feature':<18}"
        f"{'TP Mean':>12}"
        f"{'FP Mean':>12}"
        f"{'Difference':>14}"
    )

    print("-" * 60)

    for feature in FEATURES:

        tp_mean = tp[feature].mean()
        fp_mean = fp[feature].mean()

        difference = tp_mean - fp_mean

        print(
            f"{feature:<18}"
            f"{tp_mean:>12.4f}"
            f"{fp_mean:>12.4f}"
            f"{difference:>14.4f}"
        )

    print()

    print("=" * 60)
    print(" TP MEDIANS")
    print("=" * 60)

    for feature in FEATURES:
        print(
            f"{feature:<18}: "
            f"{tp[feature].median():.4f}"
        )

    print()

    print("=" * 60)
    print(" FP MEDIANS")
    print("=" * 60)

    for feature in FEATURES:
        print(
            f"{feature:<18}: "
            f"{fp[feature].median():.4f}"
        )


if __name__ == "__main__":
    main()
