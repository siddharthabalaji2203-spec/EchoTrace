"""
EchoTrace - YOLO + Sonar Feature Investigation

Stage 1:
Determine whether sonar intensity/statistical features
differentiate true-positive and false-positive detections.

No confidence fusion is performed here.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Siddhartha\Desktop\EchoTrace")

TEST_IMAGES = (
    PROJECT_ROOT
    / "data"
    / "KLSG"
    / "archive"
    / "valid"
    / "images"
)

TEST_LABELS = (
    PROJECT_ROOT
    / "data"
    / "KLSG"
    / "archive"
    / "valid"
    / "labels"
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "fusion_analysis"
)

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

CLASS_NAMES = [
    "aircraft",
    "fish",
    "other",
    "shipwreck",
]


# ============================================================
# IMAGE STATISTICS
# ============================================================

def calculate_entropy(image):
    """
    Calculate Shannon entropy of an 8-bit grayscale ROI.
    """

    if image.size == 0:
        return 0.0

    histogram = cv2.calcHist(
        [image],
        [0],
        None,
        [256],
        [0, 256]
    ).flatten()

    probabilities = histogram / histogram.sum()

    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return float(entropy)


def calculate_anomaly_score(image, roi):
    """
    Compare ROI mean against the global image mean/std.
    """

    global_mean = float(np.mean(image))
    global_std = float(np.std(image))

    if global_std == 0:
        global_std = 1.0

    roi_mean = float(np.mean(roi))

    return abs(roi_mean - global_mean) / global_std


def extract_roi_features(image, box):
    """
    Extract sonar statistics from a YOLO bounding box.
    """

    height, width = image.shape

    x1, y1, x2, y2 = box

    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return {
            "roi_mean": 0.0,
            "roi_std": 0.0,
            "roi_p10": 0.0,
            "roi_p90": 0.0,
            "roi_entropy": 0.0,
            "roi_anomaly": 0.0,
            "roi_area": 0,
        }

    roi = image[y1:y2, x1:x2]

    return {
        "roi_mean": float(np.mean(roi)),
        "roi_std": float(np.std(roi)),
        "roi_p10": float(np.percentile(roi, 10)),
        "roi_p90": float(np.percentile(roi, 90)),
        "roi_entropy": calculate_entropy(roi),
        "roi_anomaly": calculate_anomaly_score(
            image,
            roi
        ),
        "roi_area": int(roi.size),
    }


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(label_path, image_width, image_height):
    """
    Load YOLO-format ground truth labels.

    Format:
    class x_center y_center width height

    Coordinates are normalized to [0,1].
    """

    ground_truth = []

    if not label_path.exists():
        return ground_truth

    with open(label_path, "r") as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])

            x_center = float(values[1]) * image_width
            y_center = float(values[2]) * image_height

            box_width = float(values[3]) * image_width
            box_height = float(values[4]) * image_height

            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2
            x2 = x_center + box_width / 2
            y2 = y_center + box_height / 2

            ground_truth.append({
                "class_id": class_id,
                "box": [x1, y1, x2, y2],
            })

    return ground_truth


# ============================================================
# IOU
# ============================================================

def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


# ============================================================
# MATCH DETECTIONS TO GROUND TRUTH
# ============================================================

def match_detection(
    detection_class,
    detection_box,
    ground_truth,
    already_matched
):
    """
    Match a detection to the best unused ground-truth box
    of the same class.
    """

    best_iou = 0.0
    best_index = None

    for index, gt in enumerate(ground_truth):

        if index in already_matched:
            continue

        if gt["class_id"] != detection_class:
            continue

        iou = calculate_iou(
            detection_box,
            gt["box"]
        )

        if iou > best_iou:
            best_iou = iou
            best_index = index

    if best_iou >= IOU_THRESHOLD:
        return best_index, best_iou

    return None, best_iou


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 60)
    print(" EchoTrace YOLO + Sonar Feature Investigation")
    print("=" * 60)

    print(f"Model : {MODEL_PATH}")
    print(f"Images: {TEST_IMAGES}")
    print(f"Labels: {TEST_LABELS}")
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model = YOLO(str(MODEL_PATH))

    image_paths = sorted(
        TEST_IMAGES.glob("*.jpg")
    )

    if not image_paths:
        image_paths = sorted(
            TEST_IMAGES.glob("*.png")
        )

    print(
        f"Images found: {len(image_paths)}"
    )
    print()

    records = []

    total_detections = 0
    total_tp = 0
    total_fp = 0

    for image_number, image_path in enumerate(
        image_paths,
        start=1
    ):

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print(
                f"WARNING: Could not load {image_path.name}"
            )
            continue

        height, width = image.shape

        label_path = (
            TEST_LABELS
            / f"{image_path.stem}.txt"
        )

        ground_truth = load_ground_truth(
            label_path,
            width,
            height
        )

        results = model.predict(
            source=str(image_path),
            conf=CONF_THRESHOLD,
            verbose=False
        )

        result = results[0]

        matched_gt = set()

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, class_id, confidence in zip(
                boxes,
                classes,
                confidences
            ):

                class_id = int(class_id)
                confidence = float(confidence)

                detection_box = box.tolist()

                gt_index, iou = match_detection(
                    class_id,
                    detection_box,
                    ground_truth,
                    matched_gt
                )

                if gt_index is not None:

                    status = "TP"

                    matched_gt.add(gt_index)

                    total_tp += 1

                else:

                    status = "FP"

                    total_fp += 1

                features = extract_roi_features(
                    image,
                    detection_box
                )

                records.append({

                    "image": image_path.name,

                    "class_id": class_id,

                    "class_name": (
                        CLASS_NAMES[class_id]
                        if class_id < len(CLASS_NAMES)
                        else "unknown"
                    ),

                    "confidence": confidence,

                    "x1": detection_box[0],
                    "y1": detection_box[1],
                    "x2": detection_box[2],
                    "y2": detection_box[3],

                    "iou": iou,

                    "status": status,

                    **features,

                })

                total_detections += 1

        if (
            image_number % 10 == 0
            or image_number == len(image_paths)
        ):

            print(
                f"[{image_number}/{len(image_paths)}] "
                f"{image_path.name}"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    dataframe = pd.DataFrame(records)

    output_csv = (
        OUTPUT_DIR
        / "sonar_detection_features.csv"
    )

    dataframe.to_csv(
        output_csv,
        index=False
    )

    print()
    print("=" * 60)
    print(" Investigation complete.")
    print("=" * 60)

    print(
        f"Total detections : {total_detections}"
    )

    print(
        f"True positives   : {total_tp}"
    )

    print(
        f"False positives  : {total_fp}"
    )

    if total_detections > 0:

        print(
            f"TP proportion    : "
            f"{total_tp / total_detections:.4f}"
        )

    print()
    print(
        f"CSV saved to:"
    )
    print(output_csv)


if __name__ == "__main__":
    main()
