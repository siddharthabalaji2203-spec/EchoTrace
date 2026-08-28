import sys
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from pathlib import Path
from src.preprocessing.histogram import (
    calculate_statistics,
    calculate_entropy,
)
# ============================================================
# PATHS (Made Relative for Team Portability & Cloud Deployment)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
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
# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="EchoTrace",
    page_icon="🌊",
    layout="wide"
)
# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))
model = load_model()
def extract_roi(image, bbox):
    """
    Extract a YOLO detection bounding box from an image.

    Parameters
    ----------
    image : np.ndarray
        BGR image.
    bbox : tuple
        x1, y1, x2, y2 coordinates.

    Returns
    -------
    np.ndarray
        Cropped ROI.
    """

    x1, y1, x2, y2 = bbox

    height, width = image.shape[:2]

    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]

def calculate_fusion_score(confidence, entropy):
    """
    Experimental EchoTrace fusion score.

    Combines YOLO confidence with ROI entropy.
    This is a supporting detection-assessment score,
    not the YOLO confidence itself.
    """

    # Entropy observed in our validation dataset
    # is approximately 6.5–7.6.
    entropy_min = 6.5
    entropy_max = 7.6

    normalized_entropy = (
        (entropy - entropy_min)
        / (entropy_max - entropy_min)
    )

    normalized_entropy = max(
        0.0,
        min(1.0, normalized_entropy)
    )

    # Experimental weighting based on our
    # validation investigation.
    fusion_score = (
        0.75 * confidence
        + 0.25 * normalized_entropy
    )

    return fusion_score
def generate_demo_coordinates(
    count,
    base_lat=13.0800,
    base_lon=80.2700
):
    """
    Generate simulated coordinates for prototype demonstration.

    These coordinates are NOT real sonar/GPS measurements.
    """

    coordinates = []

    offsets = [
        (0.0000, 0.0000),
        (0.0020, 0.0030),
        (-0.0015, 0.0025),
        (0.0030, -0.0020),
        (-0.0020, -0.0030),
    ]

    for i in range(count):

        lat_offset, lon_offset = offsets[
            i % len(offsets)
        ]

        coordinates.append({
            "latitude": base_lat + lat_offset,
            "longitude": base_lon + lon_offset
        })

    return coordinates
# ============================================================
# HEADER
# ============================================================

st.title("🌊 ECHOTRACE")
st.subheader("AI-Powered Side-Scan Sonar Anomaly Detection")

st.divider()


# ============================================================
# IMAGE INPUT
# ============================================================

st.subheader("📡 Sonar Input")
uploaded_file = st.file_uploader(
    "Upload a Side-Scan Sonar image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:

    st.info(
        "Upload a sonar image to run EchoTrace inference."
    )

    st.divider()

    st.caption(
        "EchoTrace — AI-powered Side-Scan Sonar Anomaly Detection"
    )

    st.stop()

# ============================================================
# READ UPLOADED SONAR IMAGE
# ============================================================

image_bytes = uploaded_file.getvalue()

image_array = np.frombuffer(
    image_bytes,
    dtype=np.uint8
)

image = cv2.imdecode(
    image_array,
    cv2.IMREAD_COLOR
)

if image is None:
    st.error("Could not decode the uploaded sonar image.")
    st.stop()


# ============================================================
# RUN YOLO INFERENCE
# ============================================================

results = model.predict(
    source=image,
    conf=0.25,
    verbose=False
)

result = results[0]


# ============================================================
# EXTRACT DETECTIONS
# ============================================================

detections = []

if result.boxes is not None:

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = [
            float(v)
            for v in box.xyxy[0]
        ]

        detections.append({
            "class_id": class_id,
            "class_name": model.names[class_id],
            "confidence": confidence,
            "bbox": (x1, y1, x2, y2)
        })

# ============================================================
# ECHOTRACE DETECTION HISTORY
# ============================================================

detection_records = []

for detection in detections:

    roi = extract_roi(
        image,
        detection["bbox"]
    )

    if roi is None or roi.size == 0:
        continue

    roi_gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    roi_features = calculate_statistics(
        roi_gray
    )

    roi_features["entropy"] = calculate_entropy(
        roi_gray
    )

    fusion_score = calculate_fusion_score(
        detection["confidence"],
        roi_features["entropy"]
    )

    detection_records.append({
        "class": detection["class_name"],
        "confidence": detection["confidence"],
        "bbox": detection["bbox"],
        "roi_mean": roi_features["mean"],
        "roi_std": roi_features["std"],
        "roi_p10": roi_features["p10"],
        "roi_p90": roi_features["p90"],
        "roi_entropy": roi_features["entropy"],
        "fusion_score": fusion_score
    })

demo_coordinates = generate_demo_coordinates(
    len(detection_records)
)

for record, coordinate in zip(
    detection_records,
    demo_coordinates
):
    record["latitude"] = coordinate["latitude"]
    record["longitude"] = coordinate["longitude"]

# ============================================================
# DETECTION SUMMARY
# ============================================================

st.subheader("📊 Detection Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Detections",
        len(detections)
    )

with col2:
    st.metric(
        "Highest Confidence",
        (
            f"{max(d['confidence'] for d in detections) * 100:.2f}%"
            if detections
            else "—"
        )
    )

with col3:
    st.metric(
        "Detected Classes",
        len(set(
            d["class_name"]
            for d in detections
        ))
        if detections
        else 0
    )


st.divider()


# ============================================================
# DETECTION VISUALIZATION
# ============================================================

st.subheader("🔍 EchoTrace Detection")
annotated = result.plot()

annotated_rgb = cv2.cvtColor(
    annotated,
    cv2.COLOR_BGR2RGB
)

st.image(
    annotated_rgb,
    caption="YOLOv8n detection output",
    use_container_width=True
)

st.divider()

st.subheader("🗺️ Detection Locations")

st.warning(
    "DEMO MODE — coordinates are simulated. "
    "Real GPS integration is planned."
)

if detection_records:

    map_data = {
        "lat": [
            record["latitude"]
            for record in detection_records
        ],
        "lon": [
            record["longitude"]
            for record in detection_records
        ]
    }

    st.map(map_data)

else:

    st.info(
        "No detection locations available."
    )
# FIXED: Moved out of the loop context to prevent layout stacking bugs
total_detections = len(detection_records)
st.metric("Total Validated Detections", total_detections)
st.divider()

# ============================================================
# DETECTION HISTORY TABLE (FIXED: Now uses the original detection_records)
# ============================================================

st.subheader("📋 Detection History")

if detection_records:

    history_data = []

    for i, record in enumerate(detection_records, start=1):

        history_data.append({
    "Detection": i,
    "Class": record["class"],
    "YOLO Confidence": (
        f"{record['confidence'] * 100:.2f}%"
    ),
    "ROI Entropy": (
        f"{record['roi_entropy']:.2f}"
    ),
    "Fusion Score": (
        f"{record['fusion_score'] * 100:.1f}%"
    ),
    "Latitude": (
        f"{record['latitude']:.5f}"
    ),
    "Longitude": (
        f"{record['longitude']:.5f}"
    )
})

    st.dataframe(
        history_data,
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("No valid detections available.")


# ============================================================
# ROI SONAR FEATURE ANALYSIS
# ============================================================

# FIXED: Wrapped in an conditional statement to stop application crashes when 0 objects are detected
if detections:
    # Safely targets the most confident or last prominent detection element
    primary_detection = detections[0]
    
    roi = extract_roi(
        image,
        primary_detection["bbox"]
    )

    roi_features = None

    if roi is not None and roi.size > 0:

        roi_gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        roi_features = calculate_statistics(
            roi_gray
        )

        roi_features["entropy"] = calculate_entropy(
            roi_gray
        )
        
        fusion_score = calculate_fusion_score(
            primary_detection["confidence"],
            roi_features["entropy"]
        )
        
        st.subheader("🔬 Primary Target Feature Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "EchoTrace Fusion Score",
                f"{fusion_score * 100:.1f}%"
            )
            st.write(f"**Detection Class:** {primary_detection['class_name']}")
            st.write(f"**YOLO Confidence:** {primary_detection['confidence'] * 100:.2f}%")
        
        with col2:
            st.write(f"**ROI Mean Intensity:** {roi_features['mean']:.2f}")
            st.write(f"**ROI Signal Entropy:** {roi_features['entropy']:.2f}")
else:
    st.warning("Feature Analysis skipped: No target bounding box available to isolate.")
