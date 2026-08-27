"""
EchoTrace - Preprocessed YOLOv8n Training
"""

from ultralytics import YOLO


MODEL_NAME = "yolov8n.pt"
DATASET = "data/KLSG_processed/data.yaml"


def train_model():
    model = YOLO(MODEL_NAME)

    results = model.train(
        data=DATASET,
        epochs=50,
        imgsz=640,
        batch=8,
        project="runs/echo_trace",
        name="physics_histogram",
    )

    return results


if __name__ == "__main__":
    print("Starting EchoTrace preprocessed training...")
    train_model()
    print("Preprocessed training complete.")
