from ultralytics import YOLO


MODEL_NAME = "yolov8n.pt"
DATASET = "data/KLSG/archive/data.yaml"


def train_model():
    model = YOLO(MODEL_NAME)

    results = model.train(
        data=DATASET,
        epochs=50,
        imgsz=640,
        batch=8,
        project="runs/echo_trace",
        name="baseline",
    )

    return results


if __name__ == "__main__":
    print("Starting EchoTrace YOLOv8n baseline training...")
    train_model()
    print("Baseline training complete.")
