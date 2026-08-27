from ultralytics import YOLO


MODEL_NAME = "yolov8n.pt"
DATASET = "data/KLSG_adaptive/data.yaml"


def train_model():

    model = YOLO(MODEL_NAME)

    model.train(
        data=DATASET,
        epochs=50,
        imgsz=640,
        batch=8,
        project="runs/echo_trace",
        name="adaptive_physics",
    )


if __name__ == "__main__":
    print("Starting EchoTrace adaptive training...")

    train_model()

    print("Adaptive training complete.")
