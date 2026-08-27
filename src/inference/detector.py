from pathlib import Path

from ultralytics import YOLO


class EchoTraceDetector:
    """
    EchoTrace YOLO inference wrapper.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25
    ):
        self.model_path = Path(model_path)
        self.confidence = confidence

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

    def predict(self, image_path: str | Path):
        """
        Run YOLO inference on a sonar image.
        """

        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence,
            verbose=False
        )

        return results[0]

    def get_detections(self, image_path: str | Path):
        """
        Return detections in a dashboard-friendly format.
        """

        result = self.predict(image_path)

        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = (
                float(value)
                for value in box.xyxy[0]
            )

            detections.append({
                "class_id": class_id,
                "class_name": self.model.names[class_id],
                "confidence": confidence,
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            })

        return detections
