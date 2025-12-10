import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False


class Detector:
    """
    Lightweight wrapper around YOLO for object detection.
    Returns a single best bounding box and label for simplicity.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None

        if _HAS_YOLO:
            try:
                # Uses the small model by default. User can swap to custom farm model later.
                self.model = YOLO("yolov8n.pt")
            except Exception:
                self.model = None

    def detect_single(self, img: np.ndarray):
        """
        Detect the most confident object in the frame.
        Returns: (label: str or None, bbox: (x1, y1, x2, y2) or None)
        """
        h, w, _ = img.shape

        if self.model is None:
            # Fallback: no model found, return None
            return None, None

        results = self.model.predict(
            source=img,
            conf=self.confidence_threshold,
            verbose=False,
        )

        if not results or len(results) == 0:
            return None, None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, None

        # Take the highest confidence box
        best_idx = int(boxes.conf.argmax().item())
        best_box = boxes[best_idx]
        cls_id = int(best_box.cls.item())
        label = results[0].names.get(cls_id, "object")

        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Clip to frame
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        return label, (x1, y1, x2, y2)
