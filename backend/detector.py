import numpy as np

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False


class Detector:
    """
    Lightweight wrapper around YOLO for object detection.

    Now supports:
      - detect_all(img): list of {label, bbox, score}
      - detect_single(img): best detection (for compatibility)
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None

        if _HAS_YOLO:
            try:
                self.model = YOLO("yolov8n.pt")
            except Exception:
                self.model = None

    def _run_model(self, img: np.ndarray):
        if self.model is None:
            return None
        return self.model.predict(
            source=img,
            conf=self.confidence_threshold,
            verbose=False,
        )

    def detect_all(self, img: np.ndarray):
        """
        Return all detections as a list of dicts:
        [
          {"label": str, "bbox": (x1,y1,x2,y2), "score": float},
          ...
        ]
        """
        h, w, _ = img.shape

        results = self._run_model(img)
        if results is None or len(results) == 0:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections = []
        names = results[0].names

        for b in boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            label = names.get(cls_id, "object")

            x1, y1, x2, y2 = b.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Clip to frame
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            detections.append(
                {
                    "label": label,
                    "bbox": (x1, y1, x2, y2),
                    "score": conf,
                }
            )

        return detections

    def detect_single(self, img: np.ndarray):
        """
        Kept for compatibility: return best detection only.
        """
        dets = self.detect_all(img)
        if not dets:
            return None, None

        best = max(dets, key=lambda d: d["score"])
        return best["label"], best["bbox"]
