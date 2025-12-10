import numpy as np
from typing import List, Dict, Tuple, Optional


def iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    """
    Intersection-over-Union between two boxes in (x1, y1, x2, y2) format.
    """
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    if inter <= 0:
        return 0.0

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


class SimpleSortTracker:
    """
    Lightweight SORT-style tracker:
      - Greedy IoU matching between detections and existing tracks.
      - Smoothed bounding boxes for stability.
      - Tracks have IDs, age, and time_since_update.
    This is designed to stabilise AR overlays in a WebRTC/Streamlit setting.
    """

    def __init__(
        self,
        max_age: int = 10,
        min_iou: float = 0.3,
        smoothing: float = 0.5,
    ) -> None:
        """
        :param max_age: how many frames a track can survive without an update.
        :param min_iou: minimum IoU to associate detection with an existing track.
        :param smoothing: 0..1, how much to trust new box vs previous box
                          (higher = more jittery but more responsive).
        """
        self.max_age = max_age
        self.min_iou = min_iou
        self.smoothing = smoothing
        self._next_id = 1
        self.tracks: List[Dict] = []

    def _new_track(self, det: Dict) -> None:
        self.tracks.append(
            {
                "id": self._next_id,
                "bbox": tuple(det["bbox"]),
                "label": det.get("label", "object"),
                "score": float(det.get("score", 0.0)),
                "age": 1,
                "time_since_update": 0,
            }
        )
        self._next_id += 1

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        :param detections: list of dicts with keys:
                           - "bbox": (x1, y1, x2, y2)
                           - "score": float
                           - "label": str
        :return: list of active tracks with same keys + "id".
        """
        # Age all tracks
        for tr in self.tracks:
            tr["age"] += 1
            tr["time_since_update"] += 1

        if len(detections) == 0:
            # Remove stale tracks
            self.tracks = [
                tr for tr in self.tracks if tr["time_since_update"] <= self.max_age
            ]
            return list(self.tracks)

        # Sort detections by score (high → low) for greedy matching
        dets_sorted = sorted(detections, key=lambda d: d.get("score", 0.0), reverse=True)

        used_track_indices = set()
        used_det_indices = set()

        # Greedy IoU matching
        for d_idx, det in enumerate(dets_sorted):
            best_iou = 0.0
            best_t_idx: Optional[int] = None
            for t_idx, tr in enumerate(self.tracks):
                if t_idx in used_track_indices:
                    continue
                iou_val = iou(tr["bbox"], det["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_t_idx = t_idx

            if best_t_idx is not None and best_iou >= self.min_iou:
                # Update existing track with smoothed bbox
                tr = self.tracks[best_t_idx]
                old_box = np.array(tr["bbox"], dtype=float)
                new_box = np.array(det["bbox"], dtype=float)
                alpha = self.smoothing
                smoothed = (1.0 - alpha) * old_box + alpha * new_box

                tr["bbox"] = tuple(int(v) for v in smoothed.tolist())
                tr["label"] = det.get("label", tr["label"])
                tr["score"] = float(det.get("score", tr["score"]))
                tr["time_since_update"] = 0

                used_track_indices.add(best_t_idx)
                used_det_indices.add(d_idx)

        # Any detections not matched → new tracks
        for d_idx, det in enumerate(dets_sorted):
            if d_idx in used_det_indices:
                continue
            self._new_track(det)

        # Remove stale tracks
        self.tracks = [
            tr for tr in self.tracks if tr["time_since_update"] <= self.max_age
        ]

        return list(self.tracks)
