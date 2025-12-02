# controlled_descent.py
import cv2
import numpy as np


def _get_lander_centroid(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _main_engine_on(frame: np.ndarray) -> bool:
    """Detect bright orange flame under the lander."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    return mask.sum() > 500  # heuristic threshold


class ControlledDescentVerifier:
    """
    From a single frame: if the lander is high in the air, we expect
    main engine flame under it (rough heuristic for braking vs free fall).
    """

    def __init__(self, min_height_frac: float = 0.4):
        self.min_height_frac = min_height_frac

    def __call__(self, frame: np.ndarray) -> bool:
        h, w, _ = frame.shape
        centroid = _get_lander_centroid(frame)
        if centroid is None:
            return True

        _, cy = centroid

        # Only care if somewhat high above the ground.
        if cy > h * (1.0 - self.min_height_frac):
            return True

        engine_on = _main_engine_on(frame)
        if not engine_on:
            return False
        return True
