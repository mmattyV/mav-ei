# vertical_corridor.py
import cv2
import numpy as np

# Lander body (purple)
LANDER_LOWER = np.array([110, 80, 140], dtype=np.uint8)
LANDER_UPPER = np.array([135, 255, 255], dtype=np.uint8)


def _get_lander_centroid(frame: np.ndarray) -> tuple[float, float] | None:
    """
    Use purple HSV mask to locate the lander centroid.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LANDER_LOWER, LANDER_UPPER)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


class VerticalCorridorVerifier:
    """
    While high in the air, is the lander roughly centered over the pad?
    """

    def __init__(self, corridor_frac: float = 0.2, min_height_frac: float = 0.35):
        """
        corridor_frac: fraction of screen width allowed from center.
        min_height_frac: only check when the lander is above this fraction
                         of the screen height (0 = top, 1 = bottom).
        """
        self.corridor_frac = corridor_frac
        self.min_height_frac = min_height_frac

    def __call__(self, frame: np.ndarray) -> bool:
        h, w, _ = frame.shape
        center_x = w / 2.0

        centroid = _get_lander_centroid(frame)
        if centroid is None:
            # If we can't see the lander, don't fail based on this verifier.
            return True

        cx, cy = centroid

        # Only care when "high" (near the top of the image).
        if cy > h * (1.0 - self.min_height_frac):
            return True

        max_dx = self.corridor_frac * w
        return abs(cx - center_x) <= max_dx
