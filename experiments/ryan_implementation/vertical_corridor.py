# vertical_corridor.py
import cv2
import numpy as np


def _get_lander_mask(frame: np.ndarray) -> np.ndarray:
    """Simple brightness-based mask for the lander body."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return mask


def _get_lander_centroid(frame: np.ndarray) -> tuple[float, float] | None:
    mask = _get_lander_mask(frame)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


class VerticalCorridorVerifier:
    """
    While high in the air, is the lander roughly centered over the pad?
    """

    def __init__(self, corridor_frac: float = 0.2, min_height_frac: float = 0.3):
        """
        corridor_frac: fraction of screen width allowed from center.
        min_height_frac: only check when lander is above this fraction of height.
                         (0 = very top, 1 = bottom of screen)
        """
        self.corridor_frac = corridor_frac
        self.min_height_frac = min_height_frac

    def __call__(self, frame: np.ndarray) -> bool:
        h, w, _ = frame.shape
        center_x = w / 2.0

        centroid = _get_lander_centroid(frame)
        if centroid is None:
            # Can't see lander → don't fail based on this.
            return True

        cx, cy = centroid

        # Only care when "high" (near top of the image).
        if cy > h * (1.0 - self.min_height_frac):
            return True

        max_dx = self.corridor_frac * w
        return abs(cx - center_x) <= max_dx
