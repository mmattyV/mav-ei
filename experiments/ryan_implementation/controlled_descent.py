# controlled_descent.py
import cv2
import numpy as np

# === Color ranges (BGR→HSV tuned from your screenshots) ===
# Lander body (purple)
LANDER_LOWER = np.array([110, 80, 140], dtype=np.uint8)
LANDER_UPPER = np.array([135, 255, 255], dtype=np.uint8)

# Flames (main + side thrusters) – reddish/orange
FLAME_LOWER = np.array([0, 150, 150], dtype=np.uint8)
FLAME_UPPER = np.array([15, 255, 255], dtype=np.uint8)


def _get_lander_centroid(frame: np.ndarray):
    """
    Detect the lander body via purple HSV range and return (cx, cy).
    frame is expected in BGR (as from cv2.imread).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LANDER_LOWER, LANDER_UPPER)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _main_engine_on(frame: np.ndarray, centroid=None) -> bool:
    """
    Detect main engine flame: flame-colored pixels *directly below* the lander.
    """
    h, w, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    flame_mask = cv2.inRange(hsv, FLAME_LOWER, FLAME_UPPER)

    if centroid is None:
        centroid = _get_lander_centroid(frame)
    if centroid is None:
        return False

    cx, cy = centroid

    # Narrow ROI under the lander body
    x0 = int(max(cx - 15, 0))
    x1 = int(min(cx + 15, w))
    y0 = int(min(cy + 5, h - 1))
    y1 = int(min(cy + 80, h))

    roi = flame_mask[y0:y1, x0:x1]
    return int(roi.sum()) > 200  # heuristic threshold on total mask intensity


class ControlledDescentVerifier:
    """
    From a single frame: if the lander is high above the surface,
    we expect the main engine to be firing (braking) instead of free-fall.
    """

    def __init__(
        self,
        high_air_frac: float = 0.55,   # "high" = above this fraction of screen height
        ground_buffer_frac: float = 0.20,  # don't care about engine close to the ground
    ):
        self.high_air_frac = high_air_frac
        self.ground_buffer_frac = ground_buffer_frac

    def __call__(self, frame: np.ndarray) -> bool:
        h, w, _ = frame.shape
        centroid = _get_lander_centroid(frame)
        if centroid is None:
            # If we can't see the lander at all, don't make a judgment here.
            return True

        _, cy = centroid

        # Close to the ground → we don't enforce engine usage here.
        if cy > h * (1.0 - self.ground_buffer_frac):
            return True

        # Only enforce when "high enough" in the air.
        if cy < h * self.high_air_frac:
            return _main_engine_on(frame, centroid)

        # Mid-altitude: be lenient (treat as OK either way).
        return True
