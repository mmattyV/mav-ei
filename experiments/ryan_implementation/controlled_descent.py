import math
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


def _get_lander_points(frame: np.ndarray) -> np.ndarray | None:
    """
    Return all lander-body pixels as (x, y) points, using the same purple mask.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LANDER_LOWER, LANDER_UPPER)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    pts = np.vstack([xs, ys]).T.astype(np.float32)
    return pts


def _estimate_angle(frame: np.ndarray) -> float | None:
    """
    Estimate lander body orientation using PCA on the purple pixels.

    Returns:
        angle in radians, where:
          - 0   = horizontal (pointing to the right)
          - >0  = rotated counter-clockwise
          - <0  = rotated clockwise
    """
    pts = _get_lander_points(frame)
    if pts is None or pts.shape[0] < 10:
        return None

    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)

    eigvals, eigvecs = np.linalg.eig(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]

    angle = float(math.atan2(principal_axis[1], principal_axis[0]))
    return angle


def _main_engine_on(
    frame: np.ndarray,
    centroid=None,
    roi_half_width: int = 30,
    roi_vertical_extent: int = 140,
    min_flame_pixels: int = 4,
) -> bool:
    """
    Detect main engine flame: a non-trivial cluster of flame-colored pixels
    in a reasonably wide band *below* the lander.

    We:
      - find an HSV mask for flame colors,
      - define a rectangular ROI centered horizontally on the lander,
        starting a few pixels below it and extending downward,
      - count flame-mask pixels in that ROI.

    The engine is considered ON only if at least `min_flame_pixels` pixels
    are present. This filters out 1–2 stray sparks from previous frames.
    """
    h, w, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    flame_mask = cv2.inRange(hsv, FLAME_LOWER, FLAME_UPPER)

    if centroid is None:
        centroid = _get_lander_centroid(frame)
    if centroid is None:
        return False

    cx, cy = centroid

    # Define a wider, deeper band under the lander.
    x0 = int(max(cx - roi_half_width, 0))
    x1 = int(min(cx + roi_half_width, w))

    # Start a little below the lander bottom, extend downward.
    y0 = int(min(cy + 5, h - 1))
    y1 = int(min(cy + roi_vertical_extent, h))

    if y1 <= y0 or x1 <= x0:
        return False

    roi = flame_mask[y0:y1, x0:x1]

    # Count flame pixels instead of using roi.sum() (255 per pixel).
    flame_pixels = int(np.count_nonzero(roi))
    return flame_pixels >= min_flame_pixels


class ControlledDescentVerifier:
    """
    Heuristic check for "controlled descent" with the main engine.

    Semantics:

      - If the lander is very close to the ground, we don't enforce anything
        here (FinalSuccess / other verifiers handle that region).
      - At any appreciable height above the ground buffer, we expect the
        lander to be:
          * not wildly tilted, and
          * braking with its main engine (a clear plume beneath it).

    This flags:
      - upright-but-no-engine frames as False, and
      - extremely tilted frames as False, even if the engine is on.
    """

    def __init__(
        self,
        high_air_frac: float = 0.55,             # kept for backwards compat; not used
        ground_buffer_frac: float = 0.12,        # bottom fraction we ignore
        upright_tol_rad: float = math.radians(25.0),  # max tilt allowed for "controlled"
        min_flame_pixels: int = 4,               # required flame pixels in ROI
    ):
        self.high_air_frac = high_air_frac
        self.ground_buffer_frac = ground_buffer_frac
        self.upright_tol_rad = upright_tol_rad
        self.min_flame_pixels = min_flame_pixels

    def __call__(self, frame: np.ndarray) -> bool:
        h, w, _ = frame.shape
        centroid = _get_lander_centroid(frame)
        if centroid is None:
            # If we can't see the lander at all, don't make a judgment here.
            return True

        _, cy = centroid

        # Very close to the ground → let other verifiers decide.
        if cy > h * (1.0 - self.ground_buffer_frac):
            return True

        # Strongly non-upright at height is never "controlled descent".
        angle = _estimate_angle(frame)
        if angle is not None and abs(angle) > self.upright_tol_rad:
            return False

        # For any altitude above the near-ground buffer, require main engine on.
        return _main_engine_on(frame, centroid, min_flame_pixels=self.min_flame_pixels)
