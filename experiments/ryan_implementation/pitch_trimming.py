import math

import cv2
import numpy as np

# --- Color ranges (BGR→HSV) for this environment ---

# Lander body (purple)
LANDER_LOWER = np.array([110, 80, 140], dtype=np.uint8)
LANDER_UPPER = np.array([135, 255, 255], dtype=np.uint8)

# Thruster flames (main + side, reddish/orange)
FLAME_LOWER = np.array([0, 150, 150], dtype=np.uint8)
FLAME_UPPER = np.array([15, 255, 255], dtype=np.uint8)


def _get_lander_points(frame: np.ndarray) -> np.ndarray | None:
    """
    Return all pixels that belong to the lander body as (x, y) points.

    We use an HSV mask tuned to the purple lander, so we *don't* pick up the
    bright white terrain (which was the old bug).
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


def _side_thruster_activity(
    frame: np.ndarray,
    min_pixels: int = 8,
    dominance_ratio: float = 1.5,
) -> tuple[bool, bool]:
    """
    Detect whether the *left* or *right* side thruster is firing.

    Strategy:
      - Detect flame-colored pixels in HSV.
      - Use the lander position to define left/right ROIs just below the body,
        with a small "dead zone" in the middle to avoid the main engine plume.
      - Count flame pixels in each ROI.
      - A side thruster is considered ON if:
          count >= min_pixels AND
          count >= dominance_ratio * other_side_count
        This avoids treating symmetric main-engine exhaust as side thrusters.
    """
    pts = _get_lander_points(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    flame_mask = cv2.inRange(hsv, FLAME_LOWER, FLAME_UPPER)
    h, w = flame_mask.shape

    if pts is None:
        # Fallback: rough halves of the whole image.
        left_count = int(np.count_nonzero(flame_mask[:, : w // 2]))
        right_count = int(np.count_nonzero(flame_mask[:, w // 2 :]))
    else:
        cx, cy = pts.mean(axis=0)
        cx = int(cx)
        cy = int(cy)

        # Vertical band where side exhaust is likely to appear.
        y0 = max(cy - 5, 0)
        y1 = min(cy + 120, h)

        # Horizontal ROIs around the lander, with a dead zone around the center
        # to avoid counting the main engine as a side thruster.
        dead = 8  # half-width of the central dead zone

        left0 = max(cx - 60, 0)
        left1 = max(cx - dead, 0)

        right0 = min(cx + dead, w)
        right1 = min(cx + 60, w)

        left_count = int(np.count_nonzero(flame_mask[y0:y1, left0:left1]))
        right_count = int(np.count_nonzero(flame_mask[y0:y1, right0:right1]))

    # Decide "on" based on both absolute pixels and dominance over the other side
    left_on = (
        left_count >= min_pixels and
        left_count >= dominance_ratio * right_count
    )
    right_on = (
        right_count >= min_pixels and
        right_count >= dominance_ratio * left_count
    )

    return left_on, right_on


class PitchTrimmingVerifier:
    """
    Heuristic verifier for "pitch trimming" behavior.

    Intended semantics (matching your examples):

      - If the lander is basically upright → True (no correction needed).
      - If the lander is moderately tilted:
          * we require the *correct* side thruster to be firing
            (opposite side from the lean).
          * no thruster or the wrong thruster → False.
      - If the lander is extremely tilted (≈ 90° or more) → always False.
      - If we can't estimate an angle at all → False.
    """

    def __init__(
        self,
        upright_tol_rad: float = math.radians(3.0),   # ~±3° counted as upright
        extreme_fail_rad: float = math.radians(90.0), # ≥90°: always bad
        flame_min_pixels: int = 8,
        flame_dominance_ratio: float = 1.5,
    ):
        self.upright_tol_rad = upright_tol_rad
        self.extreme_fail_rad = extreme_fail_rad
        self.flame_min_pixels = flame_min_pixels
        self.flame_dominance_ratio = flame_dominance_ratio

    def __call__(self, frame: np.ndarray) -> bool:
        angle = _estimate_angle(frame)
        if angle is None:
            # If we can't see the lander well enough, be conservative.
            return False

        a = abs(angle)

        # 1) Extreme tilt (≈ toppled) → always bad.
        if a >= self.extreme_fail_rad:
            return False

        # 2) Basically upright → fine even if no side thrusters.
        if a <= self.upright_tol_rad:
            return True

        # 3) Moderately tilted → demand a *correct* side thruster.
        left_on, right_on = _side_thruster_activity(
            frame,
            min_pixels=self.flame_min_pixels,
            dominance_ratio=self.flame_dominance_ratio,
        )

        # Note: sign convention here is chosen to match your examples.
        # For the frames you provided:
        #   - angle > 0  (example with bad CW tilt)  → want LEFT thruster.
        #   - angle < 0  (example with bad CCW tilt) → want RIGHT thruster.
        if angle >= 0:
            correcting_on = left_on
        else:
            correcting_on = right_on

        return correcting_on
