import cv2
import numpy as np

from pitch_trimming import _estimate_angle
from controlled_descent import _get_lander_centroid, _main_engine_on

class FinalSuccessVerifier:
    """
    From a single frame: lander is between flags, upright-ish, near ground,
    and not firing engines.

    This version is deliberately strict so it only fires when things *really*
    look like a final, stable landing.
    """

    def __init__(
        self,
        center_tol_frac: float = 0.12,   # narrower "between flags" band
        ground_min_frac: float = 0.86,   # must be very close to ground
        ground_max_frac: float = 0.96,   # but not buried in the terrain
        max_angle_rad: float = 0.10,     # ~6 degrees
    ):
        self.center_tol_frac = center_tol_frac
        self.ground_min_frac = ground_min_frac
        self.ground_max_frac = ground_max_frac
        self.max_angle_rad = max_angle_rad

    def __call__(self, frame: np.ndarray) -> bool:
        h, w, _ = frame.shape
        cx_expected = w / 2.0

        centroid = _get_lander_centroid(frame)
        if centroid is None:
            # If we can't even find the lander, call it NOT success.
            return False
        cx, cy = centroid

        # roughly between flags: near center horizontally
        center_ok = abs(cx - cx_expected) <= self.center_tol_frac * w

        # near ground, in a narrow vertical band
        ground_ok = (
            cy >= self.ground_min_frac * h and
            cy <= self.ground_max_frac * h
        )

        # we *require* a valid angle estimate and it must be small
        angle = _estimate_angle(frame)
        if angle is None:
            angle_ok = False
        else:
            angle_ok = abs(angle) <= self.max_angle_rad

        # must have no main engine flame for "final" success
        engines_off = not _main_engine_on(frame)

        return center_ok and ground_ok and angle_ok and engines_off
