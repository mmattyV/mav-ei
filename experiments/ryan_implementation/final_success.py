# final_success.py
import cv2
import numpy as np

from pitch_trimming import _estimate_angle
from controlled_descent import _get_lander_centroid, _main_engine_on


class FinalSuccessVerifier:
    """
    From a single frame: lander is between flags, upright-ish, near ground,
    and not firing engines.
    """

    def __init__(
        self,
        center_tol_frac: float = 0.2,
        ground_frac: float = 0.8,
        max_angle_rad: float = 0.15,
    ):
        self.center_tol_frac = center_tol_frac
        self.ground_frac = ground_frac
        self.max_angle_rad = max_angle_rad

    def __call__(self, frame: np.ndarray) -> bool:
        h, w, _ = frame.shape
        cx_expected = w / 2.0

        centroid = _get_lander_centroid(frame)
        if centroid is None:
            return False
        cx, cy = centroid

        # roughly between flags: near center horizontally
        center_ok = abs(cx - cx_expected) <= self.center_tol_frac * w

        # near ground: high y in image coordinates
        ground_ok = cy >= self.ground_frac * h

        angle = _estimate_angle(frame)
        if angle is None:
            angle_ok = True
        else:
            angle_ok = abs(angle) <= self.max_angle_rad

        engines_off = not _main_engine_on(frame)

        return center_ok and ground_ok and angle_ok and engines_off
