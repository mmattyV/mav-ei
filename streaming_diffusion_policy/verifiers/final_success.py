import cv2
import numpy as np

from .pitch_trimming import _estimate_angle
from .controlled_descent import (
    _get_lander_centroid,
    _get_lander_points,
    FLAME_LOWER,
    FLAME_UPPER,
)
from .vertical_corridor import _get_lander_pixels, _get_flag_zone


def _any_flame(frame: np.ndarray, min_flame_pixels: int = 5) -> bool:
    """
    Return True if the frame contains at least `min_flame_pixels` flame-colored
    pixels (orange/red), anywhere in the image.

    This detects both main engine and side-thruster flames without relying
    on their specific geometry.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, FLAME_LOWER, FLAME_UPPER)
    flame_pixels = int(np.count_nonzero(mask))
    return flame_pixels >= min_flame_pixels


class FinalSuccessVerifier:
    """
    One-frame check for true final landing:

      - lander is sitting on the ground,
      - upright (small angle),
      - all purple pixels are between the flag posts,
      - absolutely no flame anywhere (no main, no side).
    """

    def __init__(
        self,
        # How close the *bottom* of the lander must be to the ground.
        # (0 = top of image, 1 = bottom).
        ground_min_frac: float = 0.72,

        # Max allowed tilt for "perfectly upright".
        max_angle_rad: float = np.radians(5.0),

        # How strict we are about "all purple inside the flag corridor".
        corridor_required_fraction: float = 0.98,
        corridor_padding_px: int = 5,
    ):
        self.ground_min_frac = ground_min_frac
        self.max_angle_rad = max_angle_rad
        self.corridor_required_fraction = corridor_required_fraction
        self.corridor_padding_px = corridor_padding_px

    def __call__(self, frame: np.ndarray) -> bool:
        h, w, _ = frame.shape

        # --- 1) Basic lander visibility ---
        centroid = _get_lander_centroid(frame)
        if centroid is None:
            return False

        pts = _get_lander_points(frame)
        if pts is None:
            return False

        # --- 2) On the ground: use the *bottom* of the purple blob ---
        bottom_y = float(pts[:, 1].max())
        ground_ok = bottom_y >= self.ground_min_frac * h

        # --- 3) Upright: require small angle and a valid estimate ---
        angle = _estimate_angle(frame)
        if angle is None:
            angle_ok = False
        else:
            angle_ok = abs(angle) <= self.max_angle_rad

        # --- 4) Absolutely no flame anywhere ---
        flames_off = not _any_flame(frame)

        # --- 5) All purple pixels between the flag posts ---
        lander_xs = _get_lander_pixels(frame)
        if lander_xs is None:
            return False

        flag_zone = _get_flag_zone(frame)
        if flag_zone is not None:
            left_bound = flag_zone[0] - self.corridor_padding_px
            right_bound = flag_zone[1] + self.corridor_padding_px
        else:
            # fallback if flags can't be detected
            center_x = w / 2.0
            corridor_width = 0.12 * w
            left_bound = center_x - corridor_width
            right_bound = center_x + corridor_width

        fraction_in_zone = np.mean(
            (lander_xs >= left_bound) & (lander_xs <= right_bound)
        )
        corridor_ok = fraction_in_zone >= self.corridor_required_fraction

        # --- Final decision ---
        return ground_ok and angle_ok and flames_off and corridor_ok
