# vertical_corridor.py
from typing import Optional, Tuple

import cv2
import numpy as np

# Lander body (purple)
LANDER_LOWER = np.array([110, 80, 140], dtype=np.uint8)
LANDER_UPPER = np.array([135, 255, 255], dtype=np.uint8)

# Yellow flags (triangles)
FLAG_LOWER = np.array([20, 100, 100], dtype=np.uint8)
FLAG_UPPER = np.array([35, 255, 255], dtype=np.uint8)

# White flag posts (high brightness, low saturation)
POST_LOWER = np.array([0, 0, 200], dtype=np.uint8)
POST_UPPER = np.array([180, 30, 255], dtype=np.uint8)


def _get_lander_pixels(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Return x-coordinates of all purple lander pixels.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LANDER_LOWER, LANDER_UPPER)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return xs


def _get_lander_centroid_y(frame: np.ndarray) -> Optional[float]:
    """
    Return the y-coordinate of the lander centroid.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LANDER_LOWER, LANDER_UPPER)
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return float(np.mean(ys))


def _get_flag_zone(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Detect the two yellow flag triangles and return (left_x, right_x)
    based on their left edges.

    Strategy:
    1. Find yellow pixels corresponding to both flags.
    2. Split into left-flag and right-flag clusters by x-median.
    3. For each cluster, take the leftmost x as the flag's "post" x.
    """
    h, w, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1) Find yellow pixels (flags)
    yellow_mask = cv2.inRange(hsv, FLAG_LOWER, FLAG_UPPER)
    ys, xs = np.where(yellow_mask > 0)
    if len(xs) < 10:
        # Not enough yellow to be confident there are two flags
        return None

    # 2) Split into left and right clusters by x-median
    median_x = np.median(xs)
    left_yellow_xs = xs[xs < median_x]
    right_yellow_xs = xs[xs >= median_x]

    if len(left_yellow_xs) == 0 or len(right_yellow_xs) == 0:
        return None

    # 3) Use the LEFT edges of each yellow triangle as the corridor boundaries
    left_flag_x = int(left_yellow_xs.min())
    right_flag_x = int(right_yellow_xs.min())

    # Just in case things got swapped for some weird reason
    left_flag_x, right_flag_x = sorted((left_flag_x, right_flag_x))

    return left_flag_x, right_flag_x


class VerticalCorridorVerifier:
    """
    Check if at least `required_fraction` of lander pixels are within the flag zone.
    """

    def __init__(
        self,
        min_height_frac: float = 0.35,
        padding: int = 5,
        required_fraction: float = 0.99,
    ):
        """
        min_height_frac: only check when lander is above this fraction of screen height.
        padding: extra pixels of tolerance on each side of the flag zone.
        required_fraction: fraction of purple pixels that must be in the zone (default 95%).
        """
        self.min_height_frac = min_height_frac
        self.padding = padding
        self.required_fraction = required_fraction

    def __call__(self, frame: np.ndarray) -> bool:
        h, w, _ = frame.shape

        lander_xs = _get_lander_pixels(frame)
        if lander_xs is None:
            # Can't see lander, don't fail
            return True

        # Only check when "high" (near top of image)
        lander_cy = _get_lander_centroid_y(frame)
        if lander_cy is None or lander_cy > h * (1.0 - self.min_height_frac):
            return True

        flag_zone = _get_flag_zone(frame)
        if flag_zone is None:
            # Can't detect flags, fall back to center-based check
            center_x = w / 2.0
            corridor_width = w * 0.15
            left_bound = center_x - corridor_width
            right_bound = center_x + corridor_width
        else:
            left_bound = flag_zone[0] - self.padding
            right_bound = flag_zone[1] + self.padding

        # Count how many purple pixels are inside the zone
        pixels_in_zone = np.sum((lander_xs >= left_bound) & (lander_xs <= right_bound))
        total_pixels = len(lander_xs)

        fraction_in_zone = pixels_in_zone / total_pixels

        return fraction_in_zone >= self.required_fraction
