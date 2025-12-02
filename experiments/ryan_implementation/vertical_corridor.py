# vertical_corridor.py
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


def _get_lander_pixels(frame: np.ndarray) -> np.ndarray | None:
    """
    Return x-coordinates of all purple lander pixels.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LANDER_LOWER, LANDER_UPPER)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return xs


def _get_lander_centroid_y(frame: np.ndarray) -> float | None:
    """
    Return the y-coordinate of the lander centroid.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LANDER_LOWER, LANDER_UPPER)
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return float(np.mean(ys))


def _get_flag_zone(frame: np.ndarray) -> tuple[int, int] | None:
    """
    Detect the two white flag posts and return (left_x, right_x) defining the corridor.
    
    Strategy:
    1. Find yellow triangles to locate approximate flag positions
    2. For each flag, find the white post column attached to the triangle base
    3. Return the x-coordinates of the two white posts
    """
    h, w, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # First find yellow triangles to get approximate flag locations
    yellow_mask = cv2.inRange(hsv, FLAG_LOWER, FLAG_UPPER)
    ys, xs = np.where(yellow_mask > 0)
    if len(xs) < 10:
        return None
    
    # Split into left and right flags
    median_x = np.median(xs)
    left_yellow_xs = xs[xs < median_x]
    right_yellow_xs = xs[xs >= median_x]
    
    if len(left_yellow_xs) == 0 or len(right_yellow_xs) == 0:
        return None
    
    # The base of each triangle (where the post attaches) is on the LEFT side
    # So for the left flag: leftmost x of yellow pixels
    # For the right flag: leftmost x of yellow pixels
    left_flag_base_x = int(left_yellow_xs.min())
    right_flag_base_x = int(right_yellow_xs.min())
    
    # Now find the white posts near these x-coordinates
    white_mask = cv2.inRange(hsv, POST_LOWER, POST_UPPER)
    
    # Search for white column near left flag base
    search_width = 15
    left_search_region = white_mask[:, max(0, left_flag_base_x - search_width):left_flag_base_x + 5]
    left_ys, left_xs = np.where(left_search_region > 0)
    
    right_search_region = white_mask[:, max(0, right_flag_base_x - search_width):right_flag_base_x + 5]
    right_ys, right_xs = np.where(right_search_region > 0)
    
    # Get the x-coordinate of the white posts (adjust for search region offset)
    if len(left_xs) > 0:
        left_post_x = int(np.median(left_xs)) + max(0, left_flag_base_x - search_width)
    else:
        # Fallback to yellow triangle base
        left_post_x = left_flag_base_x
    
    if len(right_xs) > 0:
        right_post_x = int(np.median(right_xs)) + max(0, right_flag_base_x - search_width)
    else:
        # Fallback to yellow triangle base
        right_post_x = right_flag_base_x
    
    return left_post_x, right_post_x


class VerticalCorridorVerifier:
    """
    Check if at least `required_fraction` of lander pixels are within the flag zone.
    """

    def __init__(
        self,
        min_height_frac: float = 0.35,
        padding: int = 5,
        required_fraction: float = 0.95,
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