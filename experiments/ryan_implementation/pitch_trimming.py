# pitch_trimming.py
import cv2
import numpy as np


def _get_lander_points(frame: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    pts = np.vstack([xs, ys]).T.astype(np.float32)
    return pts


def _estimate_angle(frame: np.ndarray) -> float | None:
    """
    Returns angle in radians. 0 = horizontal right.
    We'll only care about magnitude.
    """
    pts = _get_lander_points(frame)
    if pts is None or pts.shape[0] < 10:
        return None

    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    angle = float(np.arctan2(principal_axis[1], principal_axis[0]))
    return angle


def _side_thruster_on(frame: np.ndarray) -> bool:
    """Look for orange flames to left/right of center."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    h, w = mask.shape
    left = mask[:, : w // 2].sum()
    right = mask[:, w // 2 :].sum()
    return max(left, right) > 200  # heuristic threshold


class PitchTrimmingVerifier:
    """
    When tilted, do we see side thruster flames?
    """

    def __init__(self, angle_tol_rad: float = 0.15):
        self.angle_tol = angle_tol_rad

    def __call__(self, frame: np.ndarray) -> bool:
        angle = _estimate_angle(frame)
        if angle is None:
            # Don't fail on missing detection.
            return True

        if abs(angle) < self.angle_tol:
            return True

        return _side_thruster_on(frame)
