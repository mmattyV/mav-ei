"""
MAV-EI Image-Based Verifier Zoo for Lunar Lander.

These verifiers analyze screenshots/frames from the Lunar Lander environment
to assess the quality of the lander's behavior.

Verifiers (from Ryan's implementation):
    - VerticalCorridorVerifier: Checks if lander is within the landing corridor
    - ControlledDescentVerifier: Checks for proper braking with main engine
    - PitchTrimmingVerifier: Checks if lander is correcting its tilt properly
    - FinalSuccessVerifier: Checks if lander has successfully landed

All verifiers take a BGR frame (from cv2) and return True/False.
"""

from .vertical_corridor import VerticalCorridorVerifier
from .controlled_descent import ControlledDescentVerifier
from .pitch_trimming import PitchTrimmingVerifier
from .final_success import FinalSuccessVerifier

__all__ = [
    'VerticalCorridorVerifier',
    'ControlledDescentVerifier', 
    'PitchTrimmingVerifier',
    'FinalSuccessVerifier',
]
