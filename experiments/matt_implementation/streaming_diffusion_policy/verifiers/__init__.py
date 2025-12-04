"""
MAV-EI Verifier Zoo for Lunar Lander.

Two types of verifiers:

1. IMAGE-BASED (from Ryan's implementation):
   Take a BGR frame (from cv2) and return True/False.
   - VerticalCorridorVerifier: Checks if lander is within the landing corridor
   - ControlledDescentVerifier: Checks for proper braking with main engine
   - PitchTrimmingVerifier: Checks if lander is correcting its tilt properly
   - FinalSuccessVerifier: Checks if lander has successfully landed

2. STATE-BASED (new):
   Take the observation vector and return True/False.
   More accurate for dynamic properties like velocity.
   - SafeVelocityVerifier: Checks horizontal/vertical velocity is safe
   - StableRotationVerifier: Checks lander isn't tilted or spinning
   - OnTargetVerifier: Predicts if trajectory will hit landing zone
   - SafeLandingConditionVerifier: Final approach safety check
   - DescendingVerifier: Checks lander is descending (not wasting fuel)
   - InCorridorVerifier: State-based corridor check (more accurate)
"""

# Image-based verifiers
from .vertical_corridor import VerticalCorridorVerifier
from .controlled_descent import ControlledDescentVerifier
from .pitch_trimming import PitchTrimmingVerifier
from .final_success import FinalSuccessVerifier

# State-based verifiers
from .state_based import (
    SafeVelocityVerifier,
    StableRotationVerifier,
    OnTargetVerifier,
    SafeLandingConditionVerifier,
    DescendingVerifier,
    InCorridorVerifier,
)

__all__ = [
    # Image-based
    'VerticalCorridorVerifier',
    'ControlledDescentVerifier', 
    'PitchTrimmingVerifier',
    'FinalSuccessVerifier',
    # State-based
    'SafeVelocityVerifier',
    'StableRotationVerifier',
    'OnTargetVerifier',
    'SafeLandingConditionVerifier',
    'DescendingVerifier',
    'InCorridorVerifier',
]
