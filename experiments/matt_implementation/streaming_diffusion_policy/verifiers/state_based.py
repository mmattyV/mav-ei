# state_based.py
"""
State-based verifiers for Lunar Lander.

These verifiers analyze the observation vector directly, which is more accurate
than image-based detection for dynamic properties like velocity.

Lunar Lander observation space:
    obs[0]: x position (-1 to 1, 0 is center of landing zone)
    obs[1]: y position (0 is ground level, ~1.4 is top)
    obs[2]: x velocity (horizontal)
    obs[3]: y velocity (vertical, negative is falling)
    obs[4]: angle (radians, 0 is upright)
    obs[5]: angular velocity
    obs[6]: left leg contact (0 or 1)
    obs[7]: right leg contact (0 or 1)

All verifiers follow the same interface as image-based verifiers:
    verifier(obs) -> bool
"""

import numpy as np


class SafeVelocityVerifier:
    """
    Check if the lander's velocity is safe for landing.
    
    - Horizontal velocity should be low (not drifting sideways)
    - Vertical velocity should be negative (descending) but not too fast
    """
    
    def __init__(
        self,
        max_horizontal_speed: float = 0.4,
        max_vertical_speed: float = 0.8,
        min_vertical_speed: float = -0.1,  # slight upward OK
    ):
        self.max_horizontal_speed = max_horizontal_speed
        self.max_vertical_speed = max_vertical_speed
        self.min_vertical_speed = min_vertical_speed
    
    def __call__(self, obs: np.ndarray) -> bool:
        vx, vy = obs[2], obs[3]
        
        horizontal_ok = abs(vx) < self.max_horizontal_speed
        vertical_ok = self.min_vertical_speed < vy < self.max_vertical_speed
        
        return horizontal_ok and vertical_ok


class StableRotationVerifier:
    """
    Check if the lander is stable (not tilted too much, not spinning).
    """
    
    def __init__(
        self,
        max_angle: float = 0.3,  # ~17 degrees
        max_angular_velocity: float = 0.4,
    ):
        self.max_angle = max_angle
        self.max_angular_velocity = max_angular_velocity
    
    def __call__(self, obs: np.ndarray) -> bool:
        angle, angular_vel = obs[4], obs[5]
        
        angle_ok = abs(angle) < self.max_angle
        rotation_ok = abs(angular_vel) < self.max_angular_velocity
        
        return angle_ok and rotation_ok


class OnTargetVerifier:
    """
    Check if the lander will land near the target zone based on current trajectory.
    
    Uses simple linear prediction: predicted_x = x + vx * time_to_ground
    """
    
    def __init__(
        self,
        target_zone_width: float = 0.4,  # acceptable landing zone
        prediction_time: float = 2.0,     # seconds to look ahead
    ):
        self.target_zone_width = target_zone_width
        self.prediction_time = prediction_time
    
    def __call__(self, obs: np.ndarray) -> bool:
        x, vx = obs[0], obs[2]
        
        # Predict where lander will be
        predicted_x = x + vx * self.prediction_time
        
        # Check if within target zone (centered at 0)
        return abs(predicted_x) < self.target_zone_width


class SafeLandingConditionVerifier:
    """
    Check if conditions are safe for final landing approach.
    
    When close to ground, requires:
    - Slow descent
    - Nearly upright
    - Low horizontal drift
    """
    
    def __init__(
        self,
        activation_altitude: float = 0.3,  # only active when low
        max_descent_rate: float = 0.5,
        max_angle: float = 0.15,           # ~8.5 degrees
        max_horizontal_speed: float = 0.2,
    ):
        self.activation_altitude = activation_altitude
        self.max_descent_rate = max_descent_rate
        self.max_angle = max_angle
        self.max_horizontal_speed = max_horizontal_speed
    
    def __call__(self, obs: np.ndarray) -> bool:
        y = obs[1]  # altitude
        
        # Only check when close to ground
        if y > self.activation_altitude:
            return True  # Not relevant yet, pass by default
        
        vx, vy = obs[2], obs[3]
        angle = obs[4]
        
        descent_ok = abs(vy) < self.max_descent_rate
        angle_ok = abs(angle) < self.max_angle
        horizontal_ok = abs(vx) < self.max_horizontal_speed
        
        return descent_ok and angle_ok and horizontal_ok


class DescendingVerifier:
    """
    Simple check that the lander is actually descending (not going up unnecessarily).
    
    Allows brief upward movement but penalizes sustained climbing.
    """
    
    def __init__(
        self,
        max_climb_rate: float = 0.3,  # allow small upward velocity
    ):
        self.max_climb_rate = max_climb_rate
    
    def __call__(self, obs: np.ndarray) -> bool:
        vy = obs[3]
        
        # vy > 0 means going up, which wastes fuel
        return vy < self.max_climb_rate


class InCorridorVerifier:
    """
    State-based check if lander is horizontally within the landing corridor.
    
    More accurate than image-based since we have exact position.
    """
    
    def __init__(
        self,
        corridor_width: float = 0.3,  # half-width from center
    ):
        self.corridor_width = corridor_width
    
    def __call__(self, obs: np.ndarray) -> bool:
        x = obs[0]
        return abs(x) < self.corridor_width

