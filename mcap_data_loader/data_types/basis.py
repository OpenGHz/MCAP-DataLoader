from pydantic import BaseModel
from typing import List


class Time(BaseModel):
    """Represents a timestamp with seconds and nanoseconds."""

    sec: int = 0
    """The seconds part of the timestamp."""
    nanosec: int = 0
    """The nanoseconds part of the timestamp."""


class PosePro(BaseModel):
    """Represents a 3D pose with position and orientation."""

    position: List[float] = []
    """Position as [x, y, z]."""
    orientation: List[float] = []
    """Orientation as [x, y, z, w] quaternion."""
    pose_link: str = ""
    """The link/frame of the pose."""
    reference_link: str = ""
    """The reference link/frame of the pose."""
    stamp: Time = Time()
    """The timestamp of the pose."""


class Pose3D(BaseModel):
    """Represents a 3D pose with x, y, and theta."""

    x: float = 0.0
    """The x coordinate."""
    y: float = 0.0
    """The y coordinate."""
    theta: float = 0.0
    """The orientation angle theta."""


class Twist3D(BaseModel):
    """Represents a 3D twist with x, y, and omega."""

    x: float = 0.0
    """The x component of the twist."""
    y: float = 0.0
    """The y component of the twist, which is useless for two-wheel robot"""
    omega: float = 0.0
    """The angular velocity omega."""


class BaseState(BaseModel):
    """Base state representation."""

    pose: Pose3D = Pose3D()
    """The pose of the base."""
    velocity: Twist3D = Twist3D()
    """The velocity of the base."""
    odometry: float = 0.0
    """The odometry reading."""
    stamp: Time = Time()
    """The timestamp of the base state."""
