"""Encode and decode example for the foxglove_schemas_flatbuffer.PoseInFrame schema.

This example uses a plain dictionary as input and output, so it does not depend
on ROS packages.

Expected input format:
    {
        "header": {
            "frame_id": "base_link",
            "stamp": {"sec": 123, "nsec": 456789000},
        },
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
"""

from __future__ import annotations
import flatbuffers
from typing import Any
from foxglove_schemas_flatbuffer import PoseInFrame, Pose, Vector3, Quaternion, Time


def encode_pose_in_frame_dict(
    builder: flatbuffers.Builder, msg: dict[str, Any]
) -> bytes:
    """Encode a PoseInFrame-like dictionary into FlatBuffers bytes."""
    header = msg.get("header", {})
    stamp = header.get("stamp", {})
    pose_dict = msg.get("pose", {})
    position_dict = pose_dict.get("position", {})
    orientation_dict = pose_dict.get("orientation", {})

    frame_id_offset = builder.CreateString(str(header.get("frame_id", "")))

    # Build Vector3 (position)
    Vector3.Start(builder)
    Vector3.AddX(builder, float(position_dict.get("x", 0.0)))
    Vector3.AddY(builder, float(position_dict.get("y", 0.0)))
    Vector3.AddZ(builder, float(position_dict.get("z", 0.0)))
    position_offset = Vector3.End(builder)

    # Build Quaternion (orientation)
    Quaternion.Start(builder)
    Quaternion.AddX(builder, float(orientation_dict.get("x", 0.0)))
    Quaternion.AddY(builder, float(orientation_dict.get("y", 0.0)))
    Quaternion.AddZ(builder, float(orientation_dict.get("z", 0.0)))
    Quaternion.AddW(builder, float(orientation_dict.get("w", 1.0)))
    orientation_offset = Quaternion.End(builder)

    # Build Pose
    Pose.Start(builder)
    Pose.AddPosition(builder, position_offset)
    Pose.AddOrientation(builder, orientation_offset)
    pose_offset = Pose.End(builder)

    sec = int(stamp.get("sec", 0))
    nsec = int(stamp.get("nsec", 0))

    # Build PoseInFrame
    PoseInFrame.Start(builder)
    PoseInFrame.AddTimestamp(builder, Time.CreateTime(builder, sec, nsec))
    PoseInFrame.AddFrameId(builder, frame_id_offset)
    PoseInFrame.AddPose(builder, pose_offset)
    root = PoseInFrame.End(builder)
    builder.Finish(root)

    encoded = bytes(builder.Output())
    builder.Clear()
    return encoded


def decode_pose_in_frame_dict(buffer: bytes) -> dict[str, Any]:
    """Decode FlatBuffers bytes into a PoseInFrame-like dictionary."""
    message = PoseInFrame.PoseInFrame.GetRootAs(buffer, 0)
    stamp = message.Timestamp()
    frame_id = message.FrameId()
    pose = message.Pose()

    position = pose.Position() if pose is not None else None
    orientation = pose.Orientation() if pose is not None else None

    return {
        "header": {
            "frame_id": frame_id.decode("utf-8")
            if isinstance(frame_id, (bytes, bytearray))
            else (frame_id or ""),
            "stamp": {
                "sec": 0 if stamp is None else int(stamp.Sec()),
                "nsec": 0 if stamp is None else int(stamp.Nsec()),
            },
        },
        "pose": {
            "position": {
                "x": 0.0 if position is None else float(position.X()),
                "y": 0.0 if position is None else float(position.Y()),
                "z": 0.0 if position is None else float(position.Z()),
            },
            "orientation": {
                "x": 0.0 if orientation is None else float(orientation.X()),
                "y": 0.0 if orientation is None else float(orientation.Y()),
                "z": 0.0 if orientation is None else float(orientation.Z()),
                "w": 1.0 if orientation is None else float(orientation.W()),
            },
        },
    }


def main() -> None:
    builder = flatbuffers.Builder()
    original = {
        "header": {
            "frame_id": "base_link",
            "stamp": {"sec": 123, "nsec": 456789000},
        },
        "pose": {
            "position": {"x": 1.5, "y": 2.5, "z": 3.5},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.707, "w": 0.707},
        },
    }

    encoded = encode_pose_in_frame_dict(builder, original)
    decoded = decode_pose_in_frame_dict(encoded)

    print(f"Encoded bytes: {len(encoded)}")
    print("Header:", decoded["header"])
    print("Position:", decoded["pose"]["position"])
    print("Orientation:", decoded["pose"]["orientation"])
    print(
        "Position equal:",
        original["pose"]["position"] == decoded["pose"]["position"],
    )
    print(
        "Orientation equal:",
        original["pose"]["orientation"] == decoded["pose"]["orientation"],
    )


if __name__ == "__main__":
    main()
