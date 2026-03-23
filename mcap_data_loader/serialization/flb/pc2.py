"""Encode and decode example for the airbot_fbs.PointCloud2 schema.

This example uses a plain dictionary as input and output, so it does not depend
on ROS packages.

Expected input format:
    {
        "header": {
            "frame_id": "lidar",
            "stamp": {"sec": 123, "nsec": 456789000},
        },
        "height": 1,
        "width": 2,
        "fields": [
            {"name": "x", "offset": 0, "datatype": "FLOAT32", "count": 1},
            {"name": "y", "offset": 4, "datatype": "FLOAT32", "count": 1},
        ],
        "is_bigendian": False,
        "point_step": 8,
        "row_step": 16,
        "data": b"...",
        "is_dense": True,
    }

Note:
    The FlatBuffers schema stores only ``name``, ``offset``, and ``type`` for
    each field, so this example supports only ``count == 1``.
"""

from __future__ import annotations
import flatbuffers
import numpy as np
from typing import Any
from foxglove_schemas_flatbuffer import NumericType, PackedElementField, Time
from mcap_data_loader.schemas.airbot_fbs import PointCloud2 as AirbotPointCloud2


NAME_TO_FB_DTYPE = {
    "INT8": NumericType.NumericType.INT8,
    "UINT8": NumericType.NumericType.UINT8,
    "INT16": NumericType.NumericType.INT16,
    "UINT16": NumericType.NumericType.UINT16,
    "INT32": NumericType.NumericType.INT32,
    "UINT32": NumericType.NumericType.UINT32,
    "FLOAT32": NumericType.NumericType.FLOAT32,
    "FLOAT64": NumericType.NumericType.FLOAT64,
}

FB_DTYPE_TO_NAME = {value: key for key, value in NAME_TO_FB_DTYPE.items()}


def encode_pointcloud2_dict(
    builder: flatbuffers.Builder, msg: dict[str, Any]
) -> bytes:
    """Encode a PointCloud2-like dictionary into FlatBuffers bytes."""
    header = msg.get("header", {})
    stamp = header.get("stamp", {})
    fields = msg.get("fields", [])
    raw_data = bytes(msg.get("data", b""))

    field_offsets = []
    for field in reversed(fields):
        count = int(field.get("count", 1))
        if count != 1:
            raise ValueError(
                f"Field '{field['name']}' has count={count}, but only count=1 "
                "is supported by airbot_fbs.PointCloud2"
            )

        datatype_name = str(field["datatype"]).upper()
        if datatype_name not in NAME_TO_FB_DTYPE:
            raise ValueError(
                f"Unsupported datatype for field '{field['name']}': {field['datatype']}"
            )

        name_offset = builder.CreateString(str(field["name"]))
        PackedElementField.Start(builder)
        PackedElementField.AddName(builder, name_offset)
        PackedElementField.AddOffset(builder, int(field["offset"]))
        PackedElementField.AddType(builder, NAME_TO_FB_DTYPE[datatype_name])
        field_offsets.append(PackedElementField.End(builder))

    AirbotPointCloud2.PointCloud2StartFieldsVector(builder, len(field_offsets))
    for field_offset in field_offsets:
        builder.PrependUOffsetTRelative(field_offset)
    fields_offset = builder.EndVector()

    data_offset = builder.CreateByteVector(raw_data)
    frame_id_offset = builder.CreateString(str(header.get("frame_id", "")))
    sec = int(stamp.get("sec", 0))
    nsec = int(stamp.get("nsec", 0))

    AirbotPointCloud2.Start(builder)
    AirbotPointCloud2.AddTimestamp(builder, Time.CreateTime(builder, sec, nsec))
    AirbotPointCloud2.AddFrameId(builder, frame_id_offset)
    AirbotPointCloud2.AddPointStride(builder, int(msg.get("point_step", 0)))
    AirbotPointCloud2.AddFields(builder, fields_offset)
    AirbotPointCloud2.AddData(builder, data_offset)
    AirbotPointCloud2.AddRowStep(builder, int(msg.get("row_step", 0)))
    AirbotPointCloud2.AddIsBigendian(builder, bool(msg.get("is_bigendian", False)))
    AirbotPointCloud2.AddIsDense(builder, bool(msg.get("is_dense", False)))
    AirbotPointCloud2.AddHeight(builder, int(msg.get("height", 0)))
    AirbotPointCloud2.AddWidth(builder, int(msg.get("width", 0)))
    root = AirbotPointCloud2.End(builder)
    builder.Finish(root)

    encoded = bytes(builder.Output())
    builder.Clear()
    return encoded


def decode_pointcloud2_dict(buffer: bytes) -> dict[str, Any]:
    """Decode FlatBuffers bytes into a PointCloud2-like dictionary."""
    message = AirbotPointCloud2.PointCloud2.GetRootAs(buffer, 0)
    frame_id = message.FrameId()

    fields = []
    for idx in range(message.FieldsLength()):
        field = message.Fields(idx)
        name = field.Name()
        fields.append(
            {
                "name": name.decode("utf-8")
                if isinstance(name, (bytes, bytearray))
                else str(name),
                "offset": int(field.Offset()),
                "datatype": FB_DTYPE_TO_NAME[int(field.Type())],
                "count": 1,
            }
        )

    raw = message.DataAsNumpy()
    data = b"" if raw is None else np.asarray(raw, dtype=np.uint8).tobytes()
    stamp = message.Timestamp()

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
        "height": int(message.Height()),
        "width": int(message.Width()),
        "fields": fields,
        "is_bigendian": bool(message.IsBigendian()),
        "point_step": int(message.PointStride()),
        "row_step": int(message.RowStep()),
        "data": data,
        "is_dense": bool(message.IsDense()),
    }


def _build_example_pointcloud2_dict() -> dict[str, Any]:
    points = np.array(
        [
            [1.0, 2.0, 3.0, 10.0],
            [4.0, 5.0, 6.0, 20.0],
        ],
        dtype=np.float32,
    )
    point_step = points.shape[1] * points.dtype.itemsize

    return {
        "header": {
            "frame_id": "lidar",
            "stamp": {"sec": 123, "nsec": 456789000},
        },
        "height": 1,
        "width": 2,
        "fields": [
            {"name": "x", "offset": 0, "datatype": "FLOAT32", "count": 1},
            {"name": "y", "offset": 4, "datatype": "FLOAT32", "count": 1},
            {"name": "z", "offset": 8, "datatype": "FLOAT32", "count": 1},
            {"name": "intensity", "offset": 12, "datatype": "FLOAT32", "count": 1},
        ],
        "is_bigendian": False,
        "point_step": point_step,
        "row_step": 2 * point_step,
        "data": points.tobytes(),
        "is_dense": True,
    }


def main() -> None:
    builder = flatbuffers.Builder()
    pointcloud = _build_example_pointcloud2_dict()
    encoded = encode_pointcloud2_dict(builder, pointcloud)
    decoded = decode_pointcloud2_dict(encoded)

    print(f"Encoded bytes: {len(encoded)}")
    print("Header:", decoded["header"])
    print("Dimensions:", decoded["height"], decoded["width"])
    print("point_step / row_step:", decoded["point_step"], decoded["row_step"])
    print("Fields:", decoded["fields"])
    print("Raw data equal:", pointcloud["data"] == decoded["data"])
    print("Decoded point array:")
    print(
        np.frombuffer(decoded["data"], dtype=np.float32).reshape(
            decoded["width"], -1
        )
    )


if __name__ == "__main__":
    main()
