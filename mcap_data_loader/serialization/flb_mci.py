"""Encode and decode example for the airbot_fbs.MultiChannelImage schema."""

import flatbuffers
import numpy as np
from mcap_data_loader.schemas.airbot_fbs import DataType, ImageShape, MultiChannelImage


NUMPY_TO_FB_DTYPE = {
    np.dtype(np.uint8): DataType.DataType.UINT8,
    np.dtype(np.int8): DataType.DataType.INT8,
    np.dtype(np.uint16): DataType.DataType.UINT16,
    np.dtype(np.int16): DataType.DataType.INT16,
    np.dtype(np.float32): DataType.DataType.FLOAT32,
    np.dtype(np.float64): DataType.DataType.FLOAT64,
}

FB_TO_NUMPY_DTYPE = {value: key for key, value in NUMPY_TO_FB_DTYPE.items()}


def encode_multi_channel_image(
    builder: flatbuffers.Builder, image: np.ndarray
) -> bytes:
    """Encode an ``H x W x C`` numpy array into FlatBuffers bytes."""
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D array shaped as (H, W, C), got {image.shape}")
    image = np.ascontiguousarray(image)
    fb_dtype = NUMPY_TO_FB_DTYPE[image.dtype]
    data_offset = builder.CreateByteVector(image.tobytes())
    shape_offset = ImageShape.CreateImageShape(
        builder,
        np.array(image.shape, dtype=np.uint32),
    )

    MultiChannelImage.Start(builder)
    MultiChannelImage.AddShape(builder, shape_offset)
    MultiChannelImage.AddDtype(builder, fb_dtype)
    MultiChannelImage.AddData(builder, data_offset)
    root = MultiChannelImage.End(builder)
    builder.Finish(root)
    encoded = bytes(builder.Output())
    builder.Clear()
    return encoded


def decode_multi_channel_image(buffer: bytes) -> np.ndarray:
    """Decode FlatBuffers bytes back into a numpy array."""
    message = MultiChannelImage.MultiChannelImage.GetRootAs(buffer, 0)

    shape = message.Shape()
    if shape is None:
        raise ValueError("Missing shape field in MultiChannelImage message")

    dims = tuple(int(v) for v in shape.DimsAsNumpy())
    np_dtype = FB_TO_NUMPY_DTYPE[message.Dtype()]
    expected_size = int(np.prod(dims))
    raw = message.DataAsNumpy()
    if raw is None:
        raise ValueError("Missing image data in MultiChannelImage message")
    if len(raw) == 0 and expected_size != 0:
        raise ValueError("Missing image data in MultiChannelImage message")

    array = np.frombuffer(raw.tobytes(), dtype=np_dtype)
    if array.size != expected_size:
        raise ValueError(
            f"Data size mismatch: expected {expected_size} elements, got {array.size}"
        )
    return array.reshape(dims)


def main() -> None:
    builder = flatbuffers.Builder()
    image = np.arange(2 * 3 * 5, dtype=np.uint8).reshape(2, 3, 5)
    encoded = encode_multi_channel_image(builder, image)
    decoded = decode_multi_channel_image(encoded)

    print(f"Original shape: {image.shape}, dtype: {image.dtype}")
    print(f"Encoded bytes: {len(encoded)}")
    print(f"Decoded shape: {decoded.shape}, dtype: {decoded.dtype}")
    print("Arrays equal:", np.array_equal(image, decoded))
    print("Decoded array:")
    print(decoded)


if __name__ == "__main__":
    main()
