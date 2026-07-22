from mcap.reader import make_reader
from mcap.writer import Writer
from mcap.well_known import MessageEncoding
from turbojpeg import TurboJPEG
from typing import Dict, IO, Optional, Any, Tuple
from collections.abc import Iterable
from foxglove_schemas_flatbuffer import (
    CompressedImage,
    RawImage,
    Time,
    PointCloud,
    get_schema,
)
from importlib.resources import read_binary
from enum import Enum
from mcap_data_loader.schemas.airbot_fbs import (
    FloatArray,
    MultiChannelImage,
    PointCloud2,
)
from mcap_data_loader.serialization.basis import McapReaderBasis, McapWriterBasis
from mcap_data_loader.serialization.flb.mci import (
    encode_multi_channel_image,
    decode_multi_channel_image,
)
from mcap_data_loader.serialization.flb.pc2 import (
    encode_pointcloud2_dict,
    decode_pointcloud2_dict,
)
from mcap_data_loader.serialization.flb.pose import (
    encode_pose_in_frame_dict,
    decode_pose_in_frame_dict,
)
from pathlib import Path
import numpy as np
import flatbuffers


def get_airbot_fbs_tuple(fbs_cls: type) -> Tuple[str, bytes]:
    name = fbs_cls.__name__.split(".")[-1]
    return (
        f"airbot_fbs.{name}",
        read_binary("mcap_data_loader.schemas.airbot_fbs.bfbs", f"{name}.bfbs"),
    )


class FlatBuffersSchemas(Enum):
    """Enum for FlatBuffers schemas used in MCAP files."""

    NONE = ()
    RAW_IMAGE = ("foxglove.RawImage", get_schema("RawImage"))
    COMPRESSED_IMAGE = ("foxglove.CompressedImage", get_schema("CompressedImage"))
    POINT_CLOUD = ("foxglove.PointCloud", get_schema("PointCloud"))
    POSE_IN_FRAME = ("foxglove.PoseInFrame", get_schema("PoseInFrame"))
    # JOINT_STATES = ("foxglove.JointStates", get_schema("JointStates"))
    # JOINT_STATE = ("foxglove.JointState", get_schema("JointState"))
    FLOAT_ARRAY = get_airbot_fbs_tuple(FloatArray)
    MULTI_CHANNEL_IMAGE = get_airbot_fbs_tuple(MultiChannelImage)
    POINT_CLOUD2 = get_airbot_fbs_tuple(PointCloud2)

    def __bool__(self):
        return self is not FlatBuffersSchemas.NONE


class McapFlatBuffersWriter(McapWriterBasis):
    """Class to handle writing MCAP files with FlatBuffers schemas."""

    message_encoding = MessageEncoding.Flatbuffer

    def __init__(self, initial_builder_size: int = 1024 * 1024):
        super().__init__()
        self.builder = flatbuffers.Builder(initial_builder_size)

    def _get_schema_name_and_data(self, schema_type: FlatBuffersSchemas):
        return schema_type.value

    def _get_all_schema_types(self):
        return set(FlatBuffersSchemas) - {FlatBuffersSchemas.NONE}

    def on_add_message(
        self,
        schema_type: FlatBuffersSchemas,
        topic: str,
        data: Any,
        publish_time: int,
        log_time: int,
        **kwargs,
    ):
        return getattr(self, f"add_{schema_type.name.lower()}")(
            topic,
            data,
            publish_time,
            log_time,
            **kwargs,
        )

    def add_compressed_image(
        self,
        topic: str,
        data: bytes,
        publish_time: int,
        log_time: int,
        format: str = "jpeg",
        frame_id: str = "",
    ):
        """Add a compressed image message to the MCAP writer."""

        builder = self.builder
        fmt_str = builder.CreateString(format)
        frame_id_str = builder.CreateString(frame_id)
        data_vec = builder.CreateByteVector(data)
        sec, nsec = divmod(publish_time, 1_000_000_000)
        CompressedImage.Start(builder)
        CompressedImage.AddFormat(builder, fmt_str)
        CompressedImage.AddFrameId(builder, frame_id_str)
        CompressedImage.AddData(builder, data_vec)
        CompressedImage.AddTimestamp(builder, Time.CreateTime(builder, sec, nsec))
        end_data = CompressedImage.End(builder)
        builder.Finish(end_data)
        msg_data = builder.Output()
        self._writer.add_message(
            channel_id=self._cmapping[topic],
            data=bytes(msg_data),
            publish_time=publish_time,
            log_time=log_time,
        )
        builder.Clear()

    def add_raw_image(
        self,
        topic: str,
        data: np.ndarray,
        publish_time: int,
        log_time: int,
        encoding: str = "",
        frame_id: str = "",
    ):
        """Add a raw image message to the MCAP writer."""
        height, width = data.shape[:2]
        step = data.strides[0]
        builder = self.builder
        frame_id_offset = builder.CreateString(frame_id)
        encoding_offset = builder.CreateString(
            encoding or self._get_image_encoding(data)
        )
        data_bytes = data.tobytes()
        data_vec = builder.CreateByteVector(data_bytes)
        RawImage.Start(builder)
        RawImage.AddFrameId(builder, frame_id_offset)
        RawImage.AddWidth(builder, width)
        RawImage.AddHeight(builder, height)
        RawImage.AddEncoding(builder, encoding_offset)
        RawImage.AddStep(builder, step)
        RawImage.AddData(builder, data_vec)
        rawimage = RawImage.End(builder)
        builder.Finish(rawimage)
        msg_data = builder.Output()
        self._writer.add_message(
            channel_id=self._cmapping[topic],
            data=bytes(msg_data),
            publish_time=publish_time,
            log_time=log_time,
        )
        builder.Clear()

    def add_point_cloud(
        self,
        topic: str,
        data: np.ndarray,
        publish_time: int,
        log_time: int,
        frame_id: str = "",
    ):
        """Add a point cloud message to the MCAP writer."""
        vec_data = self.builder.CreateNumpyVector(data)
        PointCloud.Start(self.builder)
        PointCloud.AddData(self.builder, vec_data)
        end_data = PointCloud.End(self.builder)
        self.builder.Finish(end_data)
        msg_data = self.builder.Output()
        self._writer.add_message(
            channel_id=self._cmapping[topic],
            data=bytes(msg_data),
            publish_time=publish_time,
            log_time=log_time,
        )
        self.builder.Clear()

    def add_point_cloud2(
        self,
        topic: str,
        data: dict,
        publish_time: int,
        log_time: int,
    ):
        """Add a PointCloud2 message to the MCAP writer."""
        encoded = encode_pointcloud2_dict(self.builder, data)
        self._writer.add_message(
            channel_id=self._cmapping[topic],
            data=encoded,
            publish_time=publish_time,
            log_time=log_time,
        )
        self.builder.Clear()

    def add_pose_in_frame(
        self,
        topic: str,
        data: dict,
        publish_time: int,
        log_time: int,
    ):
        """Add a PoseInFrame message to the MCAP writer."""
        encoded = encode_pose_in_frame_dict(self.builder, data)
        self._writer.add_message(
            channel_id=self._cmapping[topic],
            data=encoded,
            publish_time=publish_time,
            log_time=log_time,
        )
        self.builder.Clear()

    def add_field_array(
        self,
        topics: Dict[str, str],
        data: dict[str, Iterable[float]],
        publish_time: int,
        log_time: int,
        fields: Optional[Iterable[str]] = None,
    ):
        """Add a joint state message to the MCAP writer in separate field channel as FloatArray schema."""
        fields = fields or topics.keys()
        for field in fields:
            raw_data = data[field]
            self.add_float_array(
                topics[field],
                raw_data,
                publish_time,
                log_time,
            )

    def add_float_array(
        self, topic: str, data: Iterable[float], publish_time: int, log_time: int
    ) -> np.ndarray:
        # TODO: is float64 needed?
        arr = np.asarray(data, dtype=np.float32)
        self._stat[topic]["sum"] += arr
        self._stat[topic]["sum_sq"] += arr**2
        self._stat[topic]["min"] = np.minimum(self._stat[topic]["min"], arr)
        self._stat[topic]["max"] = np.maximum(self._stat[topic]["max"], arr)
        vec_data = self.builder.CreateNumpyVector(arr)
        FloatArray.Start(self.builder)
        FloatArray.AddValues(self.builder, vec_data)
        end_data = FloatArray.End(self.builder)
        self.builder.Finish(end_data)
        msg_data = self.builder.Output()
        self._writer.add_message(
            self._cmapping[topic], log_time, bytes(msg_data), publish_time
        )
        self.builder.Clear()
        return arr  # Return the array for potential further processing

    def add_multi_channel_image(
        self,
        topic: str,
        data: np.ndarray,
        publish_time: int,
        log_time: int,
    ):
        """Add a multi-channel image message to the MCAP writer."""
        self._writer.add_message(
            self._cmapping[topic],
            log_time,
            encode_multi_channel_image(self.builder, data),
            publish_time,
        )



class McapFlatBuffersReader(McapReaderBasis):
    """Class to handle reading MCAP files with FlatBuffers schemas."""

    def _post_init(self):
        self._decoders = {
            FlatBuffersSchemas.FLOAT_ARRAY.value[0]: self._decode_array,
            FlatBuffersSchemas.RAW_IMAGE.value[0]: self._decode_raw_image,
            FlatBuffersSchemas.COMPRESSED_IMAGE.value[0]: self._decode_compressed_image,
            FlatBuffersSchemas.MULTI_CHANNEL_IMAGE.value[0]: decode_multi_channel_image,
            FlatBuffersSchemas.POINT_CLOUD.value[0]: self._decode_point_cloud,
            FlatBuffersSchemas.POINT_CLOUD2.value[0]: decode_pointcloud2_dict,
            FlatBuffersSchemas.POSE_IN_FRAME.value[0]: self._decode_pose_in_frame,
        }
        self._stat_schemas = (FlatBuffersSchemas.FLOAT_ARRAY.value[0],)

    @staticmethod
    def _decode_array(data: bytes) -> np.ndarray:
        """Decode a FloatArray FlatBuffers message."""
        fb = FloatArray.FloatArray.GetRootAs(data, 0)
        return fb.ValuesAsNumpy()

    @staticmethod
    def _decode_raw_image(data: bytes) -> np.ndarray:
        """Decode a RawImage FlatBuffers message."""
        raw_img = RawImage.RawImage.GetRootAs(data, 0)
        width = raw_img.Width()
        height = raw_img.Height()
        step = raw_img.Step()
        encoding = raw_img.Encoding().decode("utf-8")
        np_data = raw_img.DataAsNumpy()

        if encoding in {"rgb8", "bgr8", "8UC3"}:
            channels = 3
            dtype = np.uint8
        elif encoding in {"rgba8", "bgra8"}:
            channels = 4
            dtype = np.uint8
        elif encoding in {"mono8", "8UC1"}:
            channels = 1
            dtype = np.uint8
        elif encoding in {"mono16", "16UC1"}:
            channels = 1
            dtype = np.uint16
        elif encoding == "32FC1":
            channels = 1
            dtype = np.float32
        else:
            raise NotImplementedError(f"Unsupported encoding: {encoding}")

        arr = np_data.view(dtype)
        cal_width = step // (channels * arr.itemsize)
        # TODO: should be warning?
        assert cal_width == width, (
            f"Calculated width {cal_width} does not match expected width {width}"
        )
        if channels == 1:
            img = arr.reshape((height, cal_width))[:, :width]
        else:
            img = arr.reshape((height, cal_width, channels))[:, :width, :]
        return img

    def _decode_compressed_image(self, data: bytes) -> np.ndarray:
        """Decode a CompressedImage FlatBuffers message."""
        compressed_img = CompressedImage.CompressedImage.GetRootAs(data, 0)
        img_format = compressed_img.Format().decode("utf-8")
        assert img_format == "jpeg", f"Expected JPEG format, but got {img_format}"
        return self.jpeg.decode(compressed_img.DataAsNumpy())

    def _decode_point_cloud(self, data: bytes) -> np.ndarray:
        """Decode a PointCloud FlatBuffers message."""
        point_cloud = PointCloud.PointCloud.GetRootAs(data, 0)
        return point_cloud.DataAsNumpy()

    def _decode_pose_in_frame(self, data: bytes) -> dict[str, Any]:
        """Decode a PoseInFrame FlatBuffers message."""
        pose_in_frame = decode_pose_in_frame_dict(data)
        return pose_in_frame

    def _decode(self, schema, message):
        return self._decoders[schema.name](message.data)


def h264_attachment_to_compressed_images(
    file: IO[bytes], output_path: str, quality: int = 85, finish: bool = True
) -> Writer:
    """
    Convert H.264 attachments in an MCAP file to compressed images.

    Args:
        file (IO[bytes]): Path to the MCAP file or a file-like object.
        output_path (str): Path to save the output MCAP file with compressed images.
        quality (int): JPEG compression quality (default: 85).
        finish (bool): Whether to finalize the writer after processing (default: True).

    Returns:
        Writer: An instance of Writer for the output MCAP file.
    """
    from mcap_data_loader.serialization.video.pyav import AvCoder

    jpeg = TurboJPEG()
    av_coder = AvCoder()
    reader = make_reader(file)
    mfb_writer = McapFlatBuffersWriter()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = Writer(output_path)
    writer.start()

    for metadata in reader.iter_metadata():
        writer.add_metadata(metadata.name, metadata.metadata)

    summary = reader.get_summary()
    for schema in summary.schemas.values():
        writer.register_schema(
            schema.name,
            schema.encoding,
            schema.data,
        )
    for channel in summary.channels.values():
        writer.register_channel(
            channel.topic,
            channel.message_encoding,
            channel.schema_id,
            channel.metadata,
        )
    smapping = mfb_writer.register_schemas(
        writer, {FlatBuffersSchemas.COMPRESSED_IMAGE}
    )
    for schema, channel, message in reader.iter_messages():
        writer.add_message(
            message.channel_id,
            message.log_time,
            message.data,
            message.publish_time,
            message.sequence,
        )
    for attachment in reader.iter_attachments():
        if attachment.media_type == "video/mp4":
            c_id = writer.register_channel(
                topic=attachment.name,
                message_encoding=MessageEncoding.Flatbuffer,
                schema_id=smapping[FlatBuffersSchemas.COMPRESSED_IMAGE],
            )
            for frame, pts in av_coder.iter_decode(
                attachment.data, mismatch_tolerance=0, ensure_base_stamp=True
            ):
                mfb_writer.add_compressed_image(
                    writer,
                    c_id,
                    jpeg.encode(frame, quality=quality),
                    pts,
                    # TODO: use the actual log time
                    pts,
                )
        else:
            writer.add_attachment(
                attachment.create_time,
                attachment.log_time,
                attachment.name,
                attachment.media_type,
                attachment.data,
            )
    if finish:
        writer.finish()
    return writer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert H.264 attachments in an MCAP file to compressed images."
    )
    parser.add_argument("input_file", type=str, help="Path to the input MCAP file.")
    parser.add_argument(
        "output_file", type=str, help="Path to save the output MCAP file."
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG compression quality (default: 85).",
    )
    args = parser.parse_args()

    with open(args.input_file, "rb") as input_file:
        h264_attachment_to_compressed_images(input_file, args.output_file, args.quality)
        print(
            f"Converted {args.input_file} to {args.output_file} with quality {args.quality}."
        )
