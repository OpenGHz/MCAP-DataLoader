from mcap.reader import make_reader
from mcap.writer import Writer
from mcap.well_known import SchemaEncoding, MessageEncoding
from turbojpeg import TurboJPEG
from typing import Dict, IO, Set, Optional, Iterable, List, Generator, Any
from foxglove_schemas_flatbuffer import CompressedImage, RawImage, Time, get_schema
from importlib.resources import read_binary
from enum import Enum
from functools import cache
import os
import json
import numpy as np
import flatbuffers
from mcap_data_loader.schemas.airbot_fbs import FloatArray
from mcap_data_loader.utils.basic import zip
from mcap_data_loader.utils.av_coder import AvCoder


class FlatbufferSchemas(Enum):
    """Enum for Flatbuffer schemas used in MCAP files."""

    NONE = ()
    RAW_IMAGE = ("foxglove.RawImage", get_schema("RawImage"))
    COMPRESSED_IMAGE = ("foxglove.CompressedImage", get_schema("CompressedImage"))
    FLOAT_ARRAY = (
        "airbot_fbs.FloatArray",
        read_binary(
            "mcap_data_loader.schemas.airbot_fbs.bfbs",
            "FloatArray.bfbs",
        ),
    )


class McapFlatbufferWriter:
    """Class to handle writing MCAP files with Flatbuffer schemas."""

    def __init__(self, initial_builder_size: int = 1024 * 1024):
        self.builder = flatbuffers.Builder(initial_builder_size)
        self._smapping = {}
        self._cmapping = {}
        self._writer = None
        self._img_enc_mapping = {
            1: {
                np.uint8: "8UC1",
                np.uint16: "16UC1",
                np.float32: "32FC1",
            },
            3: {
                np.uint8: "8UC3",
            },
        }

    def set_writer(self, writer: Writer, start: bool = True):
        """Set the MCAP writer for this instance."""
        self._smapping.clear()
        self._cmapping.clear()
        self.builder.Clear()
        self._writer = writer
        if start:
            writer.start()

    def unset_writer(self):
        """Unset the MCAP writer for this instance."""
        self._writer = None

    def get_writer(self) -> Optional[Writer]:
        return self._writer

    def register_schemas(
        self,
        types: Optional[Set[FlatbufferSchemas]] = None,
    ) -> Dict[FlatbufferSchemas, int]:
        types = set(FlatbufferSchemas) if types is None else types
        types.discard(FlatbufferSchemas.NONE)
        for stype in types:
            self._smapping[stype] = self._writer.register_schema(
                stype.value[0],
                SchemaEncoding.Flatbuffer,
                stype.value[1],
            )
        return self._smapping

    def register_channel(
        self, topic: str, schema_type: FlatbufferSchemas, strict: bool = True
    ) -> int:
        """Register a channel with the given topic and schema type in the MCAP writer.
        The schema will be automatically registered if not already present.
        Args:
            topic (str): The topic name for the channel.
            schema_type (FlatbufferSchemas): The schema type for the channel.
            strict (bool): Whether to enforce strict channel registration.
        Returns:
            int: The channel ID for the registered channel.
        Raises:
            ValueError: If the channel is already registered and strict is True.
        """
        if topic not in self._cmapping:
            c_id = self._writer.register_channel(
                topic,
                MessageEncoding.Flatbuffer,
                self._smapping.get(
                    schema_type, self.register_schemas({schema_type})[schema_type]
                ),
            )
            self._cmapping[topic] = c_id
        elif strict:
            raise ValueError(f"Channel '{topic}' is already registered.")
        return self._cmapping[topic]

    def add_message(
        self,
        data_type: str,
        *args,
        **kwargs,
    ):
        """Add a message to the MCAP data sampler."""
        return getattr(self, f"add_{data_type}")(
            *args,
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

    def add_field_array(
        self,
        topics: Dict[str, str],
        data: dict[str, List[float]],
        publish_time: int,
        log_time: int,
        fields: Optional[List[str]] = None,
    ):
        """Add a joint state message to the MCAP writer in separate field channel as FloatArray schema."""
        fields = fields or list(topics.keys())
        for field in fields:
            raw_data = data[field]
            self.add_array(
                topics[field],
                raw_data,
                publish_time,
                log_time,
            )

    def add_array(
        self, topic: str, data: List[float], publish_time: int, log_time: int
    ):
        FloatArray.StartValuesVector(self.builder, len(data))
        for d in reversed(data):
            self.builder.PrependFloat32(d)
        vec_data = self.builder.EndVector()
        FloatArray.Start(self.builder)
        FloatArray.AddValues(self.builder, vec_data)
        end_data = FloatArray.End(self.builder)
        self.builder.Finish(end_data)
        msg_data = self.builder.Output()
        self._writer.add_message(
            channel_id=self._cmapping[topic],
            data=bytes(msg_data),
            publish_time=publish_time,
            log_time=log_time,
        )
        self.builder.Clear()

    def _get_image_encoding(self, image: np.ndarray) -> str:
        """Get the image encoding string for a given channel and dtype."""
        channels = 1 if len(image.shape) == 2 else 3
        return self._img_enc_mapping[channels][image.dtype.type]


class McapFlatbufferReader:
    """Class to handle reading MCAP files with Flatbuffer schemas."""

    def __init__(self, file: IO[bytes]):
        self.file_io = file
        self.reader = make_reader(file)
        self._decoders = {
            "airbot_fbs.FloatArray": self._decode_array,
            "foxglove.RawImage": self._decode_raw_image,
            "foxglove.CompressedImage": self._decode_compressed_image,
        }
        self._jpeg = TurboJPEG()

    @staticmethod
    def _decode_array(data: bytes) -> np.ndarray:
        """Decode a FloatArray Flatbuffer message."""
        fb = FloatArray.FloatArray.GetRootAs(data, 0)
        return fb.ValuesAsNumpy()

    @staticmethod
    def _decode_raw_image(data: bytes) -> np.ndarray:
        """Decode a RawImage Flatbuffer message."""
        raw_img = RawImage.RawImage.GetRootAs(data, 0)
        width = raw_img.Width()
        height = raw_img.Height()
        step = raw_img.Step()
        encoding = raw_img.Encoding().decode("utf-8")
        data: np.ndarray = raw_img.DataAsNumpy()

        if encoding in ("rgb8", "bgr8", "8UC3"):
            channels = 3
            dtype = np.uint8
        elif encoding in ("rgba8", "bgra8"):
            channels = 4
            dtype = np.uint8
        elif encoding in ("mono8", "8UC1"):
            channels = 1
            dtype = np.uint8
        elif encoding in ("mono16", "16UC1"):
            channels = 1
            dtype = np.uint16
        elif encoding == "32FC1":
            channels = 1
            dtype = np.float32
        else:
            raise NotImplementedError(f"Unsupported encoding: {encoding}")

        arr = data.view(dtype)
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
        """Decode a CompressedImage Flatbuffer message."""
        compressed_img = CompressedImage.CompressedImage.GetRootAs(data, 0)
        img_format = compressed_img.Format().decode("utf-8")
        assert img_format == "jpeg", f"Expected JPEG format, but got {img_format}"
        return self._jpeg.decode(compressed_img.DataAsNumpy())

    def iter_message_samples(
        self, topics: Optional[Iterable[str]] = None, reverse: bool = False
    ) -> Generator[Dict[str, Any], None, None]:
        """Iterate over messages in the MCAP file."""
        # TODO: support iter through a reference topic
        # and inter other topics with start_time according
        # to the reference topic
        if topics is None:
            topics = self.all_topic_names()
        else:
            diff = set(topics) - self.all_topic_names()
            assert not diff, (
                f"Topics {diff} not found. Available: {self.all_topic_names()}"
            )
        messages = {}
        for schema, channel, message in self.reader.iter_messages(
            topics, reverse=reverse
        ):
            data = self._decoders[schema.name](message.data)
            messages[channel.topic] = data
            if len(messages) == len(topics):
                yield messages
                messages.clear()

    @cache
    def all_topic_names(self) -> Set[str]:
        """Get all topics in the MCAP file."""
        return {
            channel.topic for channel in self.reader.get_summary().channels.values()
        }

    @cache
    def all_attachment_names(self) -> Set[str]:
        """Get all attachment names in the MCAP file."""
        return {attachment.name for attachment in self.reader.iter_attachments()}

    def iter_attachment_samples(
        self, names: Optional[Iterable[str]] = None, reverse: bool = False
    ) -> Generator[Dict[str, Any], None, None]:
        """Iterate over target attachments in the MCAP file."""
        assert not reverse, "Reverse iteration is not supported for attachments yet."
        if names is None:
            names = self.all_attachment_names()
        else:
            diff = set(names) - self.all_attachment_names()
            assert not diff, (
                f"Attachments {diff} not found. Available: {self.all_attachment_names()}"
            )

        attch_names: List[str] = []
        iters: List[Generator] = []
        for attachment in self.reader.iter_attachments():
            name = attachment.name
            if name in names:
                media_type = attachment.media_type
                attch_names.append(name)
                if media_type == "video/mp4":
                    coder = AvCoder()
                    attach_iter = coder.iter_decode(
                        attachment.data,
                        mismatch_tolerance=5,
                        ensure_base_stamp=True,
                        target_time_base=0,
                    )
                elif media_type == "application/json":
                    attach_iter = iter(json.loads(attachment.data))
                else:
                    raise ValueError(f"Unsupported media type: {media_type}")
                iters.append(attach_iter)
                if len(attch_names) == len(names):
                    break
        else:
            assert not names, (
                f"Not all requested attachments found: {names} vs {attch_names}"
            )
        try:
            for values in zip(*iters):
                data = {}
                for name, value in zip(attch_names, values):
                    data[name] = value
                yield data
        except ValueError as e:
            raise ValueError(
                f"Attachment iterators have different lengths: {attch_names}"
            ) from e

    def iter_samples(
        self,
        keys: Optional[Iterable[str]] = None,
        topics: Optional[Iterable[str]] = None,
        attachments: Optional[Iterable[str]] = None,
        reverse: bool = False,
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        """Iterate over messages and attachments in the MCAP file.
        Args:
            keys (Optional[Iterable[str]]): Specific keys to include in the samples.
                The keys can be topic names or attachment names. If None, will ignore this filter.
                If provided, the keys must be unique across topics and attachments.
            topics (Optional[Iterable[str]]): Specific topics to include in the samples.
                If None, will include all topics.
            attachments (Optional[Iterable[str]]): Specific attachments to include in the samples.
                If None, will include all attachments.
        Returns:
            Generator[Dict[str, Any]]: A generator yielding dictionaries containing message and attachment data.
        Raises:
            ValueError: If the keys are not unique across topics and attachments.
            ValueError: If the topics or attachments are not found.
        """
        all_topics = self.all_topic_names()
        all_attachments = self.all_attachment_names()
        topics = set(topics) if topics is not None else all_topics
        attachments = set(attachments) if attachments is not None else all_attachments
        keys = keys or []
        for key in keys:
            flag = 0
            if key in all_topics:
                topics.add(key)
                flag += 1
            if key in all_attachments:
                attachments.add(key)
                flag += 1
            if flag == 0:
                raise ValueError(
                    f"Key '{key}' not found in topics or attachments. Available topics: {all_topics}, attachments: {all_attachments}."
                )
            elif flag > 1:
                raise ValueError(
                    f"Key '{key}' found in both topics and attachments, please specify only one."
                )

        def empty_iter():
            for _ in range(len(self)):
                yield {}

        # The first iteration costs more time since it needs to create these iterators.
        topic_iter = (
            self.iter_message_samples(topics, reverse) if topics else empty_iter()
        )
        attachment_iter = (
            self.iter_attachment_samples(attachments, reverse)
            if attachments
            else empty_iter()
        )
        for msg_data, att_data in zip(topic_iter, attachment_iter):
            data = {}
            data.update(msg_data)
            data.update(att_data)
            yield data

    def topic_message_counts(self) -> Dict[str, int]:
        """Get the message count for each topic in the MCAP file."""
        topic_msg_count = {}
        summary = self.reader.get_summary()
        statistics = summary.statistics
        for c_id, stats in statistics.channel_message_counts.items():
            # get topic name from channel id
            topic = summary.channels[c_id].topic
            topic_msg_count[topic] = stats
        return topic_msg_count

    @staticmethod
    def equal_message_counts(counts: Dict[str, int]) -> int:
        """Check if all topics have the same number of messages.
        Args:
            counts (Dict[str, int]): A dictionary mapping topic names to their message counts.
        Returns:
            int: The common message count if all topics have the same count, otherwise 0.
        Raises:
            AssertionError: If the counts dictionary is empty or contains non-positive counts.
        """
        assert counts, "Counts dictionary is empty"
        counts = list(counts.values())
        first_count = counts[0]
        for count in counts[1:]:
            assert count > 0, "Message count must be positive"
            if count != first_count:
                return 0
        return first_count

    @cache
    def __len__(self) -> int:
        """Get the total number of messages in the MCAP file."""
        counts = self.topic_message_counts()
        length = self.equal_message_counts(counts)
        if length == 0:
            if counts:
                raise ValueError(
                    f"Not all topics have the same number of messages. Counts: {counts}"
                )
            else:
                raise ValueError("No messages found in the MCAP file.")
        return length


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

    jpeg = TurboJPEG()
    av_coder = AvCoder()
    reader = make_reader(file)
    mfb_writer = McapFlatbufferWriter()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
    smapping = mfb_writer.register_schemas(writer, {FlatbufferSchemas.COMPRESSED_IMAGE})
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
                schema_id=smapping[FlatbufferSchemas.COMPRESSED_IMAGE],
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
