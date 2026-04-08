from mcap_data_loader.serialization.ros import (
    ROS_VERSION,
    get_datatype_and_msgdef_text,
    get_message,
    get_message_short,
    set_message_fields,
)
from typing import Any, Optional, final
from mcap.writer import Writer as McapWriter
from mcap.well_known import SchemaEncoding, MessageEncoding
from mcap_data_loader.serialization.basis import McapReaderBasis, McapWriterBasis
from abc import abstractmethod


class McapROSWriterBasis(McapWriterBasis):
    @final
    def _resolve_type(self, schema_type) -> type:
        """Resolve schema_type to a ROS message class.

        Args:
            schema_type: A ROS message class, a short name (e.g. "PoseStamped"),
                or a full type name (e.g. "geometry_msgs/msg/PoseStamped").
        """
        if isinstance(schema_type, str):
            msg_cls = get_message_short(schema_type)
            if msg_cls is None:
                msg_cls = get_message(schema_type)
            return msg_cls

        return schema_type

    @final
    def _get_schema_name_and_data(self, schema_type):
        msg_cls = self._resolve_type(schema_type)
        msg_type, msg_def = get_datatype_and_msgdef_text(msg_cls)
        return (msg_type, msg_def.encode())

    @abstractmethod
    def _serialize_message(self, message: Any) -> bytes:
        """Serialize a ROS message to bytes for writing to MCAP."""

    @final
    def on_add_message(
        self, schema_type, topic, data, publish_time, log_time, **kwargs
    ):
        if isinstance(data, dict):
            msg = schema_type()
            set_message_fields(msg, data)
            data = msg
        self._writer.add_message(
            channel_id=self._cmapping[topic],
            data=self._serialize_message(data),
            publish_time=publish_time,
            log_time=log_time,
        )


if ROS_VERSION == "1":
    from mcap_ros1.writer import Writer  # noqa: F401
    from mcap_ros1.decoder import DecoderFactory
    from mcap.well_known import MessageEncoding
    from io import BytesIO

    class McapROSReader(McapReaderBasis):
        def _post_init(self):
            self._decoder_factory = DecoderFactory().decoder_for

        def _decode(self, schema, message):
            return self._decoder_factory(MessageEncoding.ROS1, schema)(message.data)

    class McapROSWriter(McapROSWriterBasis):
        message_encoding = MessageEncoding.ROS1
        schema_encoding = SchemaEncoding.ROS1
        profile = "ros1"

        def _serialize_message(self, message: Any) -> bytes:
            buffer = BytesIO()
            message.serialize(buffer)
            return buffer.getvalue()


else:
    from io import BufferedWriter
    from typing import IO, Any, Dict, Optional, Union
    from mcap.writer import CompressionType
    from rclpy.serialization import serialize_message, deserialize_message
    from airdc import __version__
    import time
    import mcap

    def _library_identifier():
        mcap_version = getattr(mcap, "__version__", "<=0.0.10")
        return (
            f"mcap-ros{ROS_VERSION}-support-OpenGHz {__version__}; mcap {mcap_version}"
        )

    class Writer:
        def __init__(
            self,
            output: Union[str, IO[Any], BufferedWriter],
            chunk_size: int = 1024 * 1024,
            compression: CompressionType = CompressionType.ZSTD,
            enable_crcs: bool = True,
        ):
            self._ros = "ros" + ROS_VERSION
            ros_upper = self._ros.upper()
            self._schema_encoding = getattr(SchemaEncoding, ros_upper)
            self._message_encoding = {"2": MessageEncoding.CDR}.get(
                ROS_VERSION
            ) or getattr(MessageEncoding, ros_upper)
            self._metadata = {"offered_qos_profiles": "[]", "topic_type_hash": ""}
            self.__writer = McapWriter(
                output=output,
                chunk_size=chunk_size,
                compression=compression,
                enable_crcs=enable_crcs,
            )
            self.__schema_ids: Dict[str, int] = {}
            self.__channel_ids: Dict[str, int] = {}
            self.__writer.start(profile=self._ros, library=_library_identifier())
            self.__finished = False

        def finish(self):
            """
            Finishes writing to the MCAP stream. This must be called before the stream is closed.
            """
            if not self.__finished:
                self.__writer.finish()
                self.__finished = True

        def write_message(
            self,
            topic: str,
            message: Any,
            log_time: Optional[int] = None,
            publish_time: Optional[int] = None,
            sequence: int = 0,
        ):
            """
            Writes a message to the MCAP stream, automatically registering schemas and channels as
            needed.

            :param topic: The topic of the message.
            :param message: The message to write.
            :param log_time: The time at which the message was logged as a nanosecond UNIX timestamp.
                Will default to the current time if not specified.
            :param publish_time: The time at which the message was published as a nanosecond UNIX
                timestamp. Will default to ``log_time`` if not specified.
            :param sequence: An optional sequence number.
            """
            msg_type, msg_def = get_datatype_and_msgdef_text(message)
            if msg_type not in self.__schema_ids:
                schema_id = self.__writer.register_schema(
                    msg_type, self._schema_encoding, msg_def.encode()
                )
                self.__schema_ids[msg_type] = schema_id
            schema_id = self.__schema_ids[msg_type]
            if topic not in self.__channel_ids:
                channel_id = self.__writer.register_channel(
                    topic, self._message_encoding, schema_id, self._metadata
                )
                self.__channel_ids[topic] = channel_id
            channel_id = self.__channel_ids[topic]

            if log_time is None:
                log_time = time.time_ns()
            if publish_time is None:
                publish_time = log_time
            self.__writer.add_message(
                channel_id=channel_id,
                log_time=log_time,
                publish_time=publish_time,
                sequence=sequence,
                data=serialize_message(message),
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_: Any, exc_type_: Any, tb_: Any):
            self.finish()

    class McapROSReader(McapReaderBasis):
        def _decode(self, schema, message):
            msg_type = get_message(schema.name)
            ros_msg = deserialize_message(message.data, msg_type)
            return ros_msg

    class McapROSWriter(McapROSWriterBasis):
        message_encoding = MessageEncoding.CDR
        schema_encoding = SchemaEncoding.ROS2
        profile = "ros2"

        def _serialize_message(self, message: Any) -> bytes:
            return serialize_message(message)


def get_mcap_writer(writer: Writer) -> McapWriter:
    return writer._Writer__writer
