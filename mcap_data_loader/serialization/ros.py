from mcap_data_loader.serialization.basis import McapReaderBasis
import os

_ROS_VERSION = os.environ["ROS_VERSION"]


if _ROS_VERSION == "1":
    from mcap_ros1.decoder import DecoderFactory
    from mcap.well_known import MessageEncoding

    class McapROSReader(McapReaderBasis):
        def _post_init(self):
            self._decoder_factory = DecoderFactory().decoder_for

        def _decode(self, schema, message):
            return self._decoder_factory(MessageEncoding.ROS1, schema)(message.data)


elif _ROS_VERSION == "2":
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    class McapROSReader(McapReaderBasis):
        def _decode(self, schema, message):
            msg_type = get_message(schema.name)
            ros_msg = deserialize_message(message.data, msg_type)
            return ros_msg
else:
    if not _ROS_VERSION:
        raise ValueError("ROS_VERSION environment variable is not set.")
    raise ValueError(f"Unsupported ROS version: {_ROS_VERSION}")
