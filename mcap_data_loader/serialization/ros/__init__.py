import os
import sys
import logging
from importlib import import_module
from typing import TYPE_CHECKING, Optional, Dict, Any
from pathlib import Path
from contextlib import suppress
from pydantic import BaseModel
from typing import ClassVar
from typing_extensions import Self, TypedDict
from inflection import camelize


logger = logging.getLogger("ros_utils")


def find_ros_distro(path: str) -> Optional[str]:
    path_obj = Path(path)

    parts = path_obj.parts
    with suppress(ValueError, IndexError):
        ros_index = parts.index("ros")
        return parts[ros_index + 1]


ROS_VERSION = os.environ["ROS_VERSION"]
# The Python path may contain multiple versions of ROS,
# causing subsequent program processing errors.
ROS_DISTRO = os.environ["ROS_DISTRO"]
ros_paths = []
for path in sys.path.copy():
    if path.endswith("site-packages") or path.endswith("dist-packages"):
        ros_distro = find_ros_distro(path)
        if ros_distro is not None:
            if ros_distro != ROS_DISTRO:
                logger.warning(f"Removing {path} from sys.path for ROS distro mismatch")
                sys.path.remove(path)
            else:
                ros_paths.append(path)
logger.info(f"Using ROS distro: {ROS_DISTRO} with path: {ros_paths}")
if len(ros_paths) == 0:
    # ROS packages may be installed in generic site-packages (e.g. pixi/conda)
    # without a path containing "ros/<distro>". Verify by trying to import.
    try:
        import importlib

        importlib.import_module("rclpy" if ROS_VERSION == "2" else "rospy")
        logger.info("ROS packages found in generic site-packages (pixi/conda)")
    except ImportError:
        raise ImportError("No ROS paths found")
elif len(ros_paths) > 1:
    logger.warning("Multiple ROS paths found")


if TYPE_CHECKING:

    def build_short_to_full_msg_map(preferred_packages) -> dict: ...
    def get_message(identifier: str) -> type: ...
    def set_message_fields(
        msg, values, expand_header_auto=False, expand_time_now=False
    ) -> list: ...
    def get_fields_and_field_types(msg) -> Dict[str, str]: ...
    def time_ns_to_stamp(time_ns: int) -> Any: ...
    def stamp_to_time_ns(stamp: Any) -> int: ...
    def get_datatype_and_msgdef_text(msg) -> tuple: ...
    def process_camera_info_dict(cam_info_dict: Dict[str, Any]): ...
    def get_current_stamp() -> Any: ...
    def stamp_from_dict(stamp_dict: Dict[str, int]) -> Any: ...
else:
    module = import_module(f"mcap_data_loader.serialization.ros.ros{ROS_VERSION}")
    build_short_to_full_msg_map = module.build_short_to_full_msg_map
    get_message = module.get_message
    set_message_fields = module.set_message_fields
    get_fields_and_field_types = module.get_fields_and_field_types
    time_ns_to_stamp = module.time_ns_to_stamp
    stamp_to_time_ns = module.stamp_to_time_ns
    get_datatype_and_msgdef_text = module.get_datatype_and_msgdef_text
    process_camera_info_dict = getattr(
        module, "process_camera_info_dict", lambda x: None
    )
    get_current_stamp = module.get_current_stamp
    stamp_from_dict = module.stamp_from_dict


MSG_MAP: dict = {}


def get_message_short(identifier: str, cache: bool = True) -> Optional[type]:
    global MSG_MAP
    if cache:
        if not MSG_MAP:
            preferred = ("std_msgs", "geometry_msgs", "sensor_msgs")
            if ROS_VERSION != "1":
                preferred += ("builtin_interfaces",)
            MSG_MAP = build_short_to_full_msg_map(preferred)
        full_identifier = MSG_MAP.get(identifier)
    else:
        full_identifier = build_short_to_full_msg_map(preferred).get(identifier)
    if full_identifier is not None:
        return get_message(full_identifier)


class MessageDict(TypedDict):
    value: Dict[str, dict]
    t: int
    log_time: int


class TopicInfo(BaseModel, frozen=True):
    topic: str
    msg_name_snake: str
    msg_type: type
    msg_type_stamped: Optional[type]
    fields_and_field_types: Dict[str, type]
    msg_dict: MessageDict = {"value": {}, "t": 0, "log_time": 0}
    topic_info: ClassVar[Dict[str, Self]] = {}

    @classmethod
    def from_topic_name(cls, topic: str) -> Optional[Self]:
        if topic in cls.topic_info:
            return cls.topic_info[topic]
        # e.g. topic: /arm/joint_state/position -> msg_name_snake: JointState -> msg_type: sensor_msgs.msg._JointState.JointState
        split = topic.rsplit("/", 1)
        if len(split) != 2:
            return None
        msg_name_no_stamp = split[1]
        msg_identifier = camelize(msg_name_no_stamp)
        msg_type = get_message_short(msg_identifier)
        if msg_type is not None:
            msg_type_stamped = get_message_short(msg_identifier + "Stamped")
            _PRIMITIVE_TYPE_MAP = {
                "bool": bool,
                "boolean": bool,
                "byte": int,
                "char": int,
                "octet": int,
                "int8": int,
                "uint8": int,
                "int16": int,
                "uint16": int,
                "int32": int,
                "uint32": int,
                "int64": int,
                "uint64": int,
                "float32": float,
                "float64": float,
                "string": str,
                "wstring": str,
                "time": int,
                "duration": int,
            }
            fields_and_field_types = {}
            for filed, field_type_str in get_fields_and_field_types(msg_type).items():
                if field_type_str.startswith("sequence") or "[" in field_type_str:
                    field_type = list
                elif field_type_str in _PRIMITIVE_TYPE_MAP:
                    field_type = _PRIMITIVE_TYPE_MAP[field_type_str]
                else:
                    field_type = get_message(field_type_str)
                fields_and_field_types[filed] = field_type
            instance = cls(
                topic=topic,
                msg_name_snake=msg_name_no_stamp,
                msg_type=msg_type,
                msg_type_stamped=msg_type_stamped,
                fields_and_field_types=fields_and_field_types,
            )
            cls.topic_info[topic] = instance
            return instance

    @property
    def has_stamp(self) -> bool:
        return self.msg_type_stamped is not None


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # if ROS_VERSION == "1":
    #     import rospy

    #     rospy.init_node("test_set_message_fields")

    # mapping = build_short_to_full_msg_map()
    # print(f"Total messages found: {len(mapping)}")
    # from pprint import pprint

    # pprint(mapping)

    # from std_msgs.msg import String

    # msg = String()
    # set_message_fields(msg, {"data": "Hello"})
    # assert msg.data == "Hello", msg.data

    # from geometry_msgs.msg import PointStamped

    # msg = PointStamped()
    # setters = set_message_fields(msg, {"header": "auto"}, expand_header_auto=True)
    # for s in setters:
    #     s()  # sets msg.header.stamp = rospy.Time.now()
    # print(msg.header.stamp)
    # from geometry_msgs.msg import PointStamped

    # msg = PointStamped()
    # setters = set_message_fields(
    #     msg,
    #     {"header": {"stamp": "now"}, "point": {"x": 1.0, "y": 2.0, "z": 0.0}},
    #     expand_time_now=True,
    # )

    # # Apply time now
    # for s in setters:
    #     s()

    # print(msg.header.stamp)  # Current time

    # assert get_message_short("PointStamped") == PointStamped

    # pprint(get_fields_and_field_types(PointStamped()))

    # ns = 156789123456789
    # stamp = time_ns_to_stamp(ns)
    # print(stamp)
    # assert stamp_to_time_ns(stamp) == ns

    # from std_msgs.msg import String

    # datatype, msgdef = get_datatype_and_msgdef_text(String)
    # print(datatype, msgdef)

    pass
