import os
import sys
import logging
from importlib import import_module
from typing import TYPE_CHECKING, Optional, Dict, Any
from pathlib import Path
from contextlib import suppress


logger = logging.getLogger("ros_utils")


def find_ros_distro(path: str) -> Optional[str]:
    path_obj = Path(path)

    parts = path_obj.parts
    with suppress(ValueError, IndexError):
        ros_index = parts.index("ros")
        return parts[ros_index + 1]


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
    raise ImportError("No ROS paths found")
elif len(ros_paths) > 1:
    logger.warning("Multiple ROS paths found")

ROS_VERSION = os.environ["ROS_VERSION"]
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
else:
    module = import_module(f"airdc.common.utils.ros.ros{ROS_VERSION}")
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
