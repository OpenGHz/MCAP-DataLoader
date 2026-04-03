import os
import rospkg
import genpy
import rospy
import logging
from typing import Any, Dict, List, Callable
from roslib.message import get_message_class as get_message  # noqa: F401
from genpy import Time
import time


def build_short_to_full_msg_map(preferred_packages: tuple = ()):
    """
    Build a mapping from short message names to their full paths in ROS1.

    Args:
        preferred_packages (tuple): Packages to prioritize when there are name conflicts.

    Returns:
        dict: A mapping from short message names (e.g., 'String') to full paths (e.g., 'std_msgs/String').
    """
    mapping = {}
    rospack = rospkg.RosPack()

    # Get all packages that have a 'msg' directory
    all_pkgs_with_msgs = []
    for pkg in rospack.list():
        msg_dir = os.path.join(rospack.get_path(pkg), "msg")
        if os.path.isdir(msg_dir):
            all_pkgs_with_msgs.append(pkg)

    # Order packages: preferred first, then others
    ordered_pkgs = [p for p in preferred_packages if p in all_pkgs_with_msgs]
    ordered_pkgs += [p for p in all_pkgs_with_msgs if p not in ordered_pkgs]

    for pkg in ordered_pkgs:
        msg_dir = os.path.join(rospack.get_path(pkg), "msg")
        try:
            msg_files = os.listdir(msg_dir)
        except OSError:
            continue

        for f in msg_files:
            if f.endswith(".msg"):
                name = f[:-4]  # Remove .msg extension
                full_path = f"{pkg}/{name}"
                # Only set if not already present (so preferred packages win)
                if name not in mapping:
                    mapping[name] = full_path

    return mapping


def set_message_fields(
    msg: Any,
    values: Dict[str, Any],
    expand_header_auto: bool = False,
    expand_time_now: bool = False,
) -> List[Callable[[], None]]:
    """
    Set the fields of a ROS1 message from a dictionary.

    :param msg: The ROS1 message instance to populate.
    :param values: Dictionary mapping field names to values.
                   Special values:
                     - 'auto' for std_msgs/Header (if expand_header_auto=True)
                     - 'now' for time fields (if expand_time_now=True)
    :param expand_header_auto: If True and a Header field is given value 'auto',
                               create an empty Header and defer stamp setting.
    :param expand_time_now: If True and a time field is given value 'now',
                            defer setting to current time via returned setter.
    :returns: List of callable setters to assign current time (e.g., for stamp).
              Call them later with `setter()` to set to rospy.Time.now().
    """
    if not isinstance(values, dict):
        raise TypeError(f"values must be a dict: got {type(values)}")

    setters = []

    def _set_field(field_name: str, value: Any, target_obj: Any):
        if not hasattr(target_obj, field_name):
            raise AttributeError(
                f"Message {type(target_obj)} has no field '{field_name}'"
            )

        current_value = getattr(target_obj, field_name)

        # Handle nested message
        if isinstance(current_value, genpy.Message):
            if isinstance(value, dict):
                # Recurse into nested message
                _process_fields(current_value, value)
            elif expand_header_auto and value == "auto":
                # Special case: Header auto
                from std_msgs.msg import Header

                if isinstance(current_value, Header):
                    new_header = Header()
                    setattr(target_obj, field_name, new_header)

                    # Return a setter for stamp
                    def make_setter(hdr):
                        return lambda: setattr(hdr, "stamp", rospy.Time.now())

                    setters.append(make_setter(new_header))
                else:
                    raise ValueError(
                        f"'auto' is only valid for std_msgs/Header, got {type(current_value)}"
                    )
            else:
                raise TypeError(
                    f"Expected dict or 'auto' for message field '{field_name}', got {type(value)}"
                )
        # Handle time-like fields (genpy.Time or rospy.Time)
        elif isinstance(current_value, (genpy.Time, rospy.Time)):
            if expand_time_now and value == "now":

                def make_time_setter(obj, attr):
                    return lambda: setattr(obj, attr, rospy.Time.now())

                setters.append(make_time_setter(target_obj, field_name))
            else:
                # Try to convert value to Time
                try:
                    if isinstance(value, (genpy.Time, rospy.Time)):
                        time_val = value
                    elif isinstance(value, (tuple, list)) and len(value) == 2:
                        time_val = genpy.Time(*value)
                    elif isinstance(value, int):
                        # Assume secs
                        time_val = genpy.Time(value, 0)
                    else:
                        raise ValueError(f"Cannot convert {value} to Time")
                    setattr(target_obj, field_name, time_val)
                except Exception as e:
                    raise TypeError(
                        f"Cannot assign {value} to time field '{field_name}': {e}"
                    )
        # Handle arrays
        elif isinstance(current_value, list):
            if not isinstance(value, list):
                raise TypeError(
                    f"Expected list for array field '{field_name}', got {type(value)}"
                )
            elem_type = None
            if len(current_value) > 0:
                elem_type = type(current_value[0])
            new_list = []
            for i, item in enumerate(value):
                if elem_type and issubclass(elem_type, genpy.Message):
                    if not isinstance(item, dict):
                        raise TypeError(
                            f"Array element {i} for message array must be dict"
                        )
                    elem = elem_type()
                    _process_fields(elem, item)
                    new_list.append(elem)
                else:
                    new_list.append(item)
            setattr(target_obj, field_name, new_list)
        # Primitive field (int, float, string, bool, etc.)
        else:
            setattr(target_obj, field_name, value)

    def _process_fields(obj: Any, field_dict: Dict[str, Any]):
        for key, val in field_dict.items():
            _set_field(key, val, obj)

    _process_fields(msg, values)
    return setters


def get_fields_and_field_types(msg) -> Dict[str, str]:
    if not isinstance(msg, type):
        msg = type(msg)
    return dict(zip(msg.__slots__, msg._slot_types))


def time_ns_to_stamp(time_ns: int) -> Time:
    factor = 1_000_000_000
    return Time(secs=time_ns // factor, nsecs=time_ns % factor)


def stamp_to_time_ns(stamp: Time) -> int:
    return stamp.to_nsec()


def get_current_stamp() -> Time:
    if rospy.core.is_initialized():
        return rospy.Time.now()
    else:
        return time_ns_to_stamp(time.time_ns())


def get_datatype_and_msgdef_text(msg) -> tuple[str, str]:
    """Get message datatype and its .msg definition text.
    Args:
        msg: ROS message instance or class.
    Returns:
        tuple: (datatype string, msg definition string)
    """
    if not isinstance(msg, type):
        msg = type(msg)
    return msg._type, msg._full_text


def process_camera_info_dict(cam_info_dict: Dict[str, Any]):
    """Process camera info dictionary to match ROS1 CameraInfo field names.
    The keys 'd', 'k', 'r', 'p' are converted to uppercase. And this modifies
    the input dictionary in-place.
    Args:
        cam_info_dict: Original camera info dictionary.
    """
    for key in ("d", "k", "r", "p"):
        if key in cam_info_dict:
            cam_info_dict[key.upper()] = cam_info_dict.pop(key)


def init_ros_node_safe(name: str, **kwargs):
    root_logger = logging.getLogger()
    prev_level = root_logger.level
    prev_handlers = root_logger.handlers[:]

    rospy.init_node(name, **kwargs)

    root_logger.setLevel(prev_level)
    root_logger.handlers = prev_handlers
