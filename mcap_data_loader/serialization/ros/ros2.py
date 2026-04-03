from rclpy.clock import Clock, ClockType
from ament_index_python.resources import get_resources, get_resource
from rosidl_runtime_py.utilities import get_message  # noqa: F401
from rosidl_runtime_py import set_message_fields, get_interface_path  # noqa: F401
from rosidl_adapter.parser import parse_message_file, Field
from builtin_interfaces.msg import Time as TimeMsg
from typing import Tuple, Optional, Dict, Set


def build_short_to_full_msg_map(
    preferred_packages: tuple = (),
):
    """Build a mapping from short message names to their full paths.
    Args:
        preferred_packages (tuple): Packages to prioritize when there are name conflicts.
    Returns:
        dict: A mapping from short message names to full paths.
    """
    mapping = {}

    packages = list(get_resources("rosidl_interfaces"))
    ordered_pkgs = [p for p in preferred_packages if p in packages]
    ordered_pkgs += [p for p in packages if p not in ordered_pkgs]

    for pkg in ordered_pkgs:
        try:
            content, _ = get_resource("rosidl_interfaces", pkg)
        except Exception:
            continue

        for line in content.splitlines():
            entry = line.strip()
            if not entry.startswith("msg/"):
                continue

            name = entry.split("/")[-1]
            if name.endswith(".idl"):
                name = name[:-4]
            if name.endswith(".msg"):
                name = name[:-4]

            full_path = f"{pkg}/msg/{name}"
            mapping.setdefault(name, full_path)

    return mapping


def get_fields_and_field_types(msg) -> Dict[str, str]:
    if not isinstance(msg, type):
        msg = type(msg)
    return msg.get_fields_and_field_types()


def time_ns_to_stamp(time_ns: int) -> TimeMsg:
    factor = 1_000_000_000
    return TimeMsg(sec=time_ns // factor, nanosec=time_ns % factor)


def stamp_to_time_ns(stamp: TimeMsg) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


_ros_clock = Clock(clock_type=ClockType.ROS_TIME)


def get_current_stamp() -> TimeMsg:
    return _ros_clock.now().to_msg()


DATA_TYPE_AND_MSGDEF_TEXT = {}


def get_full_definition(
    package_name: str, interface_name: str, seen_interfaces: Optional[Set[str]] = None
):
    if seen_interfaces is None:
        seen_interfaces = set()

    interface_key = f"{package_name}/msg/{interface_name}"
    if interface_key in seen_interfaces:
        return ""
    seen_interfaces.add(interface_key)

    msg_path = get_interface_path(interface_key)

    with open(msg_path, "r", encoding="utf-8") as f:
        main_content = f.read().strip()

    parsed_msg = parse_message_file(package_name, msg_path)

    sub_definitions = []
    field: Field
    for field in parsed_msg.fields:
        if not field.type.is_primitive_type():
            dep_pkg = field.type.pkg_name
            dep_msg = field.type.type
            dep_def = get_full_definition(dep_pkg, dep_msg, seen_interfaces)
            if dep_def:
                header = f"\n\n{'=' * 80}\nMSG: {dep_pkg}/{dep_msg}\n"
                sub_definitions.append(header + dep_def)

    return main_content + "".join(sub_definitions)


def get_datatype_and_msgdef_text(msg) -> Tuple[str, str]:
    """Get message datatype and its .msg definition text.
    Args:
        msg: ROS message instance or class.
    Returns:
        tuple: (datatype string, msg definition string)
    """
    if not isinstance(msg, type):
        msg_cls = type(msg)
    else:
        msg_cls = msg
    CACHE = DATA_TYPE_AND_MSGDEF_TEXT
    if msg_cls in CACHE:
        return CACHE[msg_cls]

    # Extract canonical datatype: pkg/msg/Type
    module_parts = msg_cls.__module__.split(".")
    if len(module_parts) >= 3 and module_parts[-2] == "msg":
        package = module_parts[-3]
        msg_name = msg_cls.__name__
        msg_type = f"{package}/msg/{msg_name}"
    else:
        raise ValueError(f"Cannot determine ROS2 message type from {msg_cls}")

    clean_text = get_full_definition(package, msg_name)
    CACHE[msg_cls] = (msg_type, clean_text)
    return msg_type, clean_text


if __name__ == "__main__":
    from geometry_msgs.msg import TransformStamped

    mapping = build_short_to_full_msg_map()
    print(f"Total messages found: {len(mapping)}")
    from pprint import pprint

    pprint(mapping)

    pprint(get_fields_and_field_types(TransformStamped()))
