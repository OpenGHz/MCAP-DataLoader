from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from typing import Tuple, Optional, Callable


Position = Tuple[float, float, float]
Orientation = Tuple[float, float, float, float]
Pose = Tuple[Position, Orientation]


class TFPublisher(Node):
    def __init__(self, node_name: str = "tf_publisher"):
        super().__init__(node_name)
        self.tf_broadcaster = TransformBroadcaster(self)

    def broadcast_tf(
        self,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
        child_frame_id: str,
        frame_id: str = "/base_link",
    ):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id

        t.transform.translation.x = position[0]
        t.transform.translation.y = position[1]
        t.transform.translation.z = position[2]

        t.transform.rotation.x = orientation[0]
        t.transform.rotation.y = orientation[1]
        t.transform.rotation.z = orientation[2]
        t.transform.rotation.w = orientation[3]

        self.tf_broadcaster.sendTransform(t)


class TransformListenerPro(TransformListener):
    """
    :class:`TransformDiscover` is a convenient way to listen for coordinate frame transformation info.
    This class takes an object that instantiates the :class:`BufferInterface` interface, to which
    it propagates changes to the tf frame graph.
    """

    def __init__(
        self,
        buffer: Buffer,
        node: Node,
        *,
        tf_topic: str = "tf",
        spin_thread=False,
        qos=None,
        static_qos=None,
    ):
        """
        Constructor.

        :param buffer: The buffer to propagate changes to when tf info updates.
        :param node: The ROS2 node.
        :param tf_topic: The topic to subscribe to for dynamic transforms.
        :param spin_thread: Whether to create a dedidcated thread to spin this node.
        :param qos: A QoSProfile or a history depth to apply to subscribers.
        :param static_qos: A QoSProfile or a history depth to apply to tf_static subscribers.
        """
        super().__init__(
            buffer, node, spin_thread=spin_thread, qos=qos, static_qos=static_qos
        )
        qos = self.tf_sub.qos_profile
        node.destroy_subscription(self.tf_sub)
        self.tf_sub = node.create_subscription(
            TFMessage, tf_topic, self.callback, qos, callback_group=self.group
        )
        self._tf_callbacks = []

    def add_callback(self, callback: Callable):
        """
        Add a callback to be called when a new transform is received.

        :param callback: The callback function to be called.
        """
        self._tf_callbacks.append(callback)

    def callback(self, data):
        super().callback(data)
        for callback in self._tf_callbacks:
            callback(data)


class TFDiscover:
    def __init__(self, node: Node, tf_topic: str = "tf"):
        """TFDiscover is a ROS2 node that listens for coordinate frame transformations.
        It uses a TransformListenerPro to manage the buffer of transforms and provides a method to get the transform between two frames.
        """
        self._node = node
        self._buffer = Buffer()
        self.listener = TransformListenerPro(self._buffer, node, tf_topic=tf_topic)

    def get_transform(self, target_frame: str, source_frame: str) -> Optional[Pose]:
        """
        Get the transform between two frames.

        :param target_frame: The target frame to which the source frame is transformed.
        :param source_frame: The source frame to be transformed.
        :return: The transform between the two frames.
        """
        try:
            transform: TransformStamped = self._buffer.lookup_transform(
                target_frame, source_frame, Time()
            )
            trans = transform.transform.translation
            rot = transform.transform.rotation
            return (trans.x, trans.y, trans.z), (rot.x, rot.y, rot.z, rot.w)
        except Exception as e:
            self.get_logger().error(f"Failed to get transform: {e}")
            return None

    def get_logger(self):
        """
        Get the logger for this node.

        :return: The logger for this node.
        """
        return self._node.get_logger().get_child(self.__class__.__name__)
