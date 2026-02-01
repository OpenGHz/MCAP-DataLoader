"""Transformation utilities using numpy."""

from mcap_data_loader.utils.transformations import (
    compose_matrix,
    euler_from_quaternion,
    quaternion_from_matrix,
    translation_from_matrix,
    identity_matrix,
)
from functools import wraps
from typing import Sequence, Tuple, List, Dict, NamedTuple, Optional
from collections import deque, defaultdict
from logging import getLogger
import numpy as np


class TransformItem(NamedTuple):
    """A transformation between two frames."""

    target_frame: str
    """Name of the frame to transform into."""
    source_frame: str
    """Name of the input frame."""
    tf: np.ndarray
    """4x4 transformation matrix from source_frame to target_frame."""


def pose2matrix(position: Sequence[float], orientation: Sequence[float]) -> np.ndarray:
    """Convert position and orientation to a 4x4 transformation matrix.

    Args:
        position (list or np.ndarray): A list or array of 3 elements representing the position (x, y, z).
        orientation (list or np.ndarray): A list or array of 4 elements representing the orientation as a quaternion (x, y, z, w).

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    return compose_matrix(translate=position, angles=euler_from_quaternion(orientation))


def position2matrix(position: Sequence[float]) -> np.ndarray:
    """Convert position to a 4x4 transformation matrix with no rotation.

    Args:
        position (list or np.ndarray): A list or array of 3 elements representing the position (x, y, z).

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    return pose2matrix(position, [0, 0, 0, 1])


def is_matrix(input_data: Sequence[float]) -> bool:
    """Check if the input data is a 4x4 transformation matrix.

    Args:
        input_data (list or np.ndarray): The input data to check.
    Returns:
        bool: True if the input data is a 4x4 matrix, False otherwise.
    """
    return np.asanyarray(input_data).shape == (4, 4)


def to_matrix(pos_ori: Sequence[float]) -> np.ndarray:
    """Convert a combined position and orientation list to a 4x4 transformation matrix.

    Args:
        pos_ori (list or np.ndarray): A list or array representing position and orientation.
            - If length is 2: treated as (position, orientation).
            - If length is 7: treated as (x, y, z, qx, qy, qz, qw).
            - If length is 3: treated as position only (no rotation).
            - If length is 4: treated as orientation only (no translation) or a 4x4 transformation matrix.
    Returns:
        np.ndarray: A 4x4 transformation matrix.
    Raises:
        ValueError: If the length of pos_ori is not 2, 3, 4, or 7.
    """
    if len(pos_ori) == 2:
        return pose2matrix(*pos_ori)
    elif len(pos_ori) == 7:
        return pose2matrix(pos_ori[:3], pos_ori[3:])
    elif len(pos_ori) == 3:
        return position2matrix(pos_ori)
    elif len(pos_ori) == 4:
        arr = np.asanyarray(pos_ori)
        if is_matrix(arr):
            return arr
        return pose2matrix([0, 0, 0], pos_ori)
    else:
        raise ValueError(
            f"Invalid length of pos_ori: {len(pos_ori)}. Expected 2, 3, 4, or 7."
        )


def matrix2pose(matrix: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 4x4 transformation matrix to position and orientation.

    Args:
        matrix (np.ndarray): A 4x4 transformation matrix.
    Returns:
        tuple: A tuple containing:
            - position (np.ndarray): A 1D array of 3 elements representing the position
            - orientation (np.ndarray): A 1D array of 4 elements representing the orientation as a quaternion (x, y, z, w).
    """
    position = translation_from_matrix(matrix)
    orientation = quaternion_from_matrix(matrix)
    return position, orientation


def tf_between_poses(
    position_a: Sequence[float],
    orientation_a: Sequence[float],
    position_b: Sequence[float],
    orientation_b: Sequence[float],
) -> np.ndarray:
    """Compute the transformation matrix from pose A to pose B.

    Args:
        position_a (list or np.ndarray): Position of pose A (x, y, z).
        orientation_a (list or np.ndarray): Orientation of pose A as a quaternion (x, y, z, w).
        position_b (list or np.ndarray): Position of pose B (x, y, z).
        orientation_b (list or np.ndarray): Orientation of pose B as a quaternion (x, y, z, w).

    Returns:
        np.ndarray: A 4x4 transformation matrix representing the transformation from pose A to pose B.
    """
    matrix_a = pose2matrix(position_a, orientation_a)
    matrix_b = pose2matrix(position_b, orientation_b)
    matrix_a_inv = np.linalg.inv(matrix_a)
    tf_matrix = np.dot(matrix_a_inv, matrix_b)
    return tf_matrix


def rela_pose_between_poses(
    position_a: Sequence[float],
    orientation_a: Sequence[float],
    position_b: Sequence[float],
    orientation_b: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the relative pose from pose A to pose B.

    Args:
        position_a (list or np.ndarray): Position of pose A (x, y, z).
        orientation_a (list or np.ndarray): Orientation of pose A as a quaternion (x, y, z, w).
        position_b (list or np.ndarray): Position of pose B (x, y, z).
        orientation_b (list or np.ndarray): Orientation of pose B as a quaternion (x, y, z, w).

    Returns:
        tuple: A tuple containing:
            - rela_position (np.ndarray): A 1D array of 3 elements representing the relative position.
            - rela_orientation (np.ndarray): A 1D array of 4 elements representing the relative orientation as a quaternion (x, y, z, w).
    """
    tf_matrix = tf_between_poses(position_a, orientation_a, position_b, orientation_b)
    return matrix2pose(tf_matrix)


def apply_tf_to_pose(
    position: Sequence[float], orientation: Sequence[float], tf_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a transformation matrix to a pose.

    Args:
        position (np.ndarray): A 1D array of 3 elements representing the position (x, y, z).
        orientation (np.ndarray): A 1D array of 4 elements representing the orientation as a quaternion (x, y, z, w).
        tf_matrix (np.ndarray): A 4x4 transformation matrix to be applied.

    Returns:
        tuple: A tuple containing:
            - new_position (np.ndarray): A 1D array of 3 elements representing the new position.
            - new_orientation (np.ndarray): A 1D array of 4 elements representing the new orientation as a quaternion (x, y, z, w).
    """
    pose_matrix = pose2matrix(position, orientation)
    new_pose_matrix = np.dot(pose_matrix, tf_matrix)
    new_position, new_orientation = matrix2pose(new_pose_matrix)
    return new_position, new_orientation


def apply_rela_pose_to_pose(
    position: Sequence[float],
    orientation: Sequence[float],
    rela_position: Sequence[float],
    rela_orientation: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a relative pose to a pose.

    Args:
        position (np.ndarray): A 1D array of 3 elements representing the position (x, y, z).
        orientation (np.ndarray): A 1D array of 4 elements representing the orientation as a quaternion (x, y, z, w).
        rela_position (np.ndarray): A 1D array of 3 elements representing the relative position (x, y, z).
        rela_orientation (np.ndarray): A 1D array of 4 elements representing the relative orientation as a quaternion (x, y, z, w).

    Returns:
        tuple: A tuple containing:
            - new_position (np.ndarray): A 1D array of 3 elements representing the new position.
            - new_orientation (np.ndarray): A 1D array of 4 elements representing the new orientation as a quaternion (x, y, z, w).
    """
    rela_pose_matrix = pose2matrix(rela_position, rela_orientation)
    return apply_tf_to_pose(position, orientation, rela_pose_matrix)


def array_pose_to_list(
    position: np.ndarray, orientation: np.ndarray
) -> Tuple[List[float], List[float]]:
    """Convert position and orientation arrays to a single list.

    Args:
        position (np.ndarray): A 1D array of 3 elements representing the position (x, y, z).
        orientation (np.ndarray): A 1D array of 4 elements representing the orientation as a quaternion (x, y, z, w).

    Returns:
        list: A list containing the position and orientation elements.
    """
    return position.tolist(), orientation.tolist()


def array_pose_to_list_wrapper(func, **p_kwargs):
    """A decorator to convert array pose outputs of a function to list pose."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return array_pose_to_list(*func(*args, **kwargs, **p_kwargs))

    return wrapper


def is_identity_matrix(matrix: np.ndarray, tol: float = 1e-6) -> bool:
    """Check if a matrix is an identity matrix within a tolerance.

    Args:
        matrix (np.ndarray): The matrix to check.
        tol (float): The tolerance for the check.

    Returns:
        bool: True if the matrix is an identity matrix within the tolerance, False otherwise.
    """
    return np.allclose(matrix, identity_matrix(), atol=tol)


class StaticTFBuffer:
    def __init__(self, buffer: List[TransformItem] = None):
        # 原始数据存储
        self.buffer: List[TransformItem] = []
        # 邻接表：{source_frame: [(target_frame, matrix), ...]}
        self._adj: Dict[str, List[Tuple[str, np.ndarray]]] = defaultdict(list)
        for transform in buffer or []:
            # print("Adding transform to StaticTFBuffer:", transform)
            self._add_transform(transform)
        # 缓存：避免对相同路径进行重复搜索
        self._cache: Dict[Tuple[str, str], np.ndarray] = {}

    def add_transform(self, transform: TransformItem):
        """添加或更新一个变换"""
        self._add_transform(transform)
        # 清除缓存，因为图结构改变了
        self._cache.clear()

    def _add_transform(self, transform: TransformItem):
        target_frame, source_frame, tf = transform
        mat = np.asanyarray(tf)
        self.buffer.append((target_frame, source_frame, mat))
        # 更新邻接表
        # 正向：A -> B
        self._adj[source_frame].append((target_frame, mat))
        # 逆向：B -> A (计算逆矩阵)
        try:
            # print("Computing inverse for transform:", transform)
            inv_mat = np.linalg.inv(mat)
            self._adj[target_frame].append((source_frame, inv_mat))
            # print("Inverse computed successfully.")
        except np.linalg.LinAlgError:
            self.get_logger().warning(f"Transform {transform} is not invertible.")

    def lookup_transform(
        self, target_frame: str, source_frame: str
    ) -> Optional[np.ndarray]:
        """
        查找变换。支持多级链式查找和逆变换。
        """
        # 1. 检查相同帧
        if source_frame == target_frame:
            return np.eye(4)

        # 2. 检查缓存
        cache_key = (source_frame, target_frame)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 3. BFS 路径搜索
        queue = deque([(source_frame, np.eye(4))])
        visited = {source_frame}

        while queue:
            curr_frame, curr_tf = queue.popleft()

            if curr_frame == target_frame:
                # 存入缓存并返回
                self._cache[cache_key] = curr_tf
                return curr_tf

            for neighbor, edge_tf in self._adj.get(curr_frame, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    # 矩阵合成：T_ac = T_bc @ T_ab
                    combined_tf = edge_tf @ curr_tf
                    queue.append((neighbor, combined_tf))

        return None

    def clear(self):
        self.buffer.clear()
        self._adj.clear()
        self._cache.clear()

    @classmethod
    def get_logger(cls):
        return getLogger(cls.__name__)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test functions for pose and transformation matrix conversions."
    )
    args = parser.parse_args()

    pos_a = [0.5, 0.0, 0.5]
    qat_a = [0.0, 0.0, 0.0, 1.0]
    pos_b = [0.5, 0.5, 0.5]
    qat_b = [0.0, 0.0, 0.7071, 0.7071]

    tf_ab = tf_between_poses(pos_a, qat_a, pos_b, qat_b)
    new_pos_b, new_qat_b = apply_tf_to_pose(pos_a, qat_a, tf_ab)
    new_pos_b_list, new_qat_b_list = array_pose_to_list_wrapper(
        apply_tf_to_pose, tf_matrix=tf_ab
    )(new_pos_b, new_qat_b)

    assert np.allclose(new_pos_b, pos_b)
    assert np.allclose(new_qat_b, qat_b)

    rela_pose_b = matrix2pose(tf_ab)
    rela_pose_b_new = rela_pose_between_poses(pos_a, qat_a, pos_b, qat_b)
    assert np.allclose(rela_pose_b[0], rela_pose_b_new[0])
    assert np.allclose(rela_pose_b[1], rela_pose_b_new[1])
    assert np.allclose(pose2matrix(*rela_pose_b), tf_ab)

    new_pos_b2, new_qat_b2 = apply_rela_pose_to_pose(pos_a, qat_a, *rela_pose_b)
    assert np.allclose(new_pos_b2, pos_b)
    assert np.allclose(new_qat_b2, qat_b)

    tf_manager = StaticTFBuffer()

    # 模拟数据：World -> Robot (平移 x=1)
    t_world_robot = np.eye(4)
    t_world_robot[0, 3] = 1.0
    tf_manager.add_transform(("robot", "world", t_world_robot))

    # 模拟数据：Robot -> Camera (平移 z=0.5)
    t_robot_camera = np.eye(4)
    t_robot_camera[2, 3] = 0.5
    tf_manager.add_transform(("camera", "robot", t_robot_camera))

    # 自动计算跨级变换：World -> Camera
    res = tf_manager.lookup_transform("camera", "world")
    print("World to Camera:\n", res)

    # 自动计算逆变换：Camera -> World
    res_inv = tf_manager.lookup_transform("world", "camera")
    print("\nCamera to World (Inverse):\n", res_inv)

    assert np.allclose(tf_manager.lookup_transform("world", "world"), np.eye(4))

    assert tf_manager.lookup_transform("camera", "any") is None

    print("All tests passed.")
