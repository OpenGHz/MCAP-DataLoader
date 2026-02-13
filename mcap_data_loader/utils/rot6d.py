import numpy as np
import math
from mcap_data_loader.utils.transformations import quaternion_matrix


def quaternion_from_matrix(matrix):
    """Return quaternion [x, y, z, w] from a 3x3 rotation matrix.

    >>> R = np.array([[1, 0, 0],
    ...                  [0, 0, -1],
    ...                  [0, 1, 0]])  # 90° around x-axis
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.70710678, 0.0, 0.0, 0.70710678])  # [x,y,z,w]
    True
    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    assert M.shape == (3, 3), "Input must be a 3x3 rotation matrix"

    # Compute trace of the matrix
    tr = np.trace(M)

    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2  # S = 4 * w
        w = 0.25 * S
        x = (M[2, 1] - M[1, 2]) / S
        y = (M[0, 2] - M[2, 0]) / S
        z = (M[1, 0] - M[0, 1]) / S
    else:
        # Find the major diagonal element with the greatest value
        if M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            S = math.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2  # S = 4 * x
            x = 0.25 * S
            y = (M[0, 1] + M[1, 0]) / S
            z = (M[0, 2] + M[2, 0]) / S
            w = (M[2, 1] - M[1, 2]) / S
        elif M[1, 1] > M[2, 2]:
            S = math.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2  # S = 4 * y
            x = (M[0, 1] + M[1, 0]) / S
            y = 0.25 * S
            z = (M[1, 2] + M[2, 1]) / S
            w = (M[0, 2] - M[2, 0]) / S
        else:
            S = math.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2  # S = 4 * z
            x = (M[0, 2] + M[2, 0]) / S
            y = (M[1, 2] + M[2, 1]) / S
            z = 0.25 * S
            w = (M[1, 0] - M[0, 1]) / S

    q = np.array([x, y, z, w], dtype=np.float64)
    return q


class Rotation6D:
    @staticmethod
    def rot6d_to_matrix(x: np.ndarray) -> np.ndarray:
        """
        将6D旋转表示转换为3x3旋转矩阵
        输入:
            x: [6]
        输出:
            R: [3, 3] 旋转矩阵
        """
        assert x.shape[-1] == 6, "输入必须是6D向量"

        # 拆分成两个3D向量
        a1 = x[0:3]  # [3]
        a2 = x[3:6]  # [3]

        # 第一列：单位化 a1
        norm_a1 = np.linalg.norm(a1)
        if norm_a1 == 0:
            norm_a1 = 1.0
        b1 = a1 / norm_a1  # [3]

        # 第二列：从 a2 中减去 b1 方向的分量，再单位化
        proj = np.dot(b1, a2)  # 标量
        ortho_a2 = a2 - proj * b1  # [3]

        norm_ortho = np.linalg.norm(ortho_a2)
        if norm_ortho == 0:
            norm_ortho = 1.0
        b2 = ortho_a2 / norm_ortho  # [3]

        # 第三列：b1 × b2
        b3 = np.cross(b1, b2)  # [3]

        # 拼接成旋转矩阵：每一列为 b1, b2, b3 → shape [3, 3]
        R = np.stack((b1, b2, b3), axis=1)  # [3, 3]
        return R

    @staticmethod
    def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
        """
        将旋转矩阵转换为6D旋转表示
        输入:
            R: [3, 3]
        输出:
            x: [6]
        """
        assert R.shape == (3, 3), "输入必须是3x3矩阵"
        # 取前两列并展平：R[:, 0] 和 R[:, 1] → 拼成 [6]
        return np.concatenate((R[:, 0], R[:, 1]), axis=0)  # [6]

    @staticmethod
    def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
        """
        将四元数转换为6D旋转表示
        输入:
            quat: [x, y, z, w] format
        输出:
            rot6d: [6]
        """
        R = quaternion_matrix(quat)[:3, :3]  # [3, 3]
        return Rotation6D.matrix_to_rot6d(R)  # [6]

    @staticmethod
    def rot6d_to_quat(rot6d: np.ndarray) -> np.ndarray:
        """
        将6D旋转表示转换为四元数
        输入:
            rot6d: [6]
        输出:
            quat: [4] in [x, y, z, w] format
        """
        R = Rotation6D.rot6d_to_matrix(rot6d)  # [3, 3]
        quat = quaternion_from_matrix(R)  # [4]
        return quat


class BatchedRotation6D:
    @staticmethod
    def rot6d_to_matrix(x: np.ndarray) -> np.ndarray:
        """
        将6D旋转表示转换为3x3旋转矩阵
        输入:
            x: [B, 6]  (B为批量大小)
        输出:
            R: [B, 3, 3] 旋转矩阵
        """
        assert x.shape[-1] == 6, "输入必须是6D向量"

        # 拆分成两个3D向量
        a1 = x[:, 0:3]  # [B, 3]
        a2 = x[:, 3:6]  # [B, 3]

        # 第一列：单位化 a1
        norm_a1 = np.linalg.norm(a1, axis=1, keepdims=True)
        # 避免除零
        norm_a1 = np.where(norm_a1 == 0, 1.0, norm_a1)
        b1 = a1 / norm_a1  # [B, 3]

        # 第二列：从 a2 中减去 b1 方向的分量，再单位化
        proj = np.sum(b1 * a2, axis=1, keepdims=True)  # [B, 1]
        ortho_a2 = a2 - proj * b1  # [B, 3]

        norm_ortho = np.linalg.norm(ortho_a2, axis=1, keepdims=True)
        # 如果正交分量为0（退化情况），可设为任意垂直向量，这里简单处理为单位向量
        norm_ortho = np.where(norm_ortho == 0, 1.0, norm_ortho)
        b2 = ortho_a2 / norm_ortho  # [B, 3]

        # 第三列：b1 × b2
        b3 = np.cross(b1, b2, axis=1)  # [B, 3]

        # 拼接成旋转矩阵：每一列为 b1, b2, b3 → shape [B, 3, 3]
        R = np.stack((b1, b2, b3), axis=2)  # [B, 3, 3]
        return R

    @staticmethod
    def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
        """
        将旋转矩阵转换为6D旋转表示
        输入:
            R: [B, 3, 3]
        输出:
            x: [B, 6]
        """
        assert R.shape[-2:] == (3, 3), "输入必须是3x3矩阵"
        # 取前两列并展平：R[:, :, 0] 和 R[:, :, 1] → 拼成 [B, 6]
        return np.concatenate((R[:, :, 0], R[:, :, 1]), axis=1)  # [B, 6]


if __name__ == "__main__":
    import time

    np.random.seed(42)

    for rot_6d, cls in zip(
        [np.random.randn(4, 6), np.random.randn(6)], [BatchedRotation6D, Rotation6D]
    ):
        print("测试类:", cls.__name__)
        print("原始6D向量:\n", rot_6d)

        R = cls.rot6d_to_matrix(rot_6d)
        rot_6d_back = cls.matrix_to_rot6d(R)

        print("\n转换后的旋转矩阵:\n", R)
        print("\n旋转矩阵再转回6D:\n", rot_6d_back)

        R_reconstructed = cls.rot6d_to_matrix(rot_6d_back)
        print("\n重建的旋转矩阵是否一致？")
        assert np.allclose(R, R_reconstructed)

        # 验证：rot_6d_back 应该与 rot_6d 在旋转等价意义下一致（但不一定数值相等）
        # 因为6D表示不是唯一的，但重建的矩阵应相同
        print("\n原始6D与重建6D是否数值相近？")
        assert not np.allclose(rot_6d, rot_6d_back)

        print("\n通过重建的旋转矩阵再转换回6D，是否与原始6D相近？")
        assert np.allclose(rot_6d_back, cls.matrix_to_rot6d(R_reconstructed))
        print("\n测试通过！\n")

        if cls is Rotation6D:
            """
            6D 旋转表示（Rotation 6D）不是唯一的。多个 6D 向量可以映射到同一个旋转矩阵。在测试代码中，原始的 rot_6d 是通过随机生成的，它不一定满足正交归一化条件，而通过旋转矩阵转换回来的 rot_6d_from_quat（或 rot_6d_back）是经过 Gram-Schmidt 正交化处理后的“标准”形式。

            因此，应该将 rot_6d_from_quat 与经过处理后的 rot_6d_back 进行比较，而不是与原始随机的 rot_6d 比较。
            """
            quat = cls.rot6d_to_quat(rot_6d)
            start = time.perf_counter()
            rot_6d_from_quat = cls.quat_to_rot6d(quat)
            print(f"四元数转换耗时: {time.perf_counter() - start:.6f}秒")
            print("转换为四元数:\n", quat)
            print("从四元数转换回6D:\n", rot_6d_from_quat)
            assert np.allclose(rot_6d_back, rot_6d_from_quat)
            print("四元数转换测试通过！\n")
