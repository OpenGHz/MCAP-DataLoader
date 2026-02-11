import numpy as np


class Rotation6D:
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
    np.random.seed(42)

    # 随机生成一个批量的6D向量
    rot_6d = np.random.randn(4, 6)  # 4个样本

    R = Rotation6D.rot6d_to_matrix(rot_6d)
    rot_6d_back = Rotation6D.matrix_to_rot6d(R)

    print("原始6D向量:\n", rot_6d)
    print("\n转换后的旋转矩阵:\n", R)
    print("\n旋转矩阵再转回6D:\n", rot_6d_back)

    R_reconstructed = Rotation6D.rot6d_to_matrix(rot_6d_back)
    print("\n重建的旋转矩阵是否一致？")
    assert np.allclose(R, R_reconstructed)

    # 验证：rot_6d_back 应该与 rot_6d 在旋转等价意义下一致（但不一定数值相等）
    # 因为6D表示不是唯一的，但重建的矩阵应相同
    print("\n原始6D与重建6D是否数值相近？")
    assert not np.allclose(rot_6d, rot_6d_back)

    print("\n通过重建的旋转矩阵再转换回6D，是否与原始6D相近？")
    assert np.allclose(rot_6d_back, Rotation6D.matrix_to_rot6d(R_reconstructed))
