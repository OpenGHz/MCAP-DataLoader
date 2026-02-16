import numpy as np
from mcap_data_loader.utils.transformations import (
    quaternion_matrix,
    quaternion_multiply,
    quaternion_inverse,
)
from functools import wraps


def quat_conjugate(q):
    """共轭（对于单位四元数 = 逆）"""
    # q = [x, y, z, w]
    return np.concatenate([-q[:, :3], q[:, 3:]], axis=1)


def quat_mul(q1, q2):
    """四元数乘法: q = q1 * q2 (scalar-last)"""
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([x, y, z, w], axis=1)


def rotate_vector_by_quat(v, q):
    """
    用单位四元数 q 旋转三维向量 v
    v: [N, 3], q: [N, 4] (scalar-last)
    """
    # 将 v 转为纯四元数 [x, y, z, 0]
    q_v = np.concatenate([v, np.zeros((v.shape[0], 1))], axis=1)
    q_conj = quat_conjugate(q)
    # q * v * q^{-1}
    return quat_mul(quat_mul(q, q_v), q_conj)[:, :3]


def quat_to_rotation_matrix(q):
    """
    将 [x, y, z, w] 四元数转为旋转矩阵
    q: [N, 4]
    返回: [N, 3, 3]
    """
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x2, y2, z2 = x * 2, y * 2, z * 2
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2

    R = np.stack(
        [
            np.stack([1 - (yy + zz), xy - wz, xz + wy], axis=1),
            np.stack([xy + wz, 1 - (xx + zz), yz - wx], axis=1),
            np.stack([xz - wy, yz + wx, 1 - (xx + yy)], axis=1),
        ],
        axis=1,
    )  # [N, 3, 3]
    return R


def to_relative_pose(ref_pos, ref_quat, pos, quat):
    """
    输入:
        ref_pos: [3,] in xyz
        ref_quat: [4,] in [x,y,z,w] (unit quat, scalar-last)
        pos: [3,] in xyz
        quat: [4,] in [x,y,z,w] (unit quat, scalar-last)
    输出:
        rel_pos: [3,] —— 相对位置（在 ref 坐标系下）
        rel_quat: [4,] —— 相对四元数（q_rel = q_ref^{-1} * q）
        rel_rot6d: [6,] —— 相对6D旋转表示（R_rel 前两列按行展开）
    """
    ref_pos = np.asarray(ref_pos)
    ref_quat = np.asarray(ref_quat)
    pos = np.asarray(pos)
    quat = np.asarray(quat)

    assert ref_pos.shape == (3,)
    assert pos.shape == (3,)
    assert ref_quat.shape == (4,)
    assert quat.shape == (4,)

    # q_ref_inv = conjugate(q_ref) for unit quaternion
    qrx, qry, qrz, qrw = ref_quat
    iqx, iqy, iqz, iqw = -qrx, -qry, -qrz, qrw

    # rel_pos = rotate(delta, q_ref_inv) using fast quat-vector rotation
    dx, dy, dz = pos - ref_pos
    # t = 2 * cross(q_vec, v)
    tx = 2.0 * (iqy * dz - iqz * dy)
    ty = 2.0 * (iqz * dx - iqx * dz)
    tz = 2.0 * (iqx * dy - iqy * dx)
    # v' = v + w*t + cross(q_vec, t)
    rel_x = dx + iqw * tx + (iqy * tz - iqz * ty)
    rel_y = dy + iqw * ty + (iqz * tx - iqx * tz)
    rel_z = dz + iqw * tz + (iqx * ty - iqy * tx)
    rel_pos = np.array([rel_x, rel_y, rel_z], dtype=pos.dtype)

    # rel_quat = q_ref_inv * quat (scalar-last)
    qx, qy, qz, qw = quat
    rw = iqw * qw - iqx * qx - iqy * qy - iqz * qz
    rx = iqw * qx + iqx * qw + iqy * qz - iqz * qy
    ry = iqw * qy - iqx * qz + iqy * qw + iqz * qx
    rz = iqw * qz + iqx * qy - iqy * qx + iqz * qw
    rel_quat = np.array([rx, ry, rz, rw], dtype=quat.dtype)

    # rel_rot6d from quaternion -> first two columns of R, then flatten by rows:
    # [r00,r01,r10,r11,r20,r21]
    x, y, z, w = rx, ry, rz, rw
    x2, y2, z2 = x + x, y + y, z + z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2

    r00 = 1.0 - (yy + zz)
    r01 = xy - wz
    r10 = xy + wz
    r11 = 1.0 - (xx + zz)
    r20 = xz - wy
    r21 = yz + wx
    rel_rot6d = np.array([r00, r01, r10, r11, r20, r21], dtype=rel_quat.dtype)

    return rel_pos, rel_quat, rel_rot6d


def to_absolute_pose(ref_pos, ref_quat, rel_pos, rel_quat):
    """
    输入:
        ref_pos: [3,] in xyz
        ref_quat: [4,] in [x,y,z,w] (unit quat, scalar-last)
        rel_pos: [3,] in xyz (relative position in ref frame)
        rel_quat: [4,] in [x,y,z,w] (relative rotation)
    输出:
        abs_pos: [3,] in xyz
        abs_quat: [4,] in [x,y,z,w]
    """
    ref_pos = np.asarray(ref_pos)
    ref_quat = np.asarray(ref_quat)
    rel_pos = np.asarray(rel_pos)
    rel_quat = np.asarray(rel_quat)

    assert ref_pos.shape == (3,)
    assert rel_pos.shape == (3,)
    assert ref_quat.shape == (4,)
    assert rel_quat.shape == (4,)

    # abs_quat = q_ref * q_rel
    qrx, qry, qrz, qrw = ref_quat
    rqx, rqy, rqz, rqw = rel_quat

    rw = qrw * rqw - qrx * rqx - qry * rqy - qrz * rqz
    rx = qrw * rqx + qrx * rqw + qry * rqz - qrz * rqy
    ry = qrw * rqy - qrx * rqz + qry * rqw + qrz * rqx
    rz = qrw * rqz + qrx * rqy - qry * rqx + qrz * rqw
    abs_quat = np.array([rx, ry, rz, rw], dtype=rel_quat.dtype)

    # abs_pos = R_ref @ rel_pos + t_ref
    R_ref = quaternion_matrix(ref_quat)[:3, :3]  # [3, 3]
    abs_pos = R_ref @ rel_pos + ref_pos  # [3]

    return abs_pos, abs_quat


def to_relative_pose_serial(pos_serial, quat_serial):
    """
    输入:
        pos_serial: [N, 3]
        quat_serial: [N, 4] in [x, y, z, w] format
    输出:
        rel_pos: [N, 3] —— 相对位置
        rel_quat: [N, 4] —— 相对四元数
        rel_rot6d: [N, 6] —— 相对6D旋转表示
    """
    N = pos_serial.shape[0]
    assert quat_serial.shape == (N, 4)

    # 第0帧作为参考
    ref_pos = pos_serial[0:1]  # [1, 3]
    ref_quat = quat_serial[0:1]  # [1, 4]

    # 相对位置：先平移，再旋转到参考系
    delta_pos = pos_serial - ref_pos  # [N, 3]
    rel_pos = rotate_vector_by_quat(delta_pos, quat_conjugate(ref_quat))  # [N, 3]

    # 相对四元数：q_rel = q_ref^{-1} * q_i
    ref_quat_inv = quat_conjugate(ref_quat)  # [1, 4]
    # 广播：ref_quat_inv 与每个 quat_serial[i] 相乘
    rel_quat = quat_mul(
        np.tile(ref_quat_inv, (N, 1)),  # [N, 4]
        quat_serial,  # [N, 4]
    )  # [N, 4]

    # 转换为6D表示：取旋转矩阵前两列
    # 先将相对四元数转为旋转矩阵
    R = quat_to_rotation_matrix(rel_quat)  # [N, 3, 3]
    rel_rot6d = R[:, :, :2].reshape(N, 6)  # [N, 6]

    return rel_pos, rel_quat, rel_rot6d


# 6D → 旋转矩阵（Gram-Schmidt 正交化）
def rot6d_to_matrix(x):
    assert x.shape[-1] == 6
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    proj = np.sum(b1 * a2, axis=1, keepdims=True)
    b2 = a2 - proj * b1
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True)
    b3 = np.cross(b1, b2, axis=1)
    return np.stack((b1, b2, b3), axis=2)  # [B, 3, 3]


def rel_quat_to_abs(pos_0, quat_0, rel_pos, rel_quat):
    """
    输入:
        pos_0: [3,] 或 [1,3]
        quat_0: [4,] 或 [1,4] in [x,y,z,w]
        rel_pos: [N, 3]
        rel_quat: [N, 4]
    输出:
        abs_pos: [N, 3]
        abs_quat: [N, 4]
    """
    N = rel_pos.shape[0]
    pos_0 = np.asarray(pos_0).reshape(1, 3)
    quat_0 = np.asarray(quat_0).reshape(1, 4)

    # 绝对四元数: q_abs = q_0 * q_rel
    abs_quat = quat_mul(
        np.tile(quat_0, (N, 1)),  # [N, 4]
        rel_quat,  # [N, 4]
    )

    # 绝对位置: p_abs = R_0 @ p_rel + t_0
    R0 = quat_to_rotation_matrix(quat_0)  # [1, 3, 3]
    abs_pos = (R0 @ rel_pos[..., None]).squeeze(-1) + pos_0  # [N, 3]

    return abs_pos, abs_quat


# 旋转矩阵 → 四元数（可选，用于验证）
def rotation_matrix_to_quat(R):
    # 使用 Shoemake 方法（数值稳定）
    Qxx, Qxy, Qxz = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    Qyx, Qyy, Qyz = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    Qzx, Qzy, Qzz = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    K = np.stack(
        [
            [Qxx - Qyy - Qzz, 0.0, 0.0, 0.0],
            [Qyx + Qxy, Qyy - Qxx - Qzz, 0.0, 0.0],
            [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0.0],
            [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz],
        ],
        axis=1,
    )
    K = K / 4.0  # 因为 trace(R) = 4w² - 1 等
    q_vec = np.linalg.eigh(K)[1][:, :, -1]  # 最大特征值对应的特征向量
    return q_vec  # [N, 4], [x,y,z,w]


def rel_rot6d_to_abs(pos_0, quat_0, rel_pos, rel_rot6d):
    """
    输入:
        pos_0: [3,]
        quat_0: [4,] in [x,y,z,w]
        rel_pos: [N, 3]
        rel_rot6d: [N, 6]
    输出:
        abs_pos: [N, 3]
        abs_quat: [N, 4]  （可选，也可只输出旋转矩阵）
    """
    pos_0 = np.asarray(pos_0).reshape(1, 3)
    quat_0 = np.asarray(quat_0).reshape(1, 4)

    # 1. 将 6D 转为旋转矩阵
    R_rel = rot6d_to_matrix(rel_rot6d)  # [N, 3, 3]

    # 2. 绝对旋转矩阵: R_abs = R_0 @ R_rel
    R0 = quat_to_rotation_matrix(quat_0)  # [1, 3, 3]
    R_abs = R0 @ R_rel  # [N, 3, 3]

    # 3. 绝对位置: p_abs = R_0 @ p_rel + t_0
    abs_pos = (R0 @ rel_pos[..., None]).squeeze(-1) + pos_0  # [N, 3]

    # 4. （可选）转为四元数
    abs_quat = rotation_matrix_to_quat(R_abs)  # [N, 4]

    return abs_pos, abs_quat


class RelaAbsBasis:
    def __init__(self, tolist: bool = False):
        self.tolist = tolist
        self.to_relative = self._to_list_wrapper(self.to_relative)
        self.to_absolute = self._to_list_wrapper(self.to_absolute)

    def set_ref(self, pose):
        pos, quat = pose
        self.ref_pos = np.asarray(pos)
        self.ref_quat = np.asarray(quat)
        # assert self.ref_pos.shape == (3,)
        # assert self.ref_quat.shape == (4,)

    def _to_list_wrapper(self, func):
        if self.tolist:

            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return tuple(r.tolist() for r in result)

            return wrapper
        else:
            return func

    def to_relative(self, pose):
        raise NotImplementedError

    def to_absolute(self, pose):
        raise NotImplementedError


class PoseLocalRelaAbs(RelaAbsBasis):
    def to_relative(self, pose):
        pos, quat = pose
        return to_relative_pose(self.ref_pos, self.ref_quat, pos, quat)

    def to_absolute(self, pose):
        rel_pos, rel_quat = pose
        return to_absolute_pose(self.ref_pos, self.ref_quat, rel_pos, rel_quat)


class PoseGlobalRelaAbs(RelaAbsBasis):
    def to_relative(self, pose):
        pos, quat = pose
        return pos - self.ref_pos, quaternion_multiply(
            quat, quaternion_inverse(self.ref_quat)
        )

    def to_absolute(self, pose):
        rel_pos, rel_quat = pose
        return self.ref_pos + rel_pos, quaternion_multiply(rel_quat, self.ref_quat)


class VectorRelaAbs:
    def __init__(self, tolist: bool = False):
        self.tolist = tolist

    def set_ref(self, vec):
        self.ref_vec = np.asarray(vec)

    def to_relative(self, vec):
        rela = np.asarray(vec) - self.ref_vec
        if self.tolist:
            return rela.tolist()
        return rela

    def to_absolute(self, rel_vec):
        abs_vec = np.asarray(rel_vec) + self.ref_vec
        if self.tolist:
            return abs_vec.tolist()
        return abs_vec


if __name__ == "__main__":
    # 原始绝对序列
    N = 4
    # np.random.seed(42)  # 固定随机种子
    pos_serial = np.random.rand(N, 3) * 10  # 随机生成位置 [N, 3]
    quat_serial = np.random.rand(N, 4)
    quat_serial /= np.linalg.norm(quat_serial, axis=1, keepdims=True)  # 单位化四元数

    # Step 1: 转为相对表示
    rel_pos, rel_quat, rel_rot6d = to_relative_pose_serial(pos_serial, quat_serial)
    # Step 2: 从相对 + 第0帧恢复绝对
    ref_pos = pos_serial[0]
    ref_quat = quat_serial[0]
    abs_pos_rec, abs_quat_rec = rel_quat_to_abs(ref_pos, ref_quat, rel_pos, rel_quat)

    print("原始位置:\n", pos_serial)
    print("恢复位置:\n", abs_pos_rec)
    assert np.allclose(pos_serial, abs_pos_rec), "位置恢复不准确！"
    print("位置误差:", np.max(np.abs(pos_serial - abs_pos_rec)))

    print("原始四元数:\n", quat_serial)
    print("恢复四元数:\n", abs_quat_rec)
    # 四元数可能符号相反（q 和 -q 表示相同旋转）
    err1 = np.max(np.abs(quat_serial - abs_quat_rec))
    err2 = np.max(np.abs(quat_serial + abs_quat_rec))
    print("四元数误差（考虑符号）:", min(err1, err2))
    assert min(err1, err2) < 1e-6, "四元数恢复不准确！"

    rela_pos, rela_quat, rela_rot6d = to_relative_pose(
        pos_serial[0], quat_serial[0], pos_serial[1], quat_serial[1]
    )
    assert np.allclose(rel_pos[1], rela_pos), (
        "to_relative_pose 与 to_relative_pose_serial 结果不一致！"
    )
    assert np.allclose(rel_quat[1], rela_quat), (
        "to_relative_pose 与 to_relative_pose_serial 结果不一致！"
    )
    assert np.allclose(rel_rot6d[1], rela_rot6d), (
        "to_relative_pose 与 to_relative_pose_serial 结果不一致！"
    )

    pose = (pos_serial[0], quat_serial[0])
    local_rela_abs = PoseLocalRelaAbs()
    local_rela_abs.set_ref(pose)
    rel_pos, rel_quat, _ = local_rela_abs.to_relative((pos_serial[1], quat_serial[1]))
    abs_pose = local_rela_abs.to_absolute((rel_pos, rel_quat))
    assert np.allclose(pos_serial[1], abs_pose[0]), "PoseLocalRelaAbs 恢复位置不准确！"
    assert np.allclose(quat_serial[1], abs_pose[1]) or np.allclose(
        quat_serial[1], -abs_pose[1]
    ), "PoseLocalRelaAbs 恢复四元数不准确！"
    global_rela_abs = PoseGlobalRelaAbs()
    global_rela_abs.set_ref(pose)
    rel_pose_global = global_rela_abs.to_relative((pos_serial[1], quat_serial[1]))
    abs_pose_global = global_rela_abs.to_absolute(rel_pose_global)
    assert np.allclose(pos_serial[1], abs_pose_global[0]), (
        "PoseGlobalRelaAbs 恢复位置不准确！"
    )
    assert np.allclose(quat_serial[1], abs_pose_global[1]) or np.allclose(
        quat_serial[1], -abs_pose_global[1]
    ), "PoseGlobalRelaAbs 恢复四元数不准确！"

    print("所有测试通过！")
