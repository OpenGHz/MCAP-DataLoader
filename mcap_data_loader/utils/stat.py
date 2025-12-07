import numpy as np
from typing import List, Tuple
from collections.abc import Iterable
from numpy.typing import NDArray
from typing import TypedDict, Dict
from collections import defaultdict


class StatisticsBasis(TypedDict):
    n: int
    sum: NDArray[np.floating]
    sum_sq: NDArray[np.floating]


class Statistics(StatisticsBasis):
    mean: NDArray[np.floating]
    std: NDArray[np.floating]


StatGroup = Tuple[int, NDArray[np.floating], NDArray[np.floating]]


def combine_groups_decomposition(groups: Iterable[StatGroup]) -> StatGroup:
    """
    方法1：基于 Σx 和 Σx² 的分解法（支持多维/向量化）
    groups: list of (n, mean_array, sd_array)
    returns: (combined_n, combined_mean_array, combined_sd_array)
    """

    tx = 0  # Σx over all groups
    txx = 0  # Σx² over all groups
    tn = 0

    for n, mean, sd in groups:
        if n < 1:
            raise ValueError("At least one sample is required in each group.")

        # Σx = mean * n
        sigma_x = mean * n
        # Σx² = (n - 1) * SD² + (Σx)² / n
        sigma_x2 = (sd**2) * (n - 1) + (sigma_x**2) / n
        tx += sigma_x
        txx += sigma_x2
        tn += n

    if tn == 0:
        raise ValueError("Group is empty.")

    combined_mean = tx / tn
    combined_sd = (
        np.sqrt(np.maximum((txx - (tx**2) / tn) / (tn - 1), 0.0)) if tn > 1 else 0.0
    )
    return tn, combined_mean, combined_sd


def combine_groups(groups: List[Statistics]) -> Statistics:
    """
    方法1：基于 Σx 和 Σx² 的分解法（支持多维/向量化）
    groups: list of (n, Σx, Σx²)
    returns: (combined_n, combined_mean_array, combined_sd_array)
    """

    tx = 0  # Σx over all groups
    txx = 0  # Σx² over all groups
    tn = 0

    for stat in groups:
        n = stat["n"]
        if n < 1:
            raise ValueError("At least one sample is required in each group.")

        tx += stat["sum"]
        txx += stat["sum_sq"]
        tn += n

    if tn == 0:
        raise ValueError("Group is empty.")

    combined_mean = tx / tn
    combined_sd = (
        np.sqrt(np.maximum((txx - (tx**2) / tn) / (tn - 1), 0.0)) if tn > 1 else 0.0
    )
    return {
        "n": tn,
        "sum": tx,
        "sum_sq": txx,
        "mean": combined_mean,
        "std": combined_sd,
    }


def combine_dict_groups(
    dict_groups: Iterable[Dict[str, Statistics]],
) -> Dict[str, Statistics]:
    """Merge multiple statistics dictionaries by key."""
    groups = defaultdict(list)
    for dict_group in dict_groups:
        for key, stat in dict_group.items():
            groups[key].append(stat)
    return {key: combine_groups(stats) for key, stats in groups.items()}


def combine_two_groups_cochrane(
    n1: int,
    m1: NDArray[np.float64],
    s1: NDArray[np.float64],
    n2: int,
    m2: NDArray[np.float64],
    s2: NDArray[np.float64],
) -> StatGroup:
    """
    合并两个组（Cochrane 公式，向量化）
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Sample sizes must be positive.")
    if m1.shape != m2.shape or s1.shape != s2.shape or m1.shape != s1.shape:
        raise ValueError("Mean and SD arrays must have matching shapes across groups.")

    n_comb = n1 + n2
    mean_comb = (n1 * m1 + n2 * m2) / n_comb

    var1 = (n1 - 1) * (s1**2)
    var2 = (n2 - 1) * (s2**2)
    mean_diff_sq = (m1 - m2) ** 2
    between_var = (n1 * n2 / n_comb) * mean_diff_sq

    combined_variance = (var1 + var2 + between_var) / (n_comb - 1)
    combined_variance = np.maximum(combined_variance, 0.0)  # 避免负方差
    sd_comb = np.sqrt(combined_variance)

    return n_comb, mean_comb, sd_comb


def combine_groups_cochrane_iterative(groups: List[StatGroup]) -> StatGroup:
    """
    方法2：Cochrane 迭代两两合并（向量化）
    """
    if not groups:
        raise ValueError("Group list is empty.")
    if len(groups) == 1:
        n, m, s = groups[0]
        return n, m.copy(), s.copy()

    # Start with first group
    n_comb, m_comb, s_comb = groups[0]
    m_comb = np.array(m_comb, dtype=np.float64)
    s_comb = np.array(s_comb, dtype=np.float64)

    for i in range(1, len(groups)):
        n2, m2, s2 = groups[i]
        m2 = np.array(m2, dtype=np.float64)
        s2 = np.array(s2, dtype=np.float64)
        n_comb, m_comb, s_comb = combine_two_groups_cochrane(
            n_comb, m_comb, s_comb, n2, m2, s2
        )

    return n_comb, m_comb, s_comb


def test_both_methods_vectorized():
    # Example: 3 groups, each with 4 variables (e.g., 4 features)
    np.random.seed(0)
    groups = [
        (10, np.array([11.8, 5.2, 3.1, 9.0]), np.array([2.4, 1.1, 0.9, 2.0])),
        (20, np.array([15.3, 6.7, 4.5, 8.2]), np.array([3.2, 1.5, 1.2, 2.5])),
        (15, np.array([8.4, 4.0, 2.8, 7.5]), np.array([4.1, 2.0, 1.8, 3.0])),
    ]

    result1 = combine_groups_decomposition(groups)
    result2 = combine_groups_cochrane_iterative(groups)

    print("Method 1 (Decomposition):")
    print(f"  n={result1[0]}")
    print(f"  mean={result1[1]}")
    print(f"  sd={result1[2]}")

    print("\nMethod 2 (Cochrane Iterative):")
    print(f"  n={result2[0]}")
    print(f"  mean={result2[1]}")
    print(f"  sd={result2[2]}")

    # Check consistency
    tol = 1e-10
    assert result1[0] == result2[0], "n mismatch"
    assert np.allclose(result1[1], result2[1], atol=tol), "mean mismatch"
    assert np.allclose(result1[2], result2[2], atol=tol), "sd mismatch"

    print("\n✅ Both methods produce identical results (vectorized)!")


if __name__ == "__main__":
    test_both_methods_vectorized()
