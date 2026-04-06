import numpy as np

# tab10 palette (matches matplotlib's tab10)
_TAB10 = np.array(
    [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
    ],
    dtype=np.uint8,
)


def _channel_palette(n: int) -> np.ndarray:
    """Return a (n, 3) uint8 palette with one distinct color per channel."""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.array([_TAB10[i % len(_TAB10)] for i in range(n)], dtype=np.uint8)


def make_heatmap_rgb(heat_map: np.ndarray) -> np.ndarray:
    """Each one-hot channel gets a unique color; overlapping pixels are averaged."""
    heat = np.asarray(heat_map, dtype=np.float32)  # (H, W, C)
    if heat.ndim != 3:
        raise TypeError(f"Expected 3D heat map, got shape {heat.shape}")

    H, W, C = heat.shape
    if C == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    palette = _channel_palette(C).astype(np.float32)  # (C, 3)
    rgb = heat.reshape(-1, C) @ palette  # (H*W, 3)
    active = heat.sum(axis=-1).reshape(-1, 1).clip(1.0)
    rgb /= active
    return np.clip(rgb.reshape(H, W, 3), 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import cv2

    # Example usage
    heat_map = np.zeros((100, 100, 5), dtype=np.float32)
    heat_map[20:50, 20:50, 0] = 1.0  # Red square
    heat_map[40:70, 40:70, 1] = 1.0  # Green square overlapping red
    heat_map[60:90, 60:90, 2] = 1.0  # Blue square overlapping green
    rgb_image = make_heatmap_rgb(heat_map)
    print(rgb_image.shape)  # Should be (100, 100, 3)

    cv2.imshow("Heatmap RGB", rgb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
