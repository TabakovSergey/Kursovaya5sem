from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_image_paths(root: Path) -> Iterable[Path]:
    """
    Обходит директорию рекурсивно и возвращает пути к изображениям.
    """
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def extract_image_features(image_path: Path, bins_per_channel: int = 16) -> np.ndarray:
    """
    Преобразует изображение в компактный набор числовых признаков.

    Используем гистограммы по каждому цветовому каналу RGB, а также
    средние/стандартные отклонения цветов и геометрические характеристики.
    """
    if bins_per_channel <= 0:
        raise ValueError("Количество бинoв гистограммы должно быть > 0")

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        rgb_array = np.asarray(img, dtype=np.uint8)

    hist_features = []
    for channel_idx in range(3):
        channel = rgb_array[:, :, channel_idx]
        hist, _ = np.histogram(channel, bins=bins_per_channel, range=(0, 255))
        hist = hist.astype(np.float32)
        hist_sum = float(hist.sum())
        if hist_sum > 0:
            hist /= hist_sum
        hist_features.append(hist)

    flat_pixels = rgb_array.reshape(-1, 3).astype(np.float32) / 255.0
    mean_color = flat_pixels.mean(axis=0)
    std_color = flat_pixels.std(axis=0)

    height, width = rgb_array.shape[:2]
    size_features = np.array(
        [float(width), float(height), float(width) / max(float(height), 1.0)],
        dtype=np.float32,
    )

    feature_vector = np.concatenate(
        [*hist_features, mean_color, std_color, size_features]
    ).astype(np.float32)
    return feature_vector
