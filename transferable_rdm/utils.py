from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int) -> None:
    """NumPy / Python / TensorFlow seed를 동시에 고정."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def print_block(title: str, rows: Iterable[tuple[str, object]]) -> None:
    print(f"\n[{title}]")
    print("-" * len(title))
    for key, value in rows:
        print(f"{key:<28}: {value}")


def make_uniform_grid(axis: np.ndarray) -> np.ndarray:
    """uniform 3D Cartesian grid.

    Parameters
    ----------
    axis : (n_axis,)

    Returns
    -------
    points : (n_axis^3, 3)
    """
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)


def flat_index(i: int, j: int, k: int, n_axis: int) -> int:
    """(i, j, k) -> flattened point index."""
    return i * n_axis * n_axis + j * n_axis + k


def topk_descending(values: np.ndarray, k: int = 6) -> np.ndarray:
    arr = np.asarray(values)
    return np.sort(arr)[::-1][:k]


def save_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
