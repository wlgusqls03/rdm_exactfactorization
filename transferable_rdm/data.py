from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import ExperimentConfig
from .systems import SystemRecord
from .utils import print_block


_POTENTIAL_LAPLACIAN_CACHE: dict[int, np.ndarray] = {}


@dataclass
class DatasetSplit:
    """system-level split.

    현재 목표는 point-level validation이 아니라,
    "새 시스템에 대한 transferability"를 보는 것이다.
    """

    train_systems: list[SystemRecord]
    val_systems: list[SystemRecord]
    test_systems: list[SystemRecord] = field(default_factory=list)


@dataclass
class PairBatch:
    """한 시스템에서 샘플한 pair batch."""

    system: SystemRecord
    left_idx: np.ndarray
    right_idx: np.ndarray
    point_feat_r: np.ndarray
    point_feat_rp: np.ndarray
    pair_feat: np.ndarray
    gamma_true: np.ndarray
    weights: np.ndarray
    global_context: np.ndarray


def compact_system_ids(systems: list[SystemRecord], max_ids: int = 12) -> list[str]:
    """Keep split logging readable for 1000-system QMugs subsets."""
    ids = [system.system_id for system in systems]
    if len(ids) <= max_ids:
        return ids
    head = max_ids // 2
    tail = max_ids - head
    return ids[:head] + [f"... ({len(ids) - max_ids} more) ..."] + ids[-tail:]


def split_systems(systems: list[SystemRecord], config: ExperimentConfig) -> DatasetSplit:
    """시스템 단위로 train / held-out validation 분리."""
    if len(systems) < 2:
        raise RuntimeError("At least two systems are required for held-out system validation.")

    rng = np.random.default_rng(config.seed)
    indices = np.arange(len(systems))
    rng.shuffle(indices)

    exact_counts_requested = (
        config.train_system_count > 0
        or config.val_system_count > 0
        or config.test_system_count > 0
    )
    if exact_counts_requested:
        if config.train_system_count <= 0 or config.val_system_count <= 0:
            raise RuntimeError(
                "Exact split mode requires RDM_TRAIN_SYSTEM_COUNT and RDM_VAL_SYSTEM_COUNT "
                "to both be positive."
            )
        n_train = config.train_system_count
        n_val = config.val_system_count
        n_test = max(config.test_system_count, 0)
        n_required = n_train + n_val + n_test
        if len(systems) < n_required:
            raise RuntimeError(
                f"Requested {n_required} systems for exact split "
                f"({n_train} train / {n_val} val / {n_test} test), "
                f"but only {len(systems)} systems are available."
            )
        selected = indices[:n_required]
        train_idx = selected[:n_train]
        val_idx = selected[n_train : n_train + n_val]
        test_idx = selected[n_train + n_val :]
    else:
        n_train = int(round(len(systems) * config.train_system_fraction))
        n_train = max(1, min(len(systems) - 1, n_train))

        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        test_idx = np.array([], dtype=np.int64)

    split = DatasetSplit(
        train_systems=[systems[i] for i in train_idx],
        val_systems=[systems[i] for i in val_idx],
        test_systems=[systems[i] for i in test_idx],
    )
    print_block(
        "System split",
        [
            ("train systems", len(split.train_systems)),
            ("val systems", len(split.val_systems)),
            ("test systems", len(split.test_systems)),
            ("train ids", compact_system_ids(split.train_systems)),
            ("val ids", compact_system_ids(split.val_systems)),
            ("test ids", compact_system_ids(split.test_systems)),
        ],
    )
    return split


def curriculum_probs(epoch: int, total_epochs: int) -> dict[str, float]:
    """초반엔 diagonal / near-diagonal을 더 보고, 후반엔 off-diagonal 비중을 늘린다."""
    t = 0.0 if total_epochs <= 1 else epoch / float(total_epochs - 1)
    start = np.array([0.55, 0.25, 0.15, 0.05], dtype=np.float64)
    end = np.array([0.15, 0.20, 0.25, 0.40], dtype=np.float64)
    probs = (1.0 - t) * start + t * end
    probs /= probs.sum()
    return {"diag": float(probs[0]), "near": float(probs[1]), "mid": float(probs[2]), "far": float(probs[3])}


def build_pair_features(system: SystemRecord, left_idx: np.ndarray, right_idx: np.ndarray) -> np.ndarray:
    """pair-level symmetric / antisymmetric descriptor.

    Shapes
    ------
    left_idx, right_idx : (batch,)
    return              : (batch, d_pair)
    """
    points_r = system.points[left_idx]     # (batch, 3)
    points_rp = system.points[right_idx]   # (batch, 3)

    midpoint = 0.5 * (points_r + points_rp)                         # (batch, 3)
    separation = points_r - points_rp                               # (batch, 3)
    abs_separation = np.abs(separation)                             # (batch, 3)
    sep_sq_components = separation**2                               # (batch, 3)
    sep_norm = np.linalg.norm(separation, axis=1, keepdims=True)    # (batch, 1)
    sep_sq = np.sum(separation**2, axis=1, keepdims=True)           # (batch, 1)

    pot_r = system.potential[left_idx]                              # (batch, 1)
    pot_rp = system.potential[right_idx]                            # (batch, 1)
    grad_r = system.grad_potential[left_idx]                        # (batch, 3)
    grad_rp = system.grad_potential[right_idx]                      # (batch, 3)

    domain_scale = max(float(np.max(np.abs(system.axis))), 1e-6)
    step_scale = max(float(system.step), 1e-6)
    pot_scale = max(float(np.std(system.potential)), 1.0)

    pair_features = np.concatenate(
        [
            midpoint / domain_scale,
            abs_separation / domain_scale,
            sep_sq_components / (domain_scale * domain_scale),
            sep_norm / domain_scale,
            sep_sq / (domain_scale * domain_scale),
            0.5 * (pot_r + pot_rp) / pot_scale,
            0.5 * (grad_r + grad_rp) * step_scale / pot_scale,
            np.abs(grad_r - grad_rp) * step_scale / pot_scale,
        ]
        + potential_laplacian_pair_features(system, left_idx, right_idx, step_scale, pot_scale),
        axis=1,
    )
    return pair_features.astype(np.float32)


def signed_log_scaled_np(values: np.ndarray, clip: float) -> np.ndarray:
    clip = max(float(clip), 1.0)
    clipped = np.clip(values, -clip, clip)
    return np.sign(clipped) * np.log1p(np.abs(clipped)) / np.log1p(clip)


def richardson_laplacian_np(grid_values: np.ndarray, n_axis: int, h: float) -> np.ndarray:
    vol = np.reshape(grid_values, (n_axis, n_axis, n_axis))
    padded = np.pad(vol, ((2, 2), (2, 2), (2, 2)), mode="symmetric")
    center = padded[2 : n_axis + 2, 2 : n_axis + 2, 2 : n_axis + 2]

    def second_derivative(axis: int) -> np.ndarray:
        slices = [slice(2, n_axis + 2), slice(2, n_axis + 2), slice(2, n_axis + 2)]
        values = []
        for offset in (0, 1, 3, 4):
            shifted = list(slices)
            shifted[axis] = slice(offset, offset + n_axis)
            values.append(padded[tuple(shifted)])
        return (-values[0] + 16.0 * values[1] - 30.0 * center + 16.0 * values[2] - values[3]) / (
            12.0 * h * h
        )

    return np.reshape(second_derivative(0) + second_derivative(1) + second_derivative(2), (-1, 1))


def potential_laplacian_descriptor(system: SystemRecord) -> np.ndarray:
    cached = _POTENTIAL_LAPLACIAN_CACHE.get(id(system))
    if cached is not None:
        return cached
    n_axis = len(system.axis)
    lap = richardson_laplacian_np(system.potential, n_axis, system.step)
    pot_scale = max(float(np.std(system.potential)), 1.0)
    descriptor = signed_log_scaled_np(
        lap * (system.step**2) / pot_scale,
        system.metadata.get("potential_laplacian_clip", 8.0),
    ).astype(np.float32)
    _POTENTIAL_LAPLACIAN_CACHE[id(system)] = descriptor
    return descriptor


def potential_laplacian_pair_features(
    system: SystemRecord,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    step_scale: float,
    pot_scale: float,
) -> list[np.ndarray]:
    if not bool(system.metadata.get("use_potential_laplacian_feature", True)):
        return []
    lap = potential_laplacian_descriptor(system)
    lap_r = lap[left_idx]
    lap_rp = lap[right_idx]
    return [0.5 * (lap_r + lap_rp), np.abs(lap_r - lap_rp)]


def choose_system(train_systems: list[SystemRecord], rng: np.random.Generator) -> SystemRecord:
    """시스템별로 균등하게 뽑는다.

    pair 수가 많은 시스템이 과도하게 지배하는 것을 막기 위해,
    system-balanced sampling을 사용한다.
    """
    idx = int(rng.integers(0, len(train_systems)))
    return train_systems[idx]


def sample_pair_indices(
    system: SystemRecord,
    batch_size: int,
    epoch: int,
    total_epochs: int,
    rng: np.random.Generator,
    category_probs: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """category-balanced curriculum sampling."""
    category_names = ["diag", "near", "mid", "far"]
    if category_probs is None:
        probs = curriculum_probs(epoch, total_epochs)
        prob_values = [probs[name] for name in category_names]
    else:
        prob_array = np.asarray(category_probs, dtype=np.float64)
        if prob_array.shape != (4,):
            raise ValueError("category_probs must contain four values: diag,near,mid,far.")
        if np.any(prob_array < 0.0) or float(np.sum(prob_array)) <= 0.0:
            raise ValueError("category_probs must be non-negative and have positive sum.")
        prob_values = (prob_array / float(np.sum(prob_array))).tolist()
    counts = rng.multinomial(batch_size, prob_values)

    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    category_parts: list[np.ndarray] = []
    for category_id, (name, count) in enumerate(zip(category_names, counts)):
        if count == 0:
            continue
        left, right = sample_pairs_by_category(system, name, int(count), rng)
        left_parts.append(left)
        right_parts.append(right)
        category_parts.append(np.full(len(left), category_id, dtype=np.int64))

    if not left_parts:
        left, right = sample_pairs_by_category(system, "far", batch_size, rng)
        return left, right, np.full(batch_size, 3, dtype=np.int64)

    left_idx = np.concatenate(left_parts, axis=0)
    right_idx = np.concatenate(right_parts, axis=0)
    categories = np.concatenate(category_parts, axis=0)
    order = rng.permutation(len(left_idx))
    return left_idx[order].astype(np.int64), right_idx[order].astype(np.int64), categories[order].astype(np.int64)


def ravel_coords(coords: np.ndarray, axis_points: int) -> np.ndarray:
    return (
        coords[:, 0] * axis_points * axis_points
        + coords[:, 1] * axis_points
        + coords[:, 2]
    ).astype(np.int64)


def random_coords(axis_points: int, count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, axis_points, size=(count, 3), endpoint=False, dtype=np.int64)


def offset_table(category: str) -> np.ndarray:
    offsets = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            for dz in range(-2, 3):
                norm_sq = dx * dx + dy * dy + dz * dz
                if norm_sq == 0:
                    continue
                if category == "near" and norm_sq == 1:
                    offsets.append((dx, dy, dz))
                elif category == "mid" and 1 < norm_sq <= 4:
                    offsets.append((dx, dy, dz))
    return np.asarray(offsets, dtype=np.int64)


def sample_offset_pairs(
    axis_points: int,
    offsets: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    offset_idx = rng.integers(0, len(offsets), size=count, endpoint=False)
    chosen_offsets = offsets[offset_idx]
    left_coords = np.empty((count, 3), dtype=np.int64)
    for dim in range(3):
        lo = np.maximum(0, -chosen_offsets[:, dim])
        hi = np.minimum(axis_points, axis_points - chosen_offsets[:, dim])
        left_coords[:, dim] = np.floor(rng.random(count) * (hi - lo) + lo).astype(np.int64)
    right_coords = left_coords + chosen_offsets
    return ravel_coords(left_coords, axis_points), ravel_coords(right_coords, axis_points)


def sample_far_pairs(system: SystemRecord, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    axis_points = len(system.axis)
    left_all: list[np.ndarray] = []
    right_all: list[np.ndarray] = []
    need = count
    while need > 0:
        proposal = max(need * 2, 32)
        left_coords = random_coords(axis_points, proposal, rng)
        right_coords = random_coords(axis_points, proposal, rng)
        left = ravel_coords(left_coords, axis_points)
        right = ravel_coords(right_coords, axis_points)
        dist = np.linalg.norm(system.points[left] - system.points[right], axis=1)
        keep = dist > (2.0 * system.step + 1e-7)
        if np.any(keep):
            left_keep = left[keep][:need]
            right_keep = right[keep][:need]
            left_all.append(left_keep)
            right_all.append(right_keep)
            need -= len(left_keep)
    return np.concatenate(left_all), np.concatenate(right_all)


def sample_pairs_by_category(
    system: SystemRecord,
    category: str,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    axis_points = len(system.axis)
    if category == "diag":
        left = rng.integers(0, len(system.points), size=count, endpoint=False, dtype=np.int64)
        return left, left.copy()
    if category in {"near", "mid"}:
        return sample_offset_pairs(axis_points, offset_table(category), count, rng)
    return sample_far_pairs(system, count, rng)


def pair_weights_from_categories(
    categories: np.ndarray,
    category_weights: Sequence[float] | None = None,
) -> np.ndarray:
    base = (
        np.array([20.0, 8.0, 4.0, 1.0], dtype=np.float32)
        if category_weights is None
        else np.asarray(category_weights, dtype=np.float32)
    )
    if base.shape != (4,):
        raise ValueError("category_weights must contain four values: diag,near,mid,far.")
    if np.any(base < 0.0) or float(np.sum(base)) <= 0.0:
        raise ValueError("category_weights must be non-negative and have positive sum.")
    weights = base[categories].reshape(-1, 1)
    return (weights / max(float(np.mean(weights)), 1e-8)).astype(np.float32)


def sample_pair_batch(
    system: SystemRecord,
    config: ExperimentConfig,
    epoch: int,
    rng: np.random.Generator,
) -> PairBatch:
    """한 시스템에서 training pair batch 구성."""
    left_idx, right_idx, categories = sample_pair_indices(system, config.batch_size, epoch, config.epochs, rng)

    return PairBatch(
        system=system,
        left_idx=left_idx.astype(np.int64),
        right_idx=right_idx.astype(np.int64),
        point_feat_r=system.local_features[left_idx].astype(np.float32),
        point_feat_rp=system.local_features[right_idx].astype(np.float32),
        pair_feat=build_pair_features(system, left_idx, right_idx).astype(np.float32),
        gamma_true=system.gamma_values(left_idx, right_idx).astype(np.float32),
        weights=pair_weights_from_categories(categories),
        global_context=system.global_context.astype(np.float32),
    )


def full_pair_chunks(system: SystemRecord, chunk_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """full pair 예측 시 memory를 줄이기 위한 chunk index list."""
    n_points = len(system.points)
    total = n_points * n_points
    chunks = []
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        flat = np.arange(start, stop, dtype=np.int64)
        chunks.append((flat // n_points, flat % n_points))
    return chunks
