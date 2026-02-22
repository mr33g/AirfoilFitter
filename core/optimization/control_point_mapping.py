from __future__ import annotations

import numpy as np


class OptimizationLayout:
    """Variable layout helper for coupled upper/lower surface optimization."""

    def __init__(self, num_cp_upper: int, num_cp_lower: int):
        self.num_cp_upper = int(num_cp_upper)
        self.num_cp_lower = int(num_cp_lower)
        self.upper_tail_offset = 6 + max(0, 2 * (self.num_cp_upper - 3))
        self.num_vars = int(
            6 + max(0, 2 * (self.num_cp_upper - 3)) + max(0, 2 * (self.num_cp_lower - 3))
        )

    def var_index(self, is_upper: bool, cp_idx: int, coord: int) -> int | None:
        if is_upper:
            if cp_idx == 1:
                return 0 if coord == 1 else None
            if cp_idx == 2:
                return 2 + coord
            if cp_idx >= 3:
                return 6 + 2 * (cp_idx - 3) + coord
            return None
        if cp_idx == 1:
            return 1 if coord == 1 else None
        if cp_idx == 2:
            return 4 + coord
        if cp_idx >= 3:
            return self.upper_tail_offset + 2 * (cp_idx - 3) + coord
        return None

    def gradients_to_vars(self, grad_upper: np.ndarray, grad_lower: np.ndarray) -> np.ndarray:
        grad = np.zeros(self.num_vars, dtype=float)
        for i in range(self.num_cp_upper):
            for coord in (0, 1):
                j = self.var_index(True, i, coord)
                if j is not None:
                    grad[j] += float(grad_upper[i, coord])
        for i in range(self.num_cp_lower):
            for coord in (0, 1):
                j = self.var_index(False, i, coord)
                if j is not None:
                    grad[j] += float(grad_lower[i, coord])
        return grad


def vars_to_control_points(
    vars: np.ndarray,
    num_cp_upper: int,
    num_cp_lower: int,
) -> tuple[np.ndarray, np.ndarray]:
    cp_upper = np.zeros((num_cp_upper, 2), dtype=float)
    cp_lower = np.zeros((num_cp_lower, 2), dtype=float)

    cp_upper[0] = [0.0, 0.0]
    cp_lower[0] = [0.0, 0.0]

    cp_upper[1] = [0.0, vars[0]]
    cp_lower[1] = [0.0, vars[1]]

    cp_upper[2] = [vars[2], vars[3]]
    cp_lower[2] = [vars[4], vars[5]]

    idx = 6
    for i in range(3, num_cp_upper):
        cp_upper[i] = vars[idx : idx + 2]
        idx += 2

    for i in range(3, num_cp_lower):
        cp_lower[i] = vars[idx : idx + 2]
        idx += 2

    return cp_upper, cp_lower


def control_points_to_initial_vars(
    cp_upper: np.ndarray,
    cp_lower: np.ndarray,
    num_cp_upper: int,
    num_cp_lower: int,
) -> np.ndarray:
    initial_vars: list[float] = []
    initial_vars.append(float(cp_upper[1, 1]))
    initial_vars.append(float(cp_lower[1, 1]))
    initial_vars.append(float(cp_upper[2, 0]))
    initial_vars.append(float(cp_upper[2, 1]))
    initial_vars.append(float(cp_lower[2, 0]))
    initial_vars.append(float(cp_lower[2, 1]))

    for i in range(3, num_cp_upper):
        initial_vars.extend([float(cp_upper[i, 0]), float(cp_upper[i, 1])])
    for i in range(3, num_cp_lower):
        initial_vars.extend([float(cp_lower[i, 0]), float(cp_lower[i, 1])])

    return np.asarray(initial_vars, dtype=float)


def smoothing_weights(num_cp: int) -> np.ndarray:
    if num_cp <= 2:
        return np.zeros(0, dtype=float)
    idx = np.arange(num_cp - 2, dtype=float)
    grad = 0.5 + 1.5 * (idx / (num_cp - 3)) if num_cp > 3 else np.ones(num_cp - 2, dtype=float)
    return grad * grad


def build_bounds(n_free_upper: int, n_free_lower: int) -> list[tuple[float | None, float | None]]:
    bounds: list[tuple[float | None, float | None]] = []
    bounds.append((0.001, 0.1))
    bounds.append((-0.1, -0.001))
    bounds.append((0.001, 0.5))
    bounds.append((0.001, 0.3))
    bounds.append((0.001, 0.5))
    bounds.append((-0.3, -0.001))

    for _ in range(n_free_upper):
        bounds.extend([(None, None), (None, None)])
    for _ in range(n_free_lower):
        bounds.extend([(None, None), (None, None)])

    return bounds

