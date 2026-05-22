from __future__ import annotations

import numpy as np


def te_handle_quality_penalty_and_grad(
    control_points: np.ndarray,
    target_direction: np.ndarray | None,
    *,
    min_length: float,
    short_length_weight: float,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """
    Softly prefer the TE handle vector to align with the measured TE tangent.

    The handle is the vector from the last free control point to the fixed TE:
    ``v = P_te - P_{n-1}``. The angle term uses ``sin(angle)^2`` via a 2-D
    cross product, avoiding acos and giving a smooth gradient. A small hinge
    penalty discourages near-zero handle length, where direction becomes
    numerically meaningless.
    """
    cp = np.asarray(control_points, dtype=float)
    grad = np.zeros_like(cp, dtype=float)
    if cp.ndim != 2 or cp.shape[0] < 2 or cp.shape[1] < 2 or target_direction is None:
        return 0.0, grad, {"angle": 0.0, "short_length": 0.0, "total": 0.0, "length": 0.0}

    target = np.asarray(target_direction, dtype=float)
    target_norm = float(np.linalg.norm(target))
    if target.shape[0] < 2 or target_norm <= eps:
        return 0.0, grad, {"angle": 0.0, "short_length": 0.0, "total": 0.0, "length": 0.0}

    t = target[:2] / target_norm
    v = cp[-1, :2] - cp[-2, :2]
    q_raw = float(np.dot(v, v))
    q = q_raw + float(eps)
    handle_length = float(np.sqrt(max(q_raw, 0.0)))

    # cross(v, t) = dot(v, [t_y, -t_x])
    cross_grad_v = np.array([t[1], -t[0]], dtype=float)
    cross = float(np.dot(v, cross_grad_v))
    angle_penalty = float((cross * cross) / q)
    grad_v = (2.0 * cross / q) * cross_grad_v - (2.0 * cross * cross / (q * q)) * v

    short_length_penalty = 0.0
    min_len = max(0.0, float(min_length))
    length_weight = max(0.0, float(short_length_weight))
    if min_len > eps:
        min_len_sq = min_len * min_len
        if q_raw < min_len_sq:
            gap = min_len_sq - q_raw
            short_length_penalty = float((gap / min_len_sq) ** 2)
            grad_v += length_weight * (-4.0 * gap / (min_len_sq * min_len_sq)) * v

    total = angle_penalty + length_weight * short_length_penalty

    grad[-2, :2] -= grad_v
    grad[-1, :2] += grad_v

    return (
        float(total),
        grad,
        {
            "angle": float(angle_penalty),
            "short_length": float(length_weight * short_length_penalty),
            "total": float(total),
            "length": handle_length,
        },
    )
