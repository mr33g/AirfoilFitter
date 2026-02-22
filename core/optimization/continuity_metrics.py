from __future__ import annotations

import numpy as np

from utils import bspline_helper


def start_derivative_weights(
    num_cp: int,
    knot_vector: np.ndarray,
    degree: int,
    max_order: int,
) -> list[np.ndarray] | None:
    cp_coeff = np.eye(num_cp, dtype=float)
    kv = np.asarray(knot_vector, dtype=float).copy()
    d = int(degree)
    weights: list[np.ndarray] = []
    for _ in range(max_order):
        if d <= 0 or cp_coeff.shape[0] < 2 or len(kv) < d + 2:
            return None
        next_coeff = np.zeros((cp_coeff.shape[0] - 1, num_cp), dtype=float)
        for i in range(next_coeff.shape[0]):
            denom = float(kv[i + d + 1] - kv[i + 1])
            if abs(denom) <= 1e-15:
                return None
            next_coeff[i] = d * (cp_coeff[i + 1] - cp_coeff[i]) / denom
        cp_coeff = next_coeff
        kv = kv[1:-1]
        d -= 1
        weights.append(cp_coeff[0].copy())
    return weights


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def curvature_value_and_cp_grad(
    cp: np.ndarray,
    knot_vector: np.ndarray,
    degree: int,
    weights2: list[np.ndarray] | None,
) -> tuple[float, np.ndarray]:
    grad_cp = np.zeros_like(cp)
    if weights2 is None:
        kappa = float(bspline_helper.compute_curvature_at_zero(cp, knot_vector, degree))
        return kappa, grad_cp

    w1, w2 = weights2
    d1 = w1 @ cp
    d2 = w2 @ cp
    v2 = float(d1[0] * d1[0] + d1[1] * d1[1])
    if v2 <= 1e-12:
        return 0.0, grad_cp

    cross = cross2(d1, d2)
    abs_cross = abs(cross)
    denom = v2 ** 1.5
    kappa = float(abs_cross / denom)
    cross_sign = 0.0 if abs_cross <= 1e-15 else float(np.sign(cross))

    for i in range(len(cp)):
        w1i = float(w1[i])
        w2i = float(w2[i])
        for coord in (0, 1):
            if coord == 0:
                dd1 = np.array([w1i, 0.0], dtype=float)
                dd2 = np.array([w2i, 0.0], dtype=float)
            else:
                dd1 = np.array([0.0, w1i], dtype=float)
                dd2 = np.array([0.0, w2i], dtype=float)
            dcross = cross2(dd1, d2) + cross2(d1, dd2)
            dv2 = 2.0 * float(d1[0] * dd1[0] + d1[1] * dd1[1])
            grad_cp[i, coord] = (cross_sign * dcross) / denom - abs_cross * 1.5 * dv2 / (v2 ** 2.5)
    return kappa, grad_cp


def curvature_derivative_value_and_cp_grad(
    cp: np.ndarray,
    knot_vector: np.ndarray,
    degree: int,
    weights3: list[np.ndarray] | None,
) -> tuple[float, np.ndarray]:
    grad_cp = np.zeros_like(cp)
    if weights3 is None:
        dk = float(bspline_helper.compute_curvature_derivative_at_zero(cp, knot_vector, degree))
        return dk, grad_cp

    w1, w2, w3 = weights3
    d1 = w1 @ cp
    d2 = w2 @ cp
    d3 = w3 @ cp
    v2 = float(d1[0] * d1[0] + d1[1] * d1[1])
    if v2 <= 1e-12:
        return 0.0, grad_cp

    a = cross2(d1, d3)
    b = cross2(d1, d2)
    d = float(d1[0] * d2[0] + d1[1] * d2[1])
    num = a * v2 - 3.0 * b * d
    denom = v2 ** 2.5
    dk = float(num / denom)

    for i in range(len(cp)):
        w1i = float(w1[i])
        w2i = float(w2[i])
        w3i = float(w3[i])
        for coord in (0, 1):
            if coord == 0:
                dd1 = np.array([w1i, 0.0], dtype=float)
                dd2 = np.array([w2i, 0.0], dtype=float)
                dd3 = np.array([w3i, 0.0], dtype=float)
            else:
                dd1 = np.array([0.0, w1i], dtype=float)
                dd2 = np.array([0.0, w2i], dtype=float)
                dd3 = np.array([0.0, w3i], dtype=float)
            da = cross2(dd1, d3) + cross2(d1, dd3)
            db = cross2(dd1, d2) + cross2(d1, dd2)
            ddot = float(dd1[0] * d2[0] + dd1[1] * d2[1] + d1[0] * dd2[0] + d1[1] * dd2[1])
            dv2 = 2.0 * float(d1[0] * dd1[0] + d1[1] * dd1[1])
            dnum = da * v2 + a * dv2 - 3.0 * (db * d + b * ddot)
            grad_cp[i, coord] = dnum / denom - num * 2.5 * dv2 / (v2 ** 3.5)
    return dk, grad_cp


def finite_diff_jacobian(fun, vars: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(vars, dtype=float)
    f0 = np.asarray(fun(x), dtype=float)
    n = x.size
    if f0.ndim == 0:
        jac = np.zeros(n, dtype=float)
    else:
        jac = np.zeros((f0.size, n), dtype=float)
    for j in range(n):
        h = eps * (1.0 + abs(float(x[j])))
        xp = x.copy()
        xp[j] += h
        fp = np.asarray(fun(xp), dtype=float)
        diff = (fp - f0) / h
        if f0.ndim == 0:
            jac[j] = float(diff)
        else:
            jac[:, j] = diff.ravel()
    return jac

