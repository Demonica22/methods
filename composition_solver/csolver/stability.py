"""Устойчивость композиций на линейной тестовой системе X' = A X размерности 2.

Скалярная задача Далквиста к полуявному KD неприменима, поэтому, как в статье
Galchenko et al. (2026), берётся 2x2 матрица с собственными числами σ ± iω и
коэффициентом асимметрии k = a/d:
    k = 1 -> нормальная форма [[σ, ω], [-ω, σ]] (с точностью до диагонального подобия),
    k = 0 -> форма Фробениуса [[0, 1], [-(σ²+ω²), 2σ]].
Трапеции и средняя точка инвариантны относительно подобия, для них k не важен.

Шаг h = 1 (точка σ + iω — это hλ). Матрица перехода композиции
    R = R(γ_m) ... R(γ_1),
с проекцией Re после полного шага: M = Re(R). Критерий: спектральный радиус ρ(M) ≤ 1.
Всё векторизовано по сетке.
"""
import numpy as np


def test_matrix(sig, om, k):
    """Элементы a, b, c, d тестовой матрицы (массивы той же формы, что sig)."""
    a = 2 * sig * k / (1 + k)
    d = 2 * sig / (1 + k)
    b = np.ones_like(sig)
    c = a * d - sig ** 2 - om ** 2
    return a, b, c, d


def _mm(X, Y):
    return [[X[0][0] * Y[0][0] + X[0][1] * Y[1][0], X[0][0] * Y[0][1] + X[0][1] * Y[1][1]],
            [X[1][0] * Y[0][0] + X[1][1] * Y[1][0], X[1][0] * Y[0][1] + X[1][1] * Y[1][1]]]


def step_matrix(base, A, g):
    """Матрица перехода одного шага опорного метода с (комплексным) шагом g."""
    a, b, c, d = A
    one, zero = np.ones_like(a, dtype=complex), np.zeros_like(a, dtype=complex)
    I = [[one, zero], [zero, one]]
    if base == "trapezoid":
        # (I - g/2 A)^{-1} (I + g/2 A)
        P = [[1 + g / 2 * a, g / 2 * b], [g / 2 * c, 1 + g / 2 * d]]
        Q = [[1 - g / 2 * a, -g / 2 * b], [-g / 2 * c, 1 - g / 2 * d]]
        det = Q[0][0] * Q[1][1] - Q[0][1] * Q[1][0]
        Qi = [[Q[1][1] / det, -Q[0][1] / det], [-Q[1][0] / det, Q[0][0] / det]]
        return _mm(Qi, P)
    if base.startswith("midpoint"):
        n = int(base.split(":")[1]) if ":" in base else 2
        hs = g / n
        Ah = [[hs * a, hs * b], [hs * c, hs * d]]
        add = lambda X, Y, s=1: [[X[i][j] + s * Y[i][j] for j in range(2)] for i in range(2)]
        z_prev = I
        z = add(I, Ah)
        for _ in range(1, n):
            z_prev, z = z, add(z_prev, _mm(Ah, z), 2)
        S = add(add(z_prev, z), _mm(Ah, z))
        return [[S[i][j] / 2 for j in range(2)] for i in range(2)]
    if base.startswith("KD"):
        s = float(base.split(":")[1]) if ":" in base else 0.5
        h1, h2 = s * g, (1 - s) * g
        # Эйлер–Кромер: x1 += h1 (a x1 + b x2); x2 += h1 (c x1 + d x2)
        E1 = [[1 + h1 * a, h1 * b], [zero, one]]
        E2 = [[one, zero], [h1 * c, 1 + h1 * d]]
        # D-метод, обратный порядок: x2 = (x2 + h2 c x1)/(1 - h2 d); x1 = (x1 + h2 b x2)/(1 - h2 a)
        D2 = [[one, zero], [h2 * c / (1 - h2 * d), 1 / (1 - h2 * d)]]
        D1 = [[1 / (1 - h2 * a), h2 * b / (1 - h2 * a)], [zero, one]]
        return _mm(D1, _mm(D2, _mm(E2, E1)))
    raise ValueError(base)


def amplification(base, gammas, sig, om, k=1.0, project_real=True):
    """Спектральный радиус матрицы перехода одного шага композиции."""
    A = test_matrix(np.asarray(sig, float), np.asarray(om, float), k)
    one = np.ones_like(A[0], dtype=complex)
    R = [[one, 0 * one], [0 * one, one]]
    for g in gammas:
        R = _mm(step_matrix(base, A, complex(g)), R)
    if project_real:
        R = [[R[i][j].real for j in range(2)] for i in range(2)]
    tr = R[0][0] + R[1][1]
    det = R[0][0] * R[1][1] - R[0][1] * R[1][0]
    disc = np.sqrt(tr.astype(complex) ** 2 - 4 * det)
    return np.maximum(np.abs((tr + disc) / 2), np.abs((tr - disc) / 2))
