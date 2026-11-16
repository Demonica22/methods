"""Точные условия порядка композиции для явной средней точки через деревья Бутчера.

Опора (явная средняя точка, РК-2, несимметричная):
    y1 = y0 + h f(y0 + h/2 f(y0)),   A = [[0, 0], [1/2, 0]],  b = (0, 1).
Композиция Ψ_h = Φ_{γ_m h} ∘ … ∘ Φ_{γ_1 h} — это явный РК-метод с 2m стадиями
(блочная таблица Бутчера с комплексными коэффициентами). Порядок p ⇔ для всех
деревьев τ с |τ| ≤ p элементарный вес Φ(τ) = 1/τ!  (с проекцией Re: Re Φ(τ) = 1/τ!,
т.к. элементарные дифференциалы вещественной задачи вещественны).
В отличие от модели «свободной алгебры Ли» (order_conditions.py, rv_words.py) здесь
учитывается конкретная структура опоры, а не произвольный метод 2-го порядка.
"""
import numpy as np

A0 = [[0, 0], [0.5, 0]]
B0 = [0, 1]


def trees(pmax):
    """Корневые деревья до порядка pmax: канонический вид — отсортированный кортеж детей."""
    by = {1: [()]}
    for n in range(2, pmax + 1):
        res = set()

        def parts(rem, maxkey, acc):
            if rem == 0:
                res.add(tuple(sorted(acc)))
                return
            for k in range(1, rem + 1):
                for t in by[k]:
                    key = (k, t)
                    if maxkey is None or key <= maxkey:
                        parts(rem - k, key, acc + [t])
        parts(n - 1, None, [])
        by[n] = sorted(res)
    return by


def order(t):
    return 1 + sum(order(c) for c in t)


def factorial(t):
    return order(t) * np.prod([factorial(c) for c in t]) if t else 1


TREES = trees(7)
ALL = [(n, t) for n in range(1, 8) for t in TREES[n]]


def tableau(gammas, A=A0, b=B0, lib=np):
    """Блочная таблица Бутчера композиции. lib = numpy или mpmath."""
    s, m = len(b), len(gammas)
    N = s * m
    if lib is np:
        AA = np.zeros((N, N), dtype=complex)
        bb = np.zeros(N, dtype=complex)
    else:
        AA = lib.matrix(N, N)
        bb = lib.matrix(N, 1)
    for i, g in enumerate(gammas):
        for j in range(i):
            for r in range(s):
                for c in range(s):
                    AA[i * s + r, j * s + c] = gammas[j] * b[c]
        for r in range(s):
            for c in range(s):
                AA[i * s + r, i * s + c] = g * A[r][c]
            bb[i * s + r] = g * b[r]
    return AA, bb


def weights_np(gammas, pmax, A=A0, b=B0):
    """Элементарные веса Φ(τ) композиции для всех деревьев порядка ≤ pmax (numpy, complex)."""
    AA, bb = tableau(gammas, A, b)
    memo = {(): np.ones(len(bb), dtype=complex)}

    def g(t):
        if t not in memo:
            v = np.ones(len(bb), dtype=complex)
            for c in t:
                v = v * (AA @ g(c))
            memo[t] = v
        return memo[t]
    return np.array([bb @ g(t) for n in range(1, pmax + 1) for t in TREES[n]])


def targets(pmax):
    return np.array([1.0 / factorial(t) for n in range(1, pmax + 1) for t in TREES[n]])


def defects(gammas, pmax, project_real=True, A=A0, b=B0):
    """Максимальный дефект |Φ(τ) − 1/τ!| по деревьям каждого порядка 1..pmax."""
    d = weights_np(gammas, pmax, A, b) - targets(pmax)
    if project_real:
        d = d.real
    orders = [n for n in range(1, pmax + 1) for _ in TREES[n]]
    return np.array([max(abs(d[i]) for i in range(len(d)) if orders[i] == n) for n in range(1, pmax + 1)])


def achieved_order(gammas, pmax=7, project_real=True, tol=1e-10, A=A0, b=B0):
    dfc = defects(gammas, pmax, project_real, A, b)
    bad = np.where(dfc > tol)[0]
    return pmax if len(bad) == 0 else int(bad[0])


def residual(gammas, pmax, project_real=True, A=A0, b=B0):
    d = weights_np(gammas, pmax, A, b) - targets(pmax)
    return d.real if project_real else np.concatenate([d.real, d.imag])


def weights_mp(gammas, pmax, A=A0, b=B0):
    import mpmath as mp
    AA, bb = tableau([mp.mpc(g) for g in gammas], [[mp.mpf(x) for x in r] for r in A],
                     [mp.mpf(x) for x in b], lib=mp)
    N = len(bb)
    memo = {(): mp.matrix([1] * N)}

    def g(t):
        if t not in memo:
            v = mp.matrix([1] * N)
            for c in t:
                w = AA * g(c)
                v = mp.matrix([v[i] * w[i] for i in range(N)])
            memo[t] = v
        return memo[t]
    return [sum(bb[i] * g(t)[i] for i in range(N)) for n in range(1, pmax + 1) for t in TREES[n]]


if __name__ == "__main__":
    print({n: len(TREES[n]) for n in TREES})
    # контроль: сама опора имеет порядок 2, общий тройной прыжок — 3
    print("опора:", achieved_order([1.0]), defects([1.0], 4))
