"""Интегрирование с постоянным шагом."""
import numpy as np


class CountingProblem:
    """Обёртка задачи, считающая вызовы f и jac."""

    def __init__(self, problem):
        self.p = problem
        self.nf = 0
        self.nj = 0

    def f(self, t, y):
        self.nf += 1
        return self.p.f(t, y)

    def jac(self, t, y):
        self.nj += 1
        return self.p.jac(t, y)


def integrate(problem, base, scheme, h_or_N, T=None, project_real=True, return_path=False):
    """Решить задачу на [t0, t0 + T] за N постоянных шагов.

    project_real: после каждого полного шага композиции берётся Re(y)
    (для вещественной задачи). Иначе решение остаётся комплексным до конца.
    """
    T = problem.T if T is None else T
    N = int(h_or_N) if isinstance(h_or_N, (int, np.integer)) else int(round(T / h_or_N))
    h = T / N
    cp = CountingProblem(problem)
    t = problem.t0
    y = np.array(problem.y0, dtype=complex)
    path = [y.real.copy()] if return_path else None
    for n in range(N):
        y = scheme.step(base, cp, t, y, h)
        if project_real:
            y = y.real.astype(complex)
        t = problem.t0 + (n + 1) * h
        if return_path:
            path.append(y.real.copy())
    out = dict(y=y.real, y_complex=y, h=h, N=N, nf=cp.nf, nj=cp.nj)
    if return_path:
        out["path"] = np.array(path)
    return out
