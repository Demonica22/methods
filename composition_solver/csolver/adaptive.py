"""Адаптивный шаг с оценкой локальной ошибки по вложенной паре композиций.

Пара (high, low): на шаге h из одной точки (t, y) считаются
    y_high = Ψ^high_h(y)   (порядок p),
    y_low  = Ψ^low_h(y)    (порядок q < p),
и их разность — оценка локальной ошибки схемы низкого порядка, err ~ C h^{q+1}.
Решение продолжается по y_high (локальная экстраполяция, как в DOPRI),
шаг выбирается по стандартному правилу
    h_new = h · min(fac_max, max(fac_min, safety · err^{-1/(q+1)})).

Схемы «вложены» в смысле структуры: P6C = P4C(a h) ∘ P4C(b h) ∘ P4C(a h),
P4C = Φ(γ1 h) ∘ Φ(γ2 h) ∘ Φ(γ1 h). Общих стадий у Ψ^high_h и Ψ^low_h нет
(у них разные промежуточные шаги), поэтому оценка стоит лишних
stages(low) стадий опоры: +1/3 для P4C/P2, +3/9 для P6C/P4C, +3/5 для P6SC/P4C.

Для вещественной задачи обе схемы проецируются на Re после шага (как в
integrate), и ошибка меряется по вещественной части.
"""
import numpy as np

from .compositions import SCHEMES
from .solver import CountingProblem


class EmbeddedPair:
    def __init__(self, name, high, low, low_order):
        self.name, self.high, self.low = name, high, low
        self.q = low_order  # порядок оценщика: err ~ h^{q+1}

    @property
    def stages(self):
        return self.high.stages + self.low.stages

    def __repr__(self):
        return self.name


PAIRS = {
    "P4C/P2": EmbeddedPair("P4C/P2", SCHEMES["P4C"], SCHEMES["P2"], 2),
    "P6C/P4C": EmbeddedPair("P6C/P4C", SCHEMES["P6C"], SCHEMES["P4C"], 4),
    "P6SC/P4C": EmbeddedPair("P6SC/P4C", SCHEMES["P6SC"], SCHEMES["P4C"], 4),
}


def error_norm(d, y0, y1, rtol, atol):
    """RMS-норма с весами atol + rtol·max(|y0|, |y1|) (Hairer–Nørsett–Wanner)."""
    sc = atol + rtol * np.maximum(np.abs(y0), np.abs(y1))
    return float(np.sqrt(np.mean((np.abs(d) / sc) ** 2)))


def initial_step(problem, y0, t0, q, rtol, atol, T):
    """Начальный шаг по алгоритму HNW (II.4), для оценщика порядка q."""
    y0r = np.real(y0)
    f0 = np.real(problem.f(t0, y0))
    sc = atol + rtol * np.abs(y0r)
    d0 = np.sqrt(np.mean((y0r / sc) ** 2))
    d1 = np.sqrt(np.mean((f0 / sc) ** 2))
    h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
    h0 = min(h0, abs(T))
    y1 = y0r + h0 * f0
    f1 = np.real(problem.f(t0 + h0, y1.astype(complex)))
    d2 = np.sqrt(np.mean(((f1 - f0) / sc) ** 2)) / h0
    m = max(d1, d2)
    h1 = max(1e-6, h0 * 1e-3) if m <= 1e-15 else (0.01 / m) ** (1.0 / (q + 1))
    return min(100 * h0, h1, abs(T))


def integrate_adaptive(problem, base, pair, rtol=1e-6, atol=None, T=None, h0=None,
                       safety=0.9, fac_min=0.2, fac_max=5.0, h_max=None,
                       max_steps=200000, project_real=True):
    """Решить задачу на [t0, t0 + T] с адаптивным шагом.

    pair: EmbeddedPair или ключ PAIRS. atol по умолчанию = rtol.
    Возвращает y (в t0 + T), историю t/h принятых шагов, число принятых/отброшенных
    шагов и число вычислений f и jac (включая отброшенные шаги и оценщик).
    """
    pair = PAIRS[pair] if isinstance(pair, str) else pair
    atol = rtol if atol is None else atol
    T = problem.T if T is None else T
    h_max = abs(T) if h_max is None else h_max
    cp = CountingProblem(problem)
    t, t_end = problem.t0, problem.t0 + T
    y = np.array(problem.y0, dtype=complex)
    q = pair.q
    h = initial_step(cp, y, t, q, rtol, atol, T) if h0 is None else h0
    ts, hs, errs = [t], [], []
    n_acc = n_rej = 0
    rejected = False
    while t < t_end - 1e-14 * max(1.0, abs(t_end)):
        if n_acc + n_rej >= max_steps:
            raise RuntimeError(f"превышено max_steps = {max_steps} (t = {t})")
        h = min(h, h_max, t_end - t)
        yh = pair.high.step(base, cp, t, y, h)
        yl = pair.low.step(base, cp, t, y, h)
        if project_real:
            yh, yl = yh.real.astype(complex), yl.real.astype(complex)
        d = (yh - yl).real if project_real else yh - yl
        err = error_norm(d, y, yh, rtol, atol)
        if not np.isfinite(err):
            err, fac = np.inf, fac_min
        else:
            fac = safety * (1.0 / max(err, 1e-16)) ** (1.0 / (q + 1))
        if err <= 1.0:
            t = t + h
            y = yh
            n_acc += 1
            ts.append(t); hs.append(h); errs.append(err)
            # после отказа шаг не увеличиваем (HNW)
            h = h * min(1.0 if rejected else fac_max, max(fac_min, fac))
            rejected = False
        else:
            n_rej += 1
            rejected = True
            h = h * min(1.0, max(fac_min, fac))
            if h < 1e-14 * max(1.0, abs(t)):
                raise RuntimeError(f"шаг слишком мал (t = {t}, h = {h})")
    return dict(y=y.real, y_complex=y, t=np.array(ts), h=np.array(hs), err=np.array(errs),
                n_accepted=n_acc, n_rejected=n_rej, nf=cp.nf, nj=cp.nj)
