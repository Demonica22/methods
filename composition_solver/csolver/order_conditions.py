"""Условия порядка композиций и поиск коэффициентов (вопрос 4).

Модель опорного метода: Φ_h = exp(h F_1 + Σ_{k∈K} h^k F_k), где F_k — члены
модифицированного векторного поля. Симметричный метод: K = {3, 5, 7, ...};
средняя точка Грэгга (2 подшага), по измеренному дефекту симметрии h⁶: K = {3, 5, 6, 7};
произвольный несимметричный метод 2-го порядка: K = {3, 4, 5, 6, 7}.

Свободная алгебра Ли моделируется случайными вещественными матрицами F_k (размер d),
композиция считается как степенной ряд по h с матричными коэффициентами.
Ψ_h имеет порядок p, если коэффициенты при h^1..h^p у Ψ_h и exp(h F_1) совпадают.
При проекции Re после шага (вещественная задача) обнуляться должна только
вещественная часть этих коэффициентов.
"""
import numpy as np
from scipy.optimize import least_squares

K_SYMMETRIC = (3, 5, 7)
K_MIDPOINT = (3, 5, 6, 7)
K_GENERAL = (3, 4, 5, 6, 7)


class OrderModel:
    def __init__(self, K, p, d=7, seed=0, nsets=2):
        rng = np.random.default_rng(seed)
        self.K, self.p = tuple(k for k in K if k <= p), p
        self.sets = []
        for _ in range(nsets):
            F = {1: rng.standard_normal((d, d)) / np.sqrt(d)}
            for k in self.K:
                F[k] = rng.standard_normal((d, d)) / np.sqrt(d)
            self.sets.append(F)
        self.d = d
        self.targets = [self._exp_series({1: F[1]}, 1.0) for F in self.sets]

    def _mul(self, A, B):
        C = np.zeros_like(A)
        for i in range(self.p + 1):
            C[i] = sum(A[j] @ B[i - j] for j in range(i + 1))
        return C

    def _exp_series(self, F, g):
        """Ряд по h для exp(Σ_k (g h)^k F_k), обрезанный на степени p."""
        p, d = self.p, self.d
        X = np.zeros((p + 1, d, d), dtype=complex)
        for k, Fk in F.items():
            if k <= p:
                X[k] = g ** k * Fk
        E = np.zeros_like(X)
        E[0] = np.eye(d)
        term = E.copy()
        for j in range(1, p + 1):
            term = self._mul(term, X) / j
            E = E + term
        return E

    def series(self, gammas, F):
        S = None
        for g in gammas:
            E = self._exp_series(F, g)
            S = E if S is None else self._mul(E, S)
        return S

    def defects(self, gammas, project_real=True):
        """Норма отклонения коэффициента при h^j, j = 1..p (max по наборам матриц)."""
        out = np.zeros(self.p)
        for F, T in zip(self.sets, self.targets):
            D = self.series(gammas, F) - T
            if project_real:
                D = D.real
            out = np.maximum(out, [np.linalg.norm(D[j]) for j in range(1, self.p + 1)])
        return out

    def residual(self, gammas, project_real=True):
        r = []
        for F, T in zip(self.sets, self.targets):
            D = (self.series(gammas, F) - T)[1:]
            r.append(D.real.ravel())
            if not project_real:
                r.append(D.imag.ravel())
        return np.concatenate(r)


def achieved_order(gammas, K, pmax=8, project_real=True, tol=1e-9):
    m = OrderModel(K, pmax)
    dfc = m.defects(gammas, project_real)
    bad = np.where(dfc > tol)[0]
    return pmax if len(bad) == 0 else int(bad[0])


def expand(params, m, palindromic):
    """Вектор вещественных параметров -> комплексные коэффициенты γ_1..γ_m.

    palindromic: False — общий вид; True — γ_{m+1-i} = γ_i;
    "conj" — симметрично-сопряжённая схема γ_{m+1-i} = conj(γ_i) (средний коэффициент вещественный).
    """
    q = len(params) // 2
    z = params[:q] + 1j * params[q:]
    if not palindromic:
        return z
    if palindromic == "conj":
        if m % 2:
            z = z.copy()
            z[-1] = z[-1].real
        return np.concatenate([z, np.conj(z[:m // 2][::-1])])
    return np.concatenate([z, z[:m // 2][::-1]])


def search(K, p, m, palindromic=True, project_real=True, nstart=200, seed=1, tol=1e-11):
    """Многостартовый поиск коэффициентов (Левенберг–Марквардт) по условиям порядка.

    Возвращает список найденных решений, отсортированный по max|γ|.
    """
    model = OrderModel(K, p, d=5, nsets=1)   # быстрая модель для поиска
    check = OrderModel(K, p)                 # контрольная модель для отбора
    q = (m + 1) // 2 if palindromic else m
    rng = np.random.default_rng(seed)
    sols = []
    for _ in range(nstart):
        x0 = np.concatenate([rng.uniform(-0.2, 0.6, q), rng.uniform(-0.6, 0.6, q)]) / max(1, m / 3)
        f = lambda x: model.residual(expand(x, m, palindromic), project_real)
        try:
            res = least_squares(f, x0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
        except Exception:
            continue
        g = expand(res.x, m, palindromic)
        if np.max(np.abs(res.fun)) < tol and np.max(check.defects(g, project_real)) < 1e-9:
            if not any(np.allclose(np.sort_complex(g), np.sort_complex(s), atol=1e-8) for s in sols):
                sols.append(g)
    sols.sort(key=lambda g: np.max(np.abs(g)))
    return sols
