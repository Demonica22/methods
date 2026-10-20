"""Независимая реализация (ревью): опорные методы, задачи, коэффициенты композиций.

НЕ импортирует пакет csolver. Всё написано заново по определениям.
Два «бэкенда»:
  NP - numpy, complex128 (глобальные прогоны в двойной точности);
  MP - mpmath, произвольная точность (локальная ошибка без влияния округления).
Векторы: np.ndarray (NP) или mpmath.matrix(n,1) (MP).
"""
import json, os
import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "composition_solver")


# ----------------------------------------------------------------------------- backends
class NPBackend:
    name = "np"
    sin, cos, exp, sqrt = staticmethod(np.sin), staticmethod(np.cos), staticmethod(np.exp), staticmethod(np.sqrt)

    @staticmethod
    def vec(a):
        return np.array(a, dtype=complex)

    @staticmethod
    def mat(a):
        return np.array(a, dtype=complex)

    @staticmethod
    def solve(A, b):
        return np.linalg.solve(A, b)

    @staticmethod
    def norm(v):
        return float(np.max(np.abs(v)))

    @staticmethod
    def re(v):
        return np.real(v).astype(complex)

    @staticmethod
    def num(x):
        return complex(x)

    tol = 1e-15


class MPBackend:
    name = "mp"
    sin, cos, exp, sqrt = staticmethod(mp.sin), staticmethod(mp.cos), staticmethod(mp.exp), staticmethod(mp.sqrt)

    @staticmethod
    def vec(a):
        return mp.matrix([mp.mpc(x) for x in a])

    @staticmethod
    def mat(a):
        return mp.matrix([[mp.mpc(x) for x in row] for row in a])

    @staticmethod
    def solve(A, b):
        return mp.lu_solve(A, b)

    @staticmethod
    def norm(v):
        return max(abs(x) for x in v)

    @staticmethod
    def re(v):
        return mp.matrix([mp.mpc(mp.re(x)) for x in v])

    @staticmethod
    def num(x):
        return mp.mpc(x)

    @property
    def tol(self):
        return mp.mpf(10) ** (-(mp.mp.dps - 6))


NP, MP = NPBackend(), MPBackend()


def eye(B, n):
    return B.mat([[1 if i == j else 0 for j in range(n)] for i in range(n)])


# ----------------------------------------------------------------------------- problems
class Prob:
    """f(B, t, y) и jac(B, t, y) записаны через функции бэкенда B -> аналитичны по t и y."""

    def __init__(self, name, f, jac, y0, T, t0=0.0, exact=None):
        self.name, self.f_, self.jac_, self.y0, self.T, self.t0, self.exact = name, f, jac, list(y0), T, t0, exact
        self.n = len(y0)

    def f(self, B, t, y):
        return self.f_(B, t, y)

    def jac(self, B, t, y):
        return self.jac_(B, t, y)


def vdp(mu=1.0):
    f = lambda B, t, y: B.vec([y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]])
    J = lambda B, t, y: B.mat([[0, 1], [-2 * mu * y[0] * y[1] - 1, mu * (1 - y[0] ** 2)]])
    return Prob("vanderpol", f, J, [2.0, 0.0], 10.0)


def lotka():
    a, b, c, d = 1.0, 0.5, 0.75, 0.25
    f = lambda B, t, y: B.vec([a * y[0] - b * y[0] * y[1], -c * y[1] + d * y[0] * y[1]])
    J = lambda B, t, y: B.mat([[a - b * y[1], -b * y[0]], [d * y[1], -c + d * y[0]]])
    return Prob("lotka", f, J, [2.0, 1.0], 10.0)


def duffing():
    f = lambda B, t, y: B.vec([y[1], -0.2 * y[1] - y[0] - y[0] ** 3 + 0.5 * B.cos(1.2 * t)])
    J = lambda B, t, y: B.mat([[0, 1], [-1 - 3 * y[0] ** 2, -0.2]])
    return Prob("duffing", f, J, [1.0, 0.0], 10.0)


def lorenz():
    s, r, b = 10.0, 28.0, 8.0 / 3.0
    f = lambda B, t, y: B.vec([s * (y[1] - y[0]), y[0] * (r - y[2]) - y[1], y[0] * y[1] - b * y[2]])
    J = lambda B, t, y: B.mat([[-s, s, 0], [r - y[2], -1, -y[0]], [y[1], y[0], -b]])
    return Prob("lorenz", f, J, [1.0, 1.0, 1.0], 1.0)


def kepler(e=0.5):
    def f(B, t, y):
        r2 = y[0] ** 2 + y[1] ** 2
        r3 = r2 * B.sqrt(r2)
        return B.vec([y[2], y[3], -y[0] / r3, -y[1] / r3])

    def J(B, t, y):
        x, z = y[0], y[1]
        r2 = x * x + z * z
        r5 = r2 * r2 * B.sqrt(r2)
        a, b, c = (3 * x * x - r2) / r5, 3 * x * z / r5, (3 * z * z - r2) / r5
        return B.mat([[0, 0, 1, 0], [0, 0, 0, 1], [a, b, 0, 0], [b, c, 0, 0]])
    y0 = [1 - e, 0.0, 0.0, float(np.sqrt((1 + e) / (1 - e)))]
    return Prob("kepler", f, J, y0, 2 * np.pi, exact="periodic")


def generic3():
    """Сильно связанная неавтономная неполиномиальная 3-мерная задача (не из их набора)."""
    def f(B, t, y):
        return B.vec([y[1] ** 2 - B.sin(y[2]) + t,
                      B.exp(0.3 * y[0]) - y[2] * y[0],
                      B.cos(t * y[1]) + y[0] * y[1]])

    def J(B, t, y):
        return B.mat([[0, 2 * y[1], -B.cos(y[2])],
                      [0.3 * B.exp(0.3 * y[0]) - y[2], 0, -y[0]],
                      [y[1], -t * B.sin(t * y[1]) + y[0], 0]])
    return Prob("generic3", f, J, [0.3, -0.5, 0.7], 2.0, t0=0.2)


def oscillator():
    f = lambda B, t, y: B.vec([y[1], -y[0]])
    J = lambda B, t, y: B.mat([[0, 1], [-1, 0]])
    return Prob("oscillator", f, J, [1.0, 0.0], 10.0)


def logistic():
    f = lambda B, t, y: B.vec([y[0] * (1 - y[0])])
    J = lambda B, t, y: B.mat([[1 - 2 * y[0]]])
    return Prob("logistic", f, J, [0.1], 5.0)


# ----------------------------------------------------------------------------- base methods
class Trapezoid:
    name = "trapezoid"

    def step(self, B, P, t, y, h):
        f0 = P.f(B, t, y)
        t1 = t + h
        y1 = y + h * f0
        I = eye(B, P.n)
        last = None
        for it in range(60):
            G = y1 - y - (h / 2) * (f0 + P.f(B, t1, y1))
            J = I - (h / 2) * P.jac(B, t1, y1)
            d = B.solve(J, G)
            y1 = y1 - d
            nd = B.norm(d)
            if nd <= B.tol * (1 + B.norm(y1)) or (last is not None and B.name == "np" and nd >= last and nd < 1e-12):
                break
            last = nd
        return y1


class KD:
    """CD/KD по конспекту: Эйлер-Кромер (h1=s h, компоненты вперёд, время t) +
    D-метод (h2=(1-s) h, компоненты назад, каждая неявна только по себе, время t+h)."""

    def __init__(self, s=0.5):
        self.s = s
        self.name = "KD" if s == 0.5 else f"KD(s={s})"

    def step(self, B, P, t, y, h):
        h1, h2 = self.s * h, (1 - self.s) * h
        x = y.copy() if B.name == "np" else mp.matrix(y)
        n = P.n
        for i in range(n):
            x[i] = x[i] + h1 * P.f(B, t, x)[i]
        t1 = t + h
        for i in reversed(range(n)):
            c = x[i]
            xi = c + h2 * P.f(B, t1, x)[i]
            last = None
            for it in range(60):
                x[i] = xi
                g = xi - c - h2 * P.f(B, t1, x)[i]
                dg = 1 - h2 * P.jac(B, t1, x)[i, i]
                d = g / dg
                xi = xi - d
                if abs(d) <= B.tol * (1 + abs(xi)) or (last is not None and B.name == "np" and abs(d) >= last and abs(d) < 1e-12):
                    break
                last = abs(d)
            x[i] = xi
        return x


class Gragg:
    """Модифицированная средняя точка Грэгга, nsub подшагов + сглаживание
    S = (z_{n-1} + 2 z_n + z_{n+1})/4, z_{n+1} = z_{n-1} + 2 hs f(z_n)."""

    def __init__(self, nsub=2):
        self.nsub = nsub
        self.name = {1: "heun", 2: "gragg2"}.get(nsub, f"gragg{nsub}")

    def step(self, B, P, t, y, h):
        hs = h / self.nsub
        zp = y
        z = y + hs * P.f(B, t, y)
        for k in range(1, self.nsub):
            zp, z = z, zp + 2 * hs * P.f(B, t + k * hs, z)
        znext = zp + 2 * hs * P.f(B, t + h, z)
        return (zp + 2 * z + znext) / 4


class ExplicitMidpoint:
    name = "expl_midpoint"

    def step(self, B, P, t, y, h):
        return y + h * P.f(B, t + h / 2, y + (h / 2) * P.f(B, t, y))


# ----------------------------------------------------------------------------- compositions
def compose_step(B, base, P, t, y, h, gammas):
    for g in gammas:
        g = B.num(g)
        y = base.step(B, P, t, y, g * h)
        t = t + g * h
    return y


def triple_jump(p, k, prec=None):
    """γ1 = 1/(2 - 2^{1/(p+1)} w), w = exp(2πik/(p+1)); (γ1, 1-2γ1, γ1)."""
    w = mp.exp(2j * mp.pi * k / (p + 1))
    g1 = 1 / (2 - mp.power(2, mp.mpf(1) / (p + 1)) * w)
    return [g1, 1 - 2 * g1, g1]


def nest(outer, inner):
    return [a * b for a in outer for b in inner]


def schemes():
    """Коэффициенты, пересчитанные независимо (mpmath, текущая точность)."""
    P4C = triple_jump(2, 1)
    P4R = [mp.re(g) for g in triple_jump(2, 0)]
    P6C = nest(triple_jump(4, 2), P4C)
    P6R = [mp.re(g) for g in nest(triple_jump(4, 0), triple_jump(2, 0))]
    G = mp.mpf(1) / 2 + 1j / (2 * mp.sqrt(3))
    P3C = [G, mp.conj(G)]
    a = mp.mpc("0.19103270699824", "0.08837831138001")
    b = mp.mpc("0.17122959429336", "-0.18702494735351")
    P6SC_raw = [a, b, 1 - 2 * mp.re(a + b), mp.conj(b), mp.conj(a)]
    return {"P2": [mp.mpc(1)], "P3C": P3C, "P4C": P4C, "P4R": P4R, "P6C": P6C, "P6R": P6R,
            "P6SC": P6SC_raw}


def found_schemes():
    """Коэффициенты из results/coeff_search.json (данные, не код)."""
    out = {}
    for r in json.load(open(os.path.join(SRC, "results", "coeff_search.json"))):
        if r.get("stages"):
            key = f"{r['base']}_p{r['order']}_{'pal' if r['palindromic'] else 'gen'}_{'Re' if r['project_real'] else 'C'}"
            out[key] = dict(g=[mp.mpc(a, b) for a, b in r["best"]], base=r["base"], order=r["order"],
                            pal=r["palindromic"], proj=r["project_real"])
    return out


# ----------------------------------------------------------------------------- integration
def integrate(B, base, P, gammas, N, project_real=True, T=None):
    T = P.T if T is None else T
    h = T / N
    y = B.vec(P.y0)
    for n in range(N):
        t = P.t0 + n * h
        y = compose_step(B, base, P, t, y, h, gammas)
        if project_real:
            y = B.re(y)
    return y


def mp_flow(P, t0, y0, t1):
    """Точный поток в высокой точности (ряд Тейлора, mpmath.odefun)."""
    F = lambda t, y: [mp.re(v) for v in P.f(MP, t, MP.vec(y))]
    sol = mp.odefun(F, t0, [mp.mpf(v) for v in y0])
    return sol(t1)


def reference(P, dps=30, cache=True):
    """Эталон в t0+T: точное решение или odefun (Тейлор) с dps знаками; кэшируется в файл."""
    fn = os.path.join(HERE, "data", f"ref_{P.name}.json")
    if P.exact == "periodic":
        return np.array(P.y0, float)
    if cache and os.path.exists(fn):
        return np.array(json.load(open(fn)), float)
    with mp.workdps(dps):
        F = lambda t, y: [mp.re(v) for v in P.f(MP, t, MP.vec(y))]
        sol = mp.odefun(F, mp.mpf(P.t0), [mp.mpf(v) for v in P.y0])
        y = sol(mp.mpf(P.t0) + mp.mpf(P.T))
        ys = [mp.nstr(v, 25) for v in y]
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    json.dump(ys, open(fn, "w"))
    return np.array([float(v) for v in ys])
