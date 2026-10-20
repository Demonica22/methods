"""Тестовые (нежёсткие) задачи. f и jac должны принимать комплексные t и y
(аналитическое продолжение), поэтому используются только numpy-функции."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


class Problem:
    def __init__(self, name, f, jac, y0, T, t0=0.0, exact=None, descr=""):
        self.name, self._f, self._jac = name, f, jac
        self.y0, self.T, self.t0 = np.asarray(y0, dtype=float), T, t0
        self._exact, self.descr = exact, descr
        self._ref = None

    def f(self, t, y):
        return self._f(t, y)

    def jac(self, t, y):
        return self._jac(t, y)

    def reference(self):
        """Точное решение в t0 + T или эталон DOP853 (rtol = 1e-14)."""
        if self._ref is None:
            if self._exact is not None:
                self._ref = np.asarray(self._exact(), dtype=float)
            else:
                sol = solve_ivp(lambda t, y: np.real(self._f(t, y)), (self.t0, self.t0 + self.T),
                                self.y0, method="DOP853", rtol=1e-14, atol=1e-16)
                self._ref = sol.y[:, -1]
        return self._ref


def _lin(A, y0, T, name, descr):
    A = np.asarray(A, dtype=float)
    return Problem(name, lambda t, y: A @ y, lambda t, y: A.astype(complex), y0, T,
                   exact=lambda: expm(A * T) @ np.asarray(y0, float), descr=descr)


def oscillator():
    return _lin([[0, 1], [-1, 0]], [1.0, 0.0], 10.0, "oscillator",
                "Гармонический осциллятор x'' = -x, T = 10")


def linear_companion(sigma=-0.2, omega=2.0):
    # Фробениусова (сопровождающая) форма: коэффициент асимметрии k = 0
    A = [[0, 1], [-(sigma ** 2 + omega ** 2), 2 * sigma]]
    return _lin(A, [1.0, 0.0], 10.0, "companion",
                f"Линейная система в форме Фробениуса (k = 0), λ = {sigma}±{omega}i, T = 10")


def logistic():
    def ex():
        T = 5.0
        y0 = 0.1
        return [y0 * np.exp(T) / (1 - y0 + y0 * np.exp(T))]
    return Problem("logistic", lambda t, y: y * (1 - y), lambda t, y: np.array([[1 - 2 * y[0]]]),
                   [0.1], 5.0, exact=ex, descr="Скалярная логистическая y' = y(1 - y), T = 5")


def van_der_pol(mu=1.0):
    f = lambda t, y: np.array([y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]])
    J = lambda t, y: np.array([[0, 1], [-2 * mu * y[0] * y[1] - 1, mu * (1 - y[0] ** 2)]])
    return Problem("vanderpol", f, J, [2.0, 0.0], 10.0, descr=f"Ван дер Поль, μ = {mu}, T = 10")


def lotka_volterra():
    a, b, c, d = 1.0, 0.5, 0.75, 0.25
    f = lambda t, y: np.array([a * y[0] - b * y[0] * y[1], -c * y[1] + d * y[0] * y[1]])
    J = lambda t, y: np.array([[a - b * y[1], -b * y[0]], [d * y[1], -c + d * y[0]]])
    return Problem("lotka", f, J, [2.0, 1.0], 10.0, descr="Лотка–Вольтерра, T = 10")


def kepler(e=0.5):
    def f(t, y):
        r3 = (y[0] ** 2 + y[1] ** 2) ** 1.5
        return np.array([y[2], y[3], -y[0] / r3, -y[1] / r3])

    def J(t, y):
        x, z = y[0], y[1]
        r2 = x * x + z * z
        r5 = r2 ** 2.5
        a = (3 * x * x - r2) / r5
        b = 3 * x * z / r5
        c = (3 * z * z - r2) / r5
        return np.array([[0, 0, 1, 0], [0, 0, 0, 1], [a, b, 0, 0], [b, c, 0, 0]])
    y0 = [1 - e, 0.0, 0.0, np.sqrt((1 + e) / (1 - e))]
    return Problem("kepler", f, J, y0, 2 * np.pi, exact=lambda: y0,
                   descr=f"Задача Кеплера, e = {e}, один период T = 2π")


def lorenz():
    s, r, b = 10.0, 28.0, 8.0 / 3.0
    f = lambda t, y: np.array([s * (y[1] - y[0]), y[0] * (r - y[2]) - y[1], y[0] * y[1] - b * y[2]])
    J = lambda t, y: np.array([[-s, s, 0], [r - y[2], -1, -y[0]], [y[1], y[0], -b]])
    return Problem("lorenz", f, J, [1.0, 1.0, 1.0], 1.0, descr="Лоренц, T = 1")


def forced_duffing():
    # неавтономная задача: x'' + 0.2 x' + x + x^3 = 0.5 cos(1.2 t)
    f = lambda t, y: np.array([y[1], -0.2 * y[1] - y[0] - y[0] ** 3 + 0.5 * np.cos(1.2 * t)])
    J = lambda t, y: np.array([[0, 1], [-1 - 3 * y[0] ** 2, -0.2]])
    return Problem("duffing", f, J, [1.0, 0.0], 10.0,
                   descr="Вынужденный Дуффинг (неавтономная), T = 10")


ALL = [oscillator, linear_companion, logistic, van_der_pol, lotka_volterra, kepler, lorenz,
       forced_duffing]
