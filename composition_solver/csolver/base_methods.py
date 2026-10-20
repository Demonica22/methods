"""Опорные одношаговые методы 2-го порядка.

Все методы работают в комплексной арифметике: шаг h и состояние y могут быть
комплексными. Это нужно, чтобы стадии композиции с комплексными коэффициентами
gamma_i выполнялись как обычный шаг Phi_{gamma_i h}.

Интерфейс: method.step(problem, t, y, h) -> y_new.
problem.f(t, y) -> dy/dt, problem.jac(t, y) -> матрица Якоби (нужна неявным методам).
"""
import numpy as np

NEWTON_TOL = 1e-15
NEWTON_MAXIT = 50


class BaseMethod:
    name = "base"
    order = 2
    symmetric = True

    def step(self, problem, t, y, h):
        raise NotImplementedError

    def __repr__(self):
        return self.name


class Trapezoid(BaseMethod):
    """Неявный метод трапеций: y1 = y0 + h/2 (f(t0,y0) + f(t1,y1)).

    Симметричный, порядок 2, A-устойчивый (для вещественного h > 0).
    Нелинейное уравнение решается методом Ньютона в комплексной арифметике.
    """
    name = "trapezoid"
    symmetric = True

    def step(self, problem, t, y, h):
        f0 = problem.f(t, y)
        t1 = t + h
        rhs = y + 0.5 * h * f0
        y1 = y + h * f0  # предиктор: явный Эйлер
        n = y.size
        I = np.eye(n)
        for _ in range(NEWTON_MAXIT):
            G = y1 - rhs - 0.5 * h * problem.f(t1, y1)
            J = I - 0.5 * h * problem.jac(t1, y1)
            dy = np.linalg.solve(J, G)
            y1 = y1 - dy
            if np.max(np.abs(dy)) <= NEWTON_TOL * (1.0 + np.max(np.abs(y1))):
                break
        return y1


class KD(BaseMethod):
    """Полуявный композиционный диагональный метод (KD / CD, Бутусов).

    Шаг = полушаг Эйлера–Кромера (h1 = s h, проход по компонентам вперёд, каждая
    компонента сразу использует уже обновлённые) + полушаг D-метода (h2 = (1-s) h,
    проход в обратном порядке, каждая компонента неявна только по себе самой).
    При s = 0.5 метод симметричен и имеет порядок 2; при s != 0.5 порядок 1.
    Скалярные неявные уравнения D-полушага решаются Ньютоном с производной J_ii.
    """
    symmetric = True

    def __init__(self, s=0.5):
        self.s = s
        self.name = "KD" if s == 0.5 else f"KD(s={s})"
        self.symmetric = (s == 0.5)
        self.order = 2 if s == 0.5 else 1

    def step(self, problem, t, y, h):
        h1 = self.s * h
        h2 = (1 - self.s) * h
        x = np.array(y, dtype=complex)
        n = x.size
        # полушаг 1: Эйлер–Кромер, время t (левый конец)
        for i in range(n):
            x[i] = x[i] + h1 * problem.f(t, x)[i]
        # полушаг 2: D-метод, обратный порядок, время t + h (правый конец)
        t1 = t + h
        for i in reversed(range(n)):
            c = x[i]
            xi = c + h2 * problem.f(t1, x)[i]  # предиктор
            for _ in range(NEWTON_MAXIT):
                x[i] = xi
                g = xi - c - h2 * problem.f(t1, x)[i]
                dg = 1.0 - h2 * problem.jac(t1, x)[i, i]
                dxi = g / dg
                xi = xi - dxi
                if abs(dxi) <= NEWTON_TOL * (1.0 + abs(xi)):
                    break
            x[i] = xi
        return x


class GraggMidpoint(BaseMethod):
    """Модифицированная явная средняя точка (Gragg) со сглаживающим шагом.

    Шаг H разбивается на nsub подшагов hs = H / nsub:
        z0 = y0,  z1 = z0 + hs f(t0, z0),
        z_{k+1} = z_{k-1} + 2 hs f(t_k, z_k),   k = 1..nsub-1,
        y1 = (z_{nsub-1} + z_nsub + hs f(t1, z_nsub)) / 2   (сглаживание Грэгга).
    Порядок 2, явный, НЕ симметричный как одношаговое отображение
    (при nsub = 1 совпадает с методом Хойна).
    """
    symmetric = False

    def __init__(self, nsub=2):
        self.nsub = nsub
        self.name = "midpoint" if nsub == 2 else f"midpoint(n={nsub})"

    def step(self, problem, t, y, h):
        hs = h / self.nsub
        z_prev = np.array(y, dtype=complex)
        z = z_prev + hs * problem.f(t, z_prev)
        for k in range(1, self.nsub):
            z_prev, z = z, z_prev + 2 * hs * problem.f(t + k * hs, z)
        return 0.5 * (z_prev + z + hs * problem.f(t + h, z))


BASE_METHODS = {
    "trapezoid": Trapezoid(),
    "KD": KD(0.5),
    "midpoint": GraggMidpoint(2),
}
