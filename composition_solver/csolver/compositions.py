"""Композиционные схемы Psi_h = Phi_{g_m h} o ... o Phi_{g_1 h}.

Для симметричного опорного метода порядка 2 используются схемы «тройного прыжка»
(triple jump) g = (g1, 1 - 2 g1, g1), g1 = 1 / (2 - 2^{1/(p+1)} w), где
w^{p+1} = 1. Вещественный корень w = 1 даёт классические коэффициенты Йошиды
(средний шаг отрицателен), комплексный w даёт коэффициенты с Re g_i > 0.
Порядок 6 получается вложенным тройным прыжком (9 стадий).
"""
import numpy as np


def triple_jump_gamma(p, k=0):
    """Коэффициенты тройного прыжка, повышающего порядок p -> p + 2.

    k = 0: вещественные (Йошида). k != 0: комплексные, w = exp(2 pi i k / (p+1)).
    """
    w = np.exp(2j * np.pi * k / (p + 1))
    g1 = 1.0 / (2.0 - 2.0 ** (1.0 / (p + 1)) * w)
    g = np.array([g1, 1 - 2 * g1, g1])
    return g.real.copy() if k == 0 else g


def nest(outer, inner):
    """Композиция композиций: каждая стадия outer заменяется на inner."""
    return np.array([a * b for a in outer for b in inner])


class Composition:
    def __init__(self, name, gammas, nominal_order):
        self.name = name
        self.gammas = np.asarray(gammas)
        self.nominal_order = nominal_order  # порядок для СИММЕТРИЧНОЙ опоры 2-го порядка
        self.is_complex = np.iscomplexobj(self.gammas) and np.any(np.abs(self.gammas.imag) > 0)

    @property
    def stages(self):
        return len(self.gammas)

    def moments(self, kmax=7):
        return {k: complex(np.sum(self.gammas ** k)) for k in range(1, kmax + 1)}

    def step(self, base, problem, t, y, h):
        for g in self.gammas:
            y = base.step(problem, t, y, g * h)
            t = t + g * h
        return y

    def __repr__(self):
        return self.name


# Схема 2-го порядка: сам опорный метод (одна стадия, g = 1).
P2 = Composition("P2 (опора)", [1.0], 2)
# Порядок 4: тройной прыжок, комплексный и вещественный (для сравнения).
TJ4C = triple_jump_gamma(2, 1)
P4C = Composition("P4 complex TJ", TJ4C, 4)
P4R = Composition("P4 real TJ", triple_jump_gamma(2, 0), 4)
# Порядок 6: вложенный тройной прыжок 4 -> 6 (k = 2 выбран как корень с меньшими |Im g|).
P6C = Composition("P6 complex TJ", nest(triple_jump_gamma(4, 2), TJ4C), 6)
P6R = Composition("P6 real TJ", nest(triple_jump_gamma(4, 0), triple_jump_gamma(2, 0)), 6)
# Двухстадийная комплексная схема g = 1/2 ± i/(2 sqrt 3): sum g = 1, sum g^3 = 0.
# Для НЕсимметричной опоры порядка 2 даёт порядок 3 (Hansen–Ostermann 2009).
G2 = 0.5 + 1j / (2 * np.sqrt(3))
P3C = Composition("P3 complex 2-stage", [G2, np.conj(G2)], 3)

# Симметрично-сопряжённая схема 6-го порядка (γ_{6-i} = conj γ_i), найдена
# experiments/conj_search.py. Порядок 6 — для симметричной опоры с проекцией Re.
# Набор γ замкнут относительно сопряжения и Re γ > 0, поэтому композиция трапеций A-устойчива.
_a = 0.1910327069982427 + 0.08837831138001428j
_b = 0.17122959429335702 - 0.1870249473535133j
P6SC = Composition("P6 sym-conj 5-stage", [_a, _b, 1 - 2 * (_a + _b).real, np.conj(_b), np.conj(_a)], 6)

SCHEMES = {"P2": P2, "P4C": P4C, "P4R": P4R, "P6C": P6C, "P6R": P6R, "P3C": P3C, "P6SC": P6SC}


def found_schemes(path=None):
    """Схемы, найденные поиском коэффициентов (experiments/coeff_search.py).

    Ключи вида 'midpoint_p6_pal_Re': опора, порядок, палиндром/общий, с проекцией Re или без.
    """
    import json, os
    path = path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "coeff_search.json")
    out = {}
    for r in json.load(open(path)):
        if r.get("stages"):
            key = f"{r['base']}_p{r['order']}_{'pal' if r['palindromic'] else 'gen'}_{'Re' if r['project_real'] else 'C'}"
            out[key] = Composition(key, np.array([a + 1j * b for a, b in r["best"]]), r["order"])
    return out
