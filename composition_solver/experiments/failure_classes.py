"""Эксперимент 5 (вопрос 2): нежёсткие задачи, на которых схемы не работают.

A. Неаналитическая правая часть: y' = -|y|, y(0) = 1 (решение e^{-t}, на нём y > 0,
   так что f гладкая вдоль вещественной траектории). Комплексный шаг требует
   аналитического продолжения f, а |z| не аналитична.
B. Правая часть, определённая только для вещественных аргументов (таблица, np.interp).
C. Слабо затухающий осциллятор x'' + 2ζω x' + ω² x = 0, ω = 40: при hω ≈ 5..30
   композиции трапеций с комплексными коэффициентами разваливаются, хотя сами трапеции
   и вещественная задача устойчивы (связь с вопросом 3).
D. Нормальная форма и форма Фробениуса для KD: одна и та же динамика, разный результат.
Запуск: python experiments/failure_classes.py
"""
import sys, os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
np.seterr(all="ignore")
from csolver import integrate, SCHEMES, BASE_METHODS
from csolver.problems import Problem
from csolver.base_methods import Trapezoid

OUT = os.path.join(ROOT, "results")
out = []
pr = lambda s="": (print(s), out.append(s))
NS = [10, 20, 40, 80, 160, 320, 640]


def orders(p, base, sn):
    e = np.array([np.max(np.abs(integrate(p, base, SCHEMES[sn], N)["y"] - p.reference())) for N in NS])
    return e, np.log2(e[:-1] / e[1:])


tt = np.linspace(0, 10, 101)
tab = np.cos(tt)
pb = Problem("table", lambda t, y: np.array([np.interp(t, tt, tab) - y[0]]),
             lambda t, y: np.array([[-1.0]]), [0.0], 5.0)

# ---------- A
pr("## A. Неаналитическая правая часть: y' = −|y|, y(0) = 1, T = 2\n")
pr("Для сравнения: та же задача в аналитической записи y' = −y.\n")
pr("| опора | схема | порядок для −y | порядок для −\\|y\\| | ошибка −\\|y\\| при N = 640 |")
pr("|---|---|---|---|---|")
pa = Problem("abs", lambda t, y: -np.abs(y), lambda t, y: np.array([[-np.sign(y[0].real)]]),
             [1.0], 2.0, exact=lambda: [np.exp(-2.0)])
pl = Problem("lin", lambda t, y: -y, lambda t, y: np.array([[-1.0]]), [1.0], 2.0,
             exact=lambda: [np.exp(-2.0)])
for bn, b in BASE_METHODS.items():
    for sn in ["P2", "P4R", "P4C", "P6R", "P6C"]:
        el, ql = orders(pl, b, sn)
        ea, qa = orders(pa, b, sn)
        okl = ql[el[1:] > 1e-12]
        oka = qa[ea[1:] > 1e-12]
        f = lambda q: f"{q[-1]:.2f}" if len(q) else "—"
        pr(f"| {bn} | {sn} | {f(okl)} | {f(oka)} | {ea[-1]:.1e} |")

# ---------- B
pr("\n## B. Правая часть из таблицы (np.interp)\n")
try:
    np.interp(1 + 1j, [0.0, 2.0], [0.0, 2.0])
except TypeError as ex:
    pr(f"- np.interp(1+1j, ...) с питоновским complex: `TypeError: {ex}`")
v = np.interp(np.complex128(1 + 1j), [0.0, 2.0], [0.0, 2.0])
pr(f"- np.interp(np.complex128(1+1j), [0, 2], [0, 2]) = {v!r}: мнимая часть молча отбрасывается.")
pr("- Внутри решателя время стадии t + γh комплексное (numpy-скаляр), так что табличная f "
   "получает неверный аргумент без всякой ошибки.")
for sn in ["P4R", "P4C"]:
    r = integrate(pb, BASE_METHODS["midpoint"], SCHEMES[sn], 50)
    pr(f"- {sn}, 50 шагов: y(5) = {r['y'][0]:.6f}")

# ---------- C
pr("\n## C. Слабо затухающий осциллятор, ω = 40, ζ = 0.001, T = 20, опора — трапеции\n")
om, ze = 40.0, 0.001
A = np.array([[0, 1], [-om ** 2, -2 * ze * om]])
from scipy.linalg import expm
pc = Problem("osc40", lambda t, y: A @ y, lambda t, y: A.astype(complex), [1.0, 0.0], 20.0,
             exact=lambda: expm(A * 20.0) @ np.array([1.0, 0.0]))
hw = np.array([0.5, 1, 2, 5, 10, 15, 20, 30])
pr("| hω | " + " | ".join(SCHEMES[s].name for s in ["P2", "P3C", "P4C", "P4R", "P6C", "P6R", "P6SC"]) + " |")
pr("|---|" + "---|" * 7)
grid = {}
for x in hw:
    N = int(round(20.0 * om / x))
    cells = []
    for sn in ["P2", "P3C", "P4C", "P4R", "P6C", "P6R", "P6SC"]:
        y = integrate(pc, Trapezoid(), SCHEMES[sn], N)["y"]
        amp = np.hypot(y[0], y[1] / om)
        grid[(x, sn)] = amp
        cells.append("**разнос**" if not np.isfinite(amp) or amp > 10 else f"{amp:.2g}")
    pr(f"| {x:g} | " + " | ".join(cells) + " |")
pr("\nВ ячейках: амплитуда sqrt(x^2 + (x'/ω)^2) в t = T, точное значение exp(−ζωT) ≈ 0.45; «разнос» — больше 10 или переполнение.")

# ---------- D
pr("\n## D. KD: одна и та же динамика в разной записи\n")
pr("Линейная система с λ = −5 ± 1.5i в нормальной форме (k = 1) и в форме Фробениуса "
   "(k = 0, переменные u, u'). Это одна задача в разных координатах; трапеции дают "
   "в обеих одинаковый результат, KD — нет. В ячейках: max|z(T)| в исходных координатах, T = 20 "
   "(точное значение ≈ 4e-44, «разнос» — больше 1).\n")
sig, w = -5.0, 1.5
A1 = np.array([[sig, w], [-w, sig]])
S = np.array([[1.0, 0.0], [sig, w]])
A0 = S @ A1 @ np.linalg.inv(S)
z0 = np.array([1.0, 0.0])
P1 = Problem("k1", lambda t, y: A1 @ y, lambda t, y: A1.astype(complex), z0, 20.0)
P0 = Problem("k0", lambda t, y: A0 @ y, lambda t, y: A0.astype(complex), S @ z0, 20.0)
pr("| h | KD, k = 1 | KD, k = 0 | KD+P4C, k = 1 | KD+P4C, k = 0 | трапеции, k = 0 |")
pr("|---|---|---|---|---|---|")
Si = np.linalg.inv(S)
fmt = lambda v: "**разнос**" if not np.isfinite(v) or v > 1 else f"{v:.1e}"
for h in [0.1, 0.25, 0.4, 0.6, 1.0]:
    N = int(round(20 / h))
    c = []
    for sn in ["P2", "P4C"]:
        for P, back in [(P1, np.eye(2)), (P0, Si)]:
            y = integrate(P, BASE_METHODS["KD"], SCHEMES[sn], N)["y"]
            c.append(np.max(np.abs(back @ y)))
    y = integrate(P0, BASE_METHODS["trapezoid"], SCHEMES["P2"], N)["y"]
    c.append(np.max(np.abs(Si @ y)))
    pr(f"| {h:g} | " + " | ".join(fmt(v) for v in c) + " |")

open(os.path.join(OUT, "failure_classes.md"), "w").write("\n".join(out) + "\n")
