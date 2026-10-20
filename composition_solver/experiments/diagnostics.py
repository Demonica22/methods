"""Эксперимент 2: диагностика опорных методов.

(a) дефект симметричности  ||Phi_{-h}(Phi_h(y0)) - y0||  (у симметричного метода = 0);
(b) KD(s=0.5) совпадает с трапециями для скалярной задачи;
(c) влияние проекции Re после шага;
(d) модифицированная средняя точка с разным числом подшагов nsub.
"""
import sys, os, warnings
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
from csolver import integrate, problems, SCHEMES
from csolver.base_methods import Trapezoid, KD, GraggMidpoint

OUT = os.path.join(ROOT, "results")
out = []
pr = lambda s="": (print(s), out.append(s))


def slopes(xs):
    xs = np.asarray(xs)
    return " ".join(f"{v:.2f}" for v in np.log2(xs[:-1] / xs[1:]))


pr("## (a) Дефект симметричности ||Φ_{-h}∘Φ_h(y0) − y0||, Ван дер Поль, y0 = (2, 0)\n")
pr("| метод | h=0.1 | h=0.05 | h=0.025 | h=0.0125 | наклон log2 |")
pr("|---|---|---|---|---|---|")
p = problems.van_der_pol()
y0 = np.array(p.y0, complex)
for b in [Trapezoid(), KD(0.5), KD(0.3), GraggMidpoint(1), GraggMidpoint(2), GraggMidpoint(4)]:
    d = []
    for h in [0.1, 0.05, 0.025, 0.0125]:
        d.append(np.max(np.abs(b.step(p, h, b.step(p, 0.0, y0, h), -h) - y0)))
    pr(f"| {b} | " + " | ".join(f"{v:.1e}" for v in d) + f" | {slopes(d)} |")

pr("\n## (b) KD(s=0.5) против трапеций на скалярной логистической задаче\n")
p = problems.logistic()
for N in [10, 40, 160]:
    a = integrate(p, KD(0.5), SCHEMES["P4C"], N)["y"]
    b = integrate(p, Trapezoid(), SCHEMES["P4C"], N)["y"]
    pr(f"- N = {N}: |y_KD − y_trap| = {abs(a - b)[0]:.1e}")

pr("\n## (c) Проекция Re после каждого шага: наблюдаемый порядок (Ван дер Поль, N = 40..640)\n")
pr("| опора | схема | с проекцией | без проекции |")
pr("|---|---|---|---|")
p = problems.van_der_pol()
ref = p.reference()
for b in [Trapezoid(), KD(0.5), GraggMidpoint(2)]:
    for sn in ["P3C", "P4C", "P6C"]:
        res = []
        for proj in [True, False]:
            e = [np.max(np.abs(integrate(p, b, SCHEMES[sn], N, project_real=proj)["y"] - ref))
                 for N in [40, 80, 160, 320, 640]]
            res.append(slopes(e))
        pr(f"| {b} | {sn} | {res[0]} | {res[1]} |")

pr("\n## (d) Средняя точка со сглаживанием, разное число подшагов nsub (Ван дер Поль)\n")
pr("| nsub | P2 | P4C | P6C |")
pr("|---|---|---|---|")
for n in [1, 2, 3, 4]:
    b = GraggMidpoint(n)
    cells = []
    for sn in ["P2", "P4C", "P6C"]:
        e = [np.max(np.abs(integrate(p, b, SCHEMES[sn], N)["y"] - ref)) for N in [40, 80, 160, 320]]
        cells.append(slopes(e))
    pr(f"| {n} | " + " | ".join(cells) + " |")

with open(os.path.join(OUT, "diagnostics.md"), "w") as fh:
    fh.write("\n".join(out) + "\n")
