"""Ревью п.1: опорные методы по определениям.

(a) дефект симметричности ||Φ_{-h}(t+h, Φ_h(t, y0)) - y0|| в mpmath (dps=60), 6 задач, вкл. неавтономные;
(b) порядок локальной ошибки каждого метода ||Φ_h(y0) - φ_h(y0)|| (ожидается h^3);
(c) KD: совпадение с трапециями для скалярной задачи, совпадение с Верле (drift-kick-drift) для
    разделимой системы; проверка, что D-полушаг = обращение Эйлера-Кромера с шагом -h/2.
Запуск: python r1_base_methods.py  -> out/r1_base_methods.txt
"""
import os, sys
import numpy as np
import mpmath as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rv_common as R

mp.mp.dps = 60
B = R.MP
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
lines = []
pr = lambda s="": (print(s, flush=True), lines.append(s))

probs = [R.vdp(), R.lotka(), R.duffing(), R.kepler(), R.lorenz(), R.generic3()]
bases = [R.Trapezoid(), R.KD(0.5), R.KD(0.3), R.Gragg(1), R.Gragg(2), R.Gragg(3), R.Gragg(4), R.ExplicitMidpoint()]
H = [mp.mpf(1) / 10 * mp.mpf(2) ** (-k) for k in range(6)]
if True:
    # Лоренц быстрый -> меньшие шаги
    HL = [h / 8 for h in H]


def slopes(e):
    return [float(mp.log(e[i] / e[i + 1], 2)) if e[i + 1] > 0 and e[i] > 0 else float("nan") for i in range(len(e) - 1)]


pr("## (a) Дефект симметричности ||Φ_{-h}∘Φ_h(y0) − y0||, mpmath dps=60, h = 0.1·2^-k (Лоренц: /8)\n")
pr("Последний наклон log2 (ожидание: симметричный метод -> дефект ~1e-55, т.е. 0)\n")
pr("| метод | " + " | ".join(p.name for p in probs) + " |")
pr("|---|" + "---|" * len(probs))
for b in bases:
    cells = []
    for P in probs:
        hs = HL if P.name == "lorenz" else H
        y0 = B.vec(P.y0)
        t0 = mp.mpf(P.t0)
        d = []
        for h in hs:
            y1 = b.step(B, P, t0, y0, h)
            y2 = b.step(B, P, t0 + h, y1, -h)
            d.append(B.norm(y2 - y0))
        s = slopes(d)
        if max(d) < mp.mpf(10) ** -50:
            cells.append(f"0 ({mp.nstr(max(d), 2)})")
        else:
            cells.append(f"{mp.nstr(d[-1], 2)}, наклон {s[-1]:.2f}")
    pr(f"| {b.name} | " + " | ".join(cells) + " |")

pr("\n## (b) Порядок локальной ошибки ||Φ_h(y0) − φ_h(y0)|| (наклон log2 по последним шагам; p+1)\n")
pr("| метод | " + " | ".join(p.name for p in probs) + " |")
pr("|---|" + "---|" * len(probs))
flows = {}
for P in probs:
    hs = HL if P.name == "lorenz" else H
    F = lambda t, y, P=P: [mp.re(v) for v in P.f(B, t, B.vec(y))]
    sol = mp.odefun(F, mp.mpf(P.t0), [mp.mpf(v) for v in P.y0])
    flows[P.name] = [B.vec(sol(mp.mpf(P.t0) + h)) for h in hs]
for b in bases:
    cells = []
    for P in probs:
        hs = HL if P.name == "lorenz" else H
        y0 = B.vec(P.y0)
        e = [B.norm(b.step(B, P, mp.mpf(P.t0), y0, h) - ex) for h, ex in zip(hs, flows[P.name])]
        cells.append(f"{slopes(e)[-1]:.2f}")
    pr(f"| {b.name} | " + " | ".join(cells) + " |")

pr("\n## (c) KD: контрольные тождества\n")
# скалярная задача: KD == трапеции (в т.ч. при комплексном шаге)
P = R.logistic()
for h in [mp.mpf("0.3"), mp.mpc("0.2", "-0.15")]:
    a = R.KD(0.5).step(B, P, 0, B.vec(P.y0), h)
    c = R.Trapezoid().step(B, P, 0, B.vec(P.y0), h)
    pr(f"- скалярная логистическая, h = {mp.nstr(h, 3)}: |KD − трапеции| = {mp.nstr(B.norm(a - c), 3)}")
# разделимая система (маятник): KD == Верле drift-kick-drift
pend = R.Prob("pend", lambda B, t, y: B.vec([y[1], -B.sin(y[0])]),
              lambda B, t, y: B.mat([[0, 1], [-B.cos(y[0]), 0]]), [1.0, 0.3], 1.0)
y0 = B.vec(pend.y0)
for h in [mp.mpf("0.3"), mp.mpc("0.2", "0.1")]:
    a = R.KD(0.5).step(B, pend, 0, y0, h)
    q, p = y0[0], y0[1]
    q = q + h / 2 * p
    p = p - h * mp.sin(q)
    q = q + h / 2 * p
    pr(f"- маятник, h = {mp.nstr(h, 3)}: |KD − Верле(drift-kick-drift)| = {mp.nstr(max(abs(a[0]-q), abs(a[1]-p)), 3)}")


# D-полушаг = (Эйлер-Кромер с шагом -h)^{-1}  (сопряжённость, утверждение конспекта)
class EC:
    def step(self, B, P, t, y, h):
        x = mp.matrix(y)
        for i in range(P.n):
            x[i] = x[i] + h * P.f(B, t, x)[i]
        return x


P = R.generic3()
y0 = B.vec(P.y0)
h = mp.mpf("0.1")
yD = R.KD(0.0).step(B, P, mp.mpf(P.t0), y0, h)  # s=0: чистый D-метод на [t0, t0+h], время t0+h
back = EC().step(B, P, mp.mpf(P.t0) + h, yD, -h)
pr(f"- generic3 (n=3): ||EC_(−h)(D_h(y0)) − y0|| = {mp.nstr(B.norm(back - y0), 3)}  (0 => D = сопряжённый к EC)")
open(os.path.join(OUT, "r1_base_methods.txt"), "w").write("\n".join(lines) + "\n")
