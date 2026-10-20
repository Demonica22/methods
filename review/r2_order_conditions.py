"""Ревью п.2, п.6, п.7: точная проверка условий порядка всех коэффициентов (словами, без случайных матриц),
уточнение (полировка) найденных коэффициентов методом Гаусса-Ньютона в mpmath (dps=50),
ранг системы условий (число независимых уравнений) и размерность семейства решений.

Запуск: python r2_order_conditions.py -> out/r2_order_conditions.txt, data/polished.json
"""
import os, sys, json
import numpy as np
import mpmath as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rv_common as R
import rv_words as W

mp.mp.dps = 50
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
lines = []
pr = lambda s="": (print(s, flush=True), lines.append(s))

S = R.schemes()
FOUND = R.found_schemes()
MODEL_OF_BASE = {"midpoint": "gragg2", "general": "general"}

# ---------------------------------------------------------------- 1. моменты
pr("## 1. Моменты Σγ^k (встроенные схемы, коэффициенты пересчитаны независимо, dps=50)\n")
pr("| схема | стадий | Σγ | Σγ^3 | Σγ^5 | Σγ^6 | Σγ^7 | min Re γ | max|γ| |")
pr("|---|---|---|---|---|---|---|---|---|")
for n, g in S.items():
    m = [mp.fsum(x ** k for x in g) for k in (1, 3, 5, 6, 7)]
    pr(f"| {n} | {len(g)} | " + " | ".join(mp.nstr(v, 4) for v in m) +
       f" | {mp.nstr(min(mp.re(x) for x in g), 4)} | {mp.nstr(max(abs(x) for x in g), 4)} |")

# ---------------------------------------------------------------- 2. порядок по точным условиям
pr("\n## 2. Порядок по точным условиям (слова до степени 7; допуск 1e-12 к дефекту)\n")
pr("Модели опоры: sym = симметричная (K={3,5,7}), gragg2 (K={3,5,6,7}), general (K={3..7}).")
pr("Re = с проекцией на вещественную часть после шага, C = без проекции.\n")
pr("| схема | sym Re | sym C | gragg2 Re | gragg2 C | general Re | general C |")
pr("|---|---|---|---|---|---|---|")
allsch = dict(S)
for k, v in FOUND.items():
    allsch[k] = v["g"]
orders = {}
for n, g in allsch.items():
    cells = []
    for model in ("sym", "gragg2", "general"):
        for proj in (True, False):
            q, d = W.order_of(g, model, proj, tol=1e-12, use_mp=True)
            orders[(n, model, proj)] = q
            cells.append(str(q))
    pr(f"| {n} | " + " | ".join(cells) + " |")


# ---------------------------------------------------------------- 3. полировка
def gn_polish(g0, model, p, proj, it=30, dps_tol=40, structure=None):
    """Минимально-нормовый Гаусс-Ньютон по 2m вещественным параметрам."""
    ws = W.WordSet(W.MODELS[model], p)
    x = [mp.re(z) for z in g0] + [mp.im(z) for z in g0]
    m = len(g0)
    tog = lambda x: [mp.mpc(x[i], x[m + i]) for i in range(m)]
    res = lambda x: ws.residual_mp(tog(x), proj)
    r = res(x)
    r0 = max(abs(v) for v in r)
    eps = mp.mpf(10) ** -30
    rank = None
    for _ in range(it):
        nr = max(abs(v) for v in r)
        J = mp.matrix(len(r), 2 * m)
        for j in range(2 * m):
            xp = list(x); xp[j] += eps
            rp = res(xp)
            for i in range(len(r)):
                J[i, j] = (rp[i] - r[i]) / eps
        U, Sv, V = mp.svd_r(J)
        smax = max(Sv)
        rank = sum(1 for s in Sv if s > smax * mp.mpf(10) ** -18)
        if nr < mp.mpf(10) ** -dps_tol:
            break
        # шаг = -V^T S^+ U^T r
        Ut_r = [mp.fsum(U[i, k] * r[i] for i in range(len(r))) for k in range(len(Sv))]
        coef = [Ut_r[k] / Sv[k] if Sv[k] > smax * mp.mpf(10) ** -18 else 0 for k in range(len(Sv))]
        dx = [mp.fsum(V[k, j] * coef[k] for k in range(len(Sv))) for j in range(2 * m)]
        x = [x[j] - dx[j] for j in range(2 * m)]
        r = res(x)
    g = tog(x)
    dist = max(abs(a - b) for a, b in zip(g, g0))
    return g, r0, max(abs(v) for v in r), dist, rank, len(r)


pr("\n## 3. Уточнение заявленных коэффициентов (Гаусс-Ньютон, dps=50): существует ли точное решение рядом\n")
pr("r0 — невязка условий для опубликованных цифр; r* — после уточнения; Δ — сдвиг коэффициентов; "
   "ранг — число независимых вещественных уравнений; 2m — число вещественных неизвестных.\n")
pr("| схема | модель | p | Re | r0 | r* | Δ | ранг/2m | min Re γ* |")
pr("|---|---|---|---|---|---|---|---|---|")
cases = [("P6SC", S["P6SC"], "sym", 6, True), ("P3C", S["P3C"], "sym", 4, True)]
for k, v in FOUND.items():
    cases.append((k, v["g"], MODEL_OF_BASE[v["base"]], v["order"], v["proj"]))
polished = {}
for name, g0, model, p, proj in cases:
    g, r0, r1, dist, rank, nres = gn_polish(g0, model, p, proj)
    polished[name] = dict(model=model, p=p, proj=proj, g=[[mp.nstr(mp.re(z), 30), mp.nstr(mp.im(z), 30)] for z in g])
    pr(f"| {name} | {model} | {p} | {'да' if proj else 'нет'} | {mp.nstr(r0, 2)} | {mp.nstr(r1, 2)} | {mp.nstr(dist, 2)} | "
       f"{rank}/{2 * len(g)} | {mp.nstr(min(mp.re(z) for z in g), 4)} |")
json.dump(polished, open(os.path.join(HERE, "data", "polished.json"), "w"), indent=1)

pr("\nУточнённые P6SC (30 знаков):")
for a, b in polished["P6SC"]["g"]:
    pr(f"  {a} {'+' if not b.startswith('-') else ''}{b} i")
open(os.path.join(OUT, "r2_order_conditions.txt"), "w").write("\n".join(lines) + "\n")
