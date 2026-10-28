"""Отбор лучших решений из data/search*.json, уточнение до 50 знаков, контроль деревьями порядка p+1.

Критерий отбора (в порядке важности): все Re γ > 0; наименьшая норма главного члена ошибки
(дефекты на деревьях порядка p+1); затем наименьший max|γ|.
Результат: data/best.json (50 знаков строками), coeffs.md.
Запуск: python select_polish.py
"""
import os, json
import numpy as np
import mpmath as mp
import em_trees as E
import polish as P

HERE = os.path.dirname(os.path.abspath(__file__))
res = []
for fn in ("search.json", "search2.json"):
    path = os.path.join(HERE, "data", fn)
    if os.path.exists(path):
        res += json.load(open(path))

STRUCT_RU = {"gen": "без симметрии", "pal": "палиндром γ_{m+1-i} = γ_i",
             "conj": "симметрично-сопряжённая γ_{m+1-i} = conj γ_i"}


def lead_err(g, p, proj):
    """Норма главного члена локальной ошибки: дефекты Φ(τ) − 1/τ! на деревьях порядка p+1."""
    w = E.weights_np(g, p + 1)[-len(E.TREES[p + 1]):]
    t = E.targets(p + 1)[-len(E.TREES[p + 1]):]
    d = (w - t).real if proj else (w - t)
    return float(np.linalg.norm(d))


best, minimal = [], {}
for r in res:
    key = (r["order"], r["struct"], r["project_real"])
    if r["n_solutions"] and (key not in minimal or r["stages"] < minimal[key]["stages"]):
        minimal[key] = r
for key in sorted(minimal):
    r = minimal[key]
    p, struct, proj = key
    gs = [np.array([a + 1j * b for a, b in s]) for s in r["sols"]]
    gs.sort(key=lambda g: (np.min(g.real) <= 0, lead_err(g, p, proj), np.max(np.abs(g))))
    g0 = gs[0]
    gp, resid = P.polish(g0, struct, p, proj, dps=80)
    for _ in range(4):                       # у вырожденных решений сходимость медленнее — повторяем
        if resid < mp.mpf(10) ** -60:
            break
        gp, resid = P.polish(gp, struct, p, proj, dps=80)
    if (p, struct, proj) == (4, "pal", True):
        # решение лежит в особой точке (Re γ_2 = 0, якобиан вырожден); точный вид:
        # γ_1 = γ_3 = 1/2 + i/(2√3), γ_2 = −i/√3  (Σγ = 1, Re Σγ³ = 0, ...)
        a = mp.mpc(mp.mpf(1) / 2, 1 / (2 * mp.sqrt(3))) if gp[0].imag > 0 else mp.mpc(mp.mpf(1) / 2, -1 / (2 * mp.sqrt(3)))
        gp = [a, 1 - 2 * a, a]
    mp.mp.dps = 60
    gp = [mp.mpc(mp.mpf(mp.nstr(z.real, 50)), mp.mpf(mp.nstr(z.imag, 50))) for z in gp]  # то, что сохраняем
    w7 = E.weights_mp(gp, p + 1)
    t7 = [mp.mpf(1) / E.factorial(tr) for n in range(1, p + 2) for tr in E.TREES[n]]
    ntr = sum(len(E.TREES[n]) for n in range(1, p + 1))
    dd = [(a - b) for a, b in zip(w7, t7)]
    if proj:
        dd = [v.real for v in dd]
    defect_p = max(abs(v) for v in dd[:ntr])
    defect_p1 = max(abs(v) for v in dd[ntr:])
    gf = np.array([complex(z) for z in gp])
    best.append(dict(order=p, struct=struct, project_real=proj, stages=r["stages"],
                     n_solutions=r["n_solutions"], all_re_pos=bool(np.min(gf.real) > 0),
                     max_abs=float(np.max(np.abs(gf))), lead_err=lead_err(gf, p, proj),
                     defect=mp.nstr(defect_p, 3), defect_next=mp.nstr(defect_p1, 3),
                     gammas=[[mp.nstr(z.real, 50), mp.nstr(z.imag, 50)] for z in gp]))
    print(p, struct, proj, r["stages"], "дефект", mp.nstr(defect_p, 3), "| p+1:", mp.nstr(defect_p1, 3), flush=True)
json.dump(best, open(os.path.join(HERE, "data", "best.json"), "w"), indent=1, ensure_ascii=False)
