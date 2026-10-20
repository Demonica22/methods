"""Ревью п.3: глобальный порядок в двойной точности (своя реализация) против results/orders.csv.

- эталон: точное решение или mpmath.odefun (ряд Тейлора, 30 знаков), а не DOP853;
  дополнительно оценивается погрешность их эталона DOP853(rtol=1e-14, atol=1e-16);
- порядок оценивается двумя способами: (i) как у автора (МНК по последним 4 точкам в [3e-13, 1e-2]
  с отсечкой плато), (ii) медиана локальных наклонов на «чистом» участке 1e-11 < err < 1e-3;
- отдельно: P3C, P6SC, P6C без проекции Re после шага (Re только в конце).
Запуск: python r4_global_order.py -> out/r4_global_order.txt, data/r4_global.json, fig_r4_*.png
"""
import os, sys, json, csv
from multiprocessing import Pool
import numpy as np
import mpmath as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rv_common as R

np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = R.SRC
NS = sorted({int(round(2 ** (k / 2))) for k in range(6, 22)})  # 8 ... 1448


def companion():
    sig, om = -0.2, 2.0
    A = [[0, 1], [-(sig ** 2 + om ** 2), 2 * sig]]
    return R.Prob("companion", lambda B, t, y: B.vec([y[1], A[1][0] * y[0] + A[1][1] * y[1]]),
                  lambda B, t, y: B.mat(A), [1.0, 0.0], 10.0)


def probs():
    return {p.name: p for p in [R.oscillator(), companion(), R.logistic(), R.vdp(), R.lotka(), R.kepler(),
                                R.lorenz(), R.duffing()]}


def ref(P):
    if P.name == "kepler":
        return np.array(P.y0)
    return R.reference(P, dps=30)


BASES = {"trapezoid": R.Trapezoid(), "KD": R.KD(0.5), "midpoint": R.Gragg(2)}


def fit_author(hs, errs):
    hs, errs = np.asarray(hs), np.asarray(errs)
    ok = (errs > 3e-13) & (errs < 1e-2)
    for i in range(len(errs) - 1):
        if errs[i] < 1e-8 and not errs[i + 1] < 0.7 * errs[i]:
            ok[i + 1:] = False
            break
    idx = np.where(ok)[0][-4:]
    return float(np.polyfit(np.log(hs[idx]), np.log(errs[idx]), 1)[0]) if len(idx) >= 2 else float("nan")


def fit_clean(hs, errs, lo=1e-11, hi=1e-3):
    hs, errs = np.asarray(hs), np.asarray(errs)
    s = []
    for i in range(len(hs) - 1):
        if lo < errs[i] < hi and lo < errs[i + 1] < hi:
            s.append(np.log(errs[i] / errs[i + 1]) / np.log(hs[i] / hs[i + 1]))
    return (float(np.median(s)), float(np.min(s)), float(np.max(s)), len(s)) if s else (np.nan, np.nan, np.nan, 0)


def job(args):
    pn, bn, sn, proj = args
    mp.mp.dps = 30
    P = probs()[pn]
    g = [complex(x) for x in R.schemes()[sn]]
    yr = ref(P)
    errs, hs = [], []
    for N in NS:
        y = R.integrate(R.NP, BASES[bn], P, g, N, project_real=proj)
        e = np.max(np.abs(y.real - yr)) / max(1.0, np.max(np.abs(yr)))
        errs.append(float(e) if np.isfinite(e) else 1e300)
        hs.append(P.T / N)
    return (pn, bn, sn, proj, hs, errs)


if __name__ == "__main__":
    mp.mp.dps = 30
    PR = probs()
    lines = []
    pr = lambda s="": (print(s, flush=True), lines.append(s))
    # эталоны
    pr("## Эталоны: погрешность DOP853(rtol=1e-14, atol=1e-16) автора относительно mpmath (30 знаков)\n")
    from scipy.integrate import solve_ivp
    for pn, P in PR.items():
        yr = ref(P)
        F = lambda t, y, P=P: np.real(P.f(R.NP, t, y.astype(complex)))
        sol = solve_ivp(F, (P.t0, P.t0 + P.T), np.array(P.y0, float), method="DOP853", rtol=1e-14, atol=1e-16)
        pr(f"- {pn}: |DOP853 − эталон| = {np.max(np.abs(sol.y[:, -1] - yr)):.1e}")
    SCH = ["P2", "P3C", "P4C", "P4R", "P6C", "P6R", "P6SC"]
    jobs = [(pn, bn, sn, True) for pn in PR for bn in BASES for sn in SCH]
    jobs += [(pn, bn, sn, False) for pn in ["vanderpol", "lotka", "duffing", "lorenz"] for bn in ("trapezoid", "KD")
             for sn in ["P3C", "P6C", "P6SC"]]
    with Pool(4) as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(HERE, "data", "r4_global.json"), "w"))
    theirs = {}
    for row in csv.DictReader(open(os.path.join(SRC, "results", "orders.csv"))):
        theirs[(row["problem"], row["base"], row["scheme"])] = float(row["observed_order"])
    pr("\n## Глобальный порядок с проекцией Re (своя реализация)\n")
    pr("В ячейке: автор (orders.csv) | мой МНК по методике автора | медиана локальных наклонов на участке 1e-11..1e-3 [мин..макс, число пар]\n")
    for bn in BASES:
        pr(f"\n### опора {bn}\n")
        pr("| задача | " + " | ".join(SCH) + " |")
        pr("|---|" + "---|" * len(SCH))
        for pn in PR:
            cells = []
            for sn in SCH:
                r = [x for x in res if x[:4] == (pn, bn, sn, True)][0]
                a = fit_author(r[4], r[5])
                c = fit_clean(r[4], r[5])
                cells.append(f"{theirs.get((pn, bn, sn), float('nan')):.2f} \\| {a:.2f} \\| {c[0]:.2f} [{c[1]:.1f}..{c[2]:.1f}, {c[3]}]")
            pr(f"| {pn} | " + " | ".join(cells) + " |")
    pr("\n## Без проекции Re после шага (Re берётся только в конце); медиана наклонов на участке 1e-11..1e-3\n")
    pr("| задача | опора | P3C с Re | P3C без | P6C с Re | P6C без | P6SC с Re | P6SC без |")
    pr("|---|---|---|---|---|---|---|---|")
    for pn in ["vanderpol", "lotka", "duffing", "lorenz"]:
        for bn in ("trapezoid", "KD"):
            cells = []
            for sn in ["P3C", "P6C", "P6SC"]:
                for proj in (True, False):
                    r = [x for x in res if x[:4] == (pn, bn, sn, proj)][0]
                    c = fit_clean(r[4], r[5])
                    cells.append(f"{c[0]:.2f}")
            pr(f"| {pn} | {bn} | " + " | ".join(cells) + " |")
    open(os.path.join(HERE, "out", "r4_global_order.txt"), "w").write("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    col = {"P2": "#2a78d6", "P3C": "#1baf7a", "P4C": "#eb6834", "P4R": "#eb6834", "P6C": "#e87ba4",
           "P6R": "#e87ba4", "P6SC": "#008300"}
    for ax, bn in zip(axs, BASES):
        for sn in SCH:
            r = [x for x in res if x[:4] == ("vanderpol", bn, sn, True)][0]
            e = np.array(r[5]); ok = (e > 0) & (e < 1)
            ax.loglog(np.array(r[4])[ok], e[ok], "--" if sn.endswith("R") else "-", color=col[sn], marker="o", ms=3, label=sn)
        ax.set_title(f"Ван дер Поль, опора {bn} (ревью)"); ax.set_xlabel("h")
        ax.axhspan(1e-17, 1e-11, color="#eee")
    axs[0].set_ylabel("отн. ошибка в t=T"); axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_r4_vdp.png"), dpi=120)
