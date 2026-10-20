"""Эксперимент 1: порядок сходимости композиционных схем для трёх опорных методов.

Запуск:  python experiments/order_convergence.py
Результаты: results/errors.csv, results/orders.csv, results/fig_*.png, results/summary.md
"""
import sys, os, csv, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
from csolver import BASE_METHODS, SCHEMES, integrate, problems
from csolver.base_methods import Trapezoid, KD, GraggMidpoint

OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

NS = sorted({int(round(2 ** (k / 2))) for k in range(4, 23)})  # 4 ... 2048, шаг ×√2
ERR_LO, ERR_HI = 3e-13, 1e-2   # точки вне диапазона в оценку порядка не берутся
SCHEME_ORDER = ["P2", "P3C", "P4C", "P4R", "P6C", "P6R", "P6SC"]
STYLE = {  # цвет по семейству схем, сплошная = комплексные, пунктир = вещественные
    "P2": ("#2a78d6", "-", "o"), "P3C": ("#1baf7a", "-", "^"),
    "P4C": ("#eb6834", "-", "s"), "P4R": ("#eb6834", "--", "s"),
    "P6C": ("#e87ba4", "-", "D"), "P6R": ("#e87ba4", "--", "D"), "P6SC": ("#008300", "-", "v"),
}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e4e0", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.6, "lines.markersize": 4})


def err_of(problem, base, scheme, N, project_real=True):
    r = integrate(problem, base, scheme, N, project_real=project_real)
    ref = problem.reference()
    return np.max(np.abs(r["y"] - ref)) / max(1.0, np.max(np.abs(ref))), r["nf"], r["h"]


def observed_order(hs, errs):
    """Наклон МНК по последним (не более 4) точкам из диапазона [ERR_LO, ERR_HI]."""
    hs, errs = np.asarray(hs), np.asarray(errs)
    ok = (errs > ERR_LO) & (errs < ERR_HI)
    # отсечь плато ошибок округления: после первой точки с ошибкой < 1e-8,
    # за которой ошибка перестала убывать, точки не используются
    for i in range(len(errs) - 1):
        if errs[i] < 1e-8 and not errs[i + 1] < 0.7 * errs[i]:
            ok[i + 1:] = False
            break
    idx = np.where(ok)[0][-4:]
    if len(idx) < 2:
        return np.nan
    return np.polyfit(np.log(hs[idx]), np.log(errs[idx]), 1)[0]


def main():
    t0 = time.time()
    probs = [P() for P in problems.ALL]
    rows, orders = [], []
    data = {}
    for p in probs:
        p.reference()
        for bn, base in BASE_METHODS.items():
            for sn in SCHEME_ORDER:
                sch = SCHEMES[sn]
                hs, es, nfs = [], [], []
                for N in NS:
                    e, nf, h = err_of(p, base, sch, N)
                    hs.append(h); es.append(e); nfs.append(nf)
                    rows.append([p.name, bn, sn, N, h, e, nf])
                q = observed_order(hs, es)
                orders.append([p.name, bn, sn, sch.nominal_order, q])
                data[(p.name, bn, sn)] = (np.array(hs), np.array(es), np.array(nfs))
                print(f"{p.name:11s} {bn:10s} {sn:4s} порядок ≈ {q:5.2f}", flush=True)

    with open(os.path.join(OUT, "errors.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["problem", "base", "scheme", "N", "h", "error", "f_evals"])
        w.writerows(rows)
    with open(os.path.join(OUT, "orders.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["problem", "base", "scheme", "nominal_order", "observed_order"])
        w.writerows([[a, b, c, d, f"{e:.2f}"] for a, b, c, d, e in orders])

    # --- графики ошибка vs h: по одному файлу на задачу, три панели (опорные методы)
    for p in probs:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        for ax, bn in zip(axs, BASE_METHODS):
            for sn in SCHEME_ORDER:
                hs, es, _ = data[(p.name, bn, sn)]
                c, ls, m = STYLE[sn]
                ok = es > 1e-15
                ax.loglog(hs[ok], es[ok], ls, color=c, marker=m, label=SCHEMES[sn].name)
            for q, y in [(2, 3e-2), (4, 3e-4), (6, 3e-6)]:  # эталонные наклоны
                hh = np.array([p.T / 40, p.T / 640])
                ax.loglog(hh, y * (hh / hh[0]) ** q, ":", color="#8a8a85", lw=1)
                ax.text(hh[-1], y * (hh[-1] / hh[0]) ** q, f" h^{q}", color="#6b6b66", fontsize=7,
                        va="center")
            ax.set_title(bn); ax.set_xlabel("шаг h")
            ax.set_ylim(1e-15, 1e1)
        axs[0].set_ylabel("ошибка в t = T")
        axs[0].legend(fontsize=7, frameon=False, loc="lower right")
        fig.suptitle(p.descr, fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"fig_order_{p.name}.png"), dpi=130)
        plt.close(fig)

    # --- эффективность: ошибка vs число вычислений f (Ван дер Поль, Кеплер)
    for pn in ["vanderpol", "kepler"]:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        for ax, bn in zip(axs, BASE_METHODS):
            for sn in SCHEME_ORDER:
                hs, es, nfs = data[(pn, bn, sn)]
                c, ls, m = STYLE[sn]
                ok = es > 1e-15
                ax.loglog(nfs[ok], es[ok], ls, color=c, marker=m, label=SCHEMES[sn].name)
            ax.set_title(bn); ax.set_xlabel("вычислений f")
            ax.set_ylim(1e-15, 1e1)
        axs[0].set_ylabel("ошибка в t = T"); axs[0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Эффективность: {pn}", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig_cost_{pn}.png"), dpi=130)
        plt.close(fig)

    # --- сводная таблица наблюдаемых порядков (медиана по задачам)
    med = {}
    for bn in BASE_METHODS:
        for sn in SCHEME_ORDER:
            qs = [q for (pn, b, s, _, q) in orders if b == bn and s == sn and np.isfinite(q)]
            med[(bn, sn)] = np.median(qs) if qs else np.nan
    lines = ["| опорный метод | " + " | ".join(SCHEMES[s].name for s in SCHEME_ORDER) + " |",
             "|---|" + "---|" * len(SCHEME_ORDER)]
    for bn in BASE_METHODS:
        lines.append(f"| {bn} | " + " | ".join(f"{med[(bn, s)]:.2f}" for s in SCHEME_ORDER) + " |")
    lines.append("| *номинал (симм. опора)* | " +
                 " | ".join(str(SCHEMES[s].nominal_order) for s in SCHEME_ORDER) + " |")
    per = ["| задача | опора | " + " | ".join(SCHEME_ORDER) + " |", "|---|---|" + "---|" * len(SCHEME_ORDER)]
    for p in probs:
        for bn in BASE_METHODS:
            qs = {s: q for (pn, b, s, _, q) in orders if pn == p.name and b == bn}
            per.append(f"| {p.name} | {bn} | " + " | ".join(f"{qs[s]:.2f}" for s in SCHEME_ORDER) + " |")
    with open(os.path.join(OUT, "orders_table.md"), "w") as fh:
        fh.write("## Медиана наблюдаемого порядка по 8 задачам\n\n" + "\n".join(lines) +
                 "\n\n## По задачам\n\n" + "\n".join(per) + "\n")
    print("\n".join(lines))
    print(f"время: {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
