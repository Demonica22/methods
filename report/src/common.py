"""Общие функции для графиков отчёта. Пакет csolver используется без изменений."""
import json, os, sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # корень репозитория
sys.path.insert(0, os.path.join(ROOT, "composition_solver"))
from csolver import integrate, BASE_METHODS, SCHEMES, problems           # noqa: E402
from csolver.compositions import Composition                             # noqa: E402
from csolver.base_methods import GraggMidpoint, KD, Trapezoid            # noqa: E402

FIG = os.path.join(ROOT, "report", "figs")
DATA = os.path.join(ROOT, "report", "data")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "legend.fontsize": 8.5, "lines.linewidth": 1.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "#fcfcfb",
    "savefig.dpi": 160, "savefig.bbox": "tight",
})

COL = {"P2": "#52514e", "P3C": "#e87ba4", "P4C": "#2a78d6", "P4R": "#1baf7a",
       "P6C": "#eb6834", "P6R": "#eda100", "P6SC": "#6250d6",
       "F1": "#e34948", "F2": "#008300", "F3": "#1c5cab", "REF": "#0b0b0b"}
MRK = {"P2": "o", "P3C": "v", "P4C": "s", "P4R": "D", "P6C": "^", "P6R": "x", "P6SC": "P",
       "F1": "*", "F2": "h", "F3": "<"}

# уточнённые коэффициенты из ревью (review/data/polished.json)
_pol = json.load(open(os.path.join(ROOT, "review", "data", "polished.json")))


def polished(name):
    return np.array([complex(float(a), float(b)) for a, b in _pol[name]["g"]])


def scheme(name):
    if name == "P6SC":
        return Composition("P6SC", polished("P6SC"), 6)
    if name == "general_p4_pal_Re":   # точное решение из ревью: Re γ2 = 0
        g1 = 0.5 - 1j / (2 * np.sqrt(3))
        return Composition(name, [g1, 1j / np.sqrt(3), g1], 4)
    if name in _pol:
        return Composition(name, polished(name), _pol[name]["p"])
    return SCHEMES[name]


BASES = {"trapezoid": BASE_METHODS["trapezoid"], "KD": BASE_METHODS["KD"],
         "midpoint": BASE_METHODS["midpoint"], "heun": GraggMidpoint(1)}
BASE_RU = {"trapezoid": "трапеции", "KD": "KD (s = 0.5)", "midpoint": "средняя точка Грэгга + сглаживание",
           "heun": "метод Хойна"}


def err(problem, base, sch, N, project_real=True):
    r = integrate(problem, base, sch, int(N), project_real=project_real)
    ref = problem.reference()
    return float(np.max(np.abs(r["y"] - ref)) / max(1.0, np.max(np.abs(ref)))), r["nf"]


def conv(problem, base, sch, Ns, project_real=True, floor=3e-14):
    out = []
    for N in Ns:
        e, nf = err(problem, base, sch, N, project_real)
        if not np.isfinite(e) or e > 1e3:
            continue
        out.append((problem.T / N, e, nf))
        if e < floor:
            break
    return np.array(out)


def slope(hs, es, lo=1e-12, hi=1e-3):
    m = (es > lo) & (es < hi)
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(hs[m]), np.log(es[m]), 1)[0])


def ref_slopes(ax, h0, e0, orders, color="#898781"):
    hh = np.array([h0, h0 / 8])
    for p in orders:
        ax.loglog(hh, e0 * (hh / h0) ** p, ":", color=color, lw=1)
        ax.text(hh[1], e0 * (hh[1] / h0) ** p, f" h^{p}", color=color, fontsize=7, va="center")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("saved", name)
