"""Проверка найденных схем на ОДУ: глобальная ошибка при постоянном шаге, явная средняя точка как опора.

Схемы: data/best.json (эта папка) + для сравнения стандартный P4C/P6C (тройной прыжок, строился
для симметричной опоры) и схемы general_* из composition_solver (общая несимметричная опора).
Результат: verify.md, fig_order.png, fig_stability.png
Запуск: python verify.py
"""
import os, sys, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "composition_solver"))
warnings.filterwarnings("ignore")
from csolver import integrate, problems, SCHEMES
from csolver.base_methods import BaseMethod
from csolver.compositions import Composition
import em_trees as E


class ExplicitMidpoint(BaseMethod):
    """Явная средняя точка (без сглаживания): y1 = y0 + h f(t0 + h/2, y0 + h/2 f(t0, y0))."""
    name = "explicit midpoint"
    symmetric = False

    def step(self, problem, t, y, h):
        k1 = problem.f(t, y)
        return y + h * problem.f(t + h / 2, y + h / 2 * k1)


BASE = ExplicitMidpoint()
NS = sorted({int(round(2 ** (k / 2))) for k in range(6, 23)})
PROBS = [problems.van_der_pol(), problems.kepler(), problems.lotka_volterra(), problems.forced_duffing()]


def order_fit(hs, es):
    hs, es = np.asarray(hs), np.asarray(es)
    ok = (es > 3e-13) & (es < 1e-2)
    for i in range(len(es) - 1):
        if es[i] < 1e-8 and not es[i + 1] < 0.7 * es[i]:
            ok[i + 1:] = False
            break
    idx = np.where(ok)[0][-4:]
    return np.polyfit(np.log(hs[idx]), np.log(es[idx]), 1)[0] if len(idx) >= 2 else np.nan


def conv(sch, proj):
    out = {}
    for p in PROBS:
        hs, es, nf = [], [], []
        ref = p.reference()
        for N in NS:
            y = integrate(p, BASE, sch, N, project_real=proj)
            hs.append(p.T / N); es.append(np.max(np.abs(y["y"] - ref))); nf.append(y["nf"])
        out[p.name] = (hs, es, nf, order_fit(hs, es))
    return out


best = json.load(open(os.path.join(HERE, "data", "best.json")))
rows = []
for b in best:
    g = np.array([complex(float(a), float(c)) for a, c in b["gammas"]])
    tag = f"EM p{b['order']} {b['struct']} {'Re' if b['project_real'] else 'C'} ({b['stages']} ст.)"
    rows.append((tag, Composition(tag, g, b["order"]), b["project_real"], b))
ref_schemes = [("P4C (тройной прыжок, для симм. опоры)", SCHEMES["P4C"], True, None),
               ("P6C (тройной прыжок, для симм. опоры)", SCHEMES["P6C"], True, None),
               ("P2 = сама явная средняя точка", SCHEMES["P2"], True, None)]
results = []
for tag, sch, proj, b in rows + ref_schemes:
    c = conv(sch, proj)
    q_tree = E.achieved_order(sch.gammas, 7, proj, tol=1e-9)
    results.append((tag, sch, proj, b, c, q_tree))
    print(tag, q_tree, [f"{c[k][3]:.2f}" for k in c], flush=True)

# ---- таблица
L = ["# Явная средняя точка: проверка найденных схем на ОДУ\n",
     "Глобальная ошибка в конце интервала, постоянный шаг, N = 8…2048 шагов. "
     "Наклон — МНК по последним 4 точкам до выхода на округление. "
     "«Порядок по деревьям» — по точным условиям на деревьях Бутчера до порядка 7.\n",
     "| схема | стадий | порядок по деревьям | " + " | ".join(p.name for p in PROBS) + " |",
     "|---|---|---|" + "---|" * len(PROBS)]
for tag, sch, proj, b, c, q in results:
    L.append(f"| {tag} | {sch.stages} | {q} | " + " | ".join(f"{c[p.name][3]:.2f}" for p in PROBS) + " |")
open(os.path.join(HERE, "verify.md"), "w").write("\n".join(L) + "\n")

# ---- рисунок сходимости и стоимости (Ван дер Поль, Кеплер)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
cols = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
for (tag, sch, proj, b, c, q), col in zip(results, cols):
    ls = "-o" if b else "--"
    for ax, pn in zip(axs, ["vanderpol", "kepler"]):
        hs, es, nf, _ = c[pn]
        e = np.array(es); ok = e > 1e-15
        ax.loglog(np.array(nf)[ok], e[ok], ls, ms=3, color=col, lw=1.2, label=tag)
for ax, t in zip(axs, ["Ван дер Поль (μ = 1)", "Кеплер (e = 0.5)"]):
    ax.set_title(t); ax.set_xlabel("число вычислений f"); ax.grid(alpha=.3, which="both")
axs[0].set_ylabel("ошибка в конце интервала")
axs[1].legend(fontsize=6.5, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_order.png"), dpi=130)

# ---- области устойчивости: y' = λy, R(z) = Π (1 + γz + (γz)²/2); с проекцией Re для вещественной
# системы множитель для собственного значения λ — многочлен с вещественными частями коэффициентов
sel = [r for r in results if r[3]]
fig, axs = plt.subplots(2, (len(sel) + 1) // 2, figsize=(2.6 * ((len(sel) + 1) // 2), 5.6), squeeze=False)
L.append("\n## Устойчивость (y' = λy)\n")
L.append("R(z) = Π(1 + γ_i z + (γ_i z)²/2); с проекцией Re у вещественной системы множитель — многочлен "
         "с вещественными частями коэффициентов. Интервалы нормированы на число вычислений f (2m), "
         "у самой явной средней точки отрицательный интервал 2/2 = 1.0.\n")
L.append("| схема | стадий | −интервал на Re z < 0 | то же / 2m | интервал на мнимой оси | то же / 2m |")
L.append("|---|---|---|---|---|---|")
for ax, (tag, sch, proj, b, c, q) in zip(axs.T.ravel() if False else axs.ravel(), sel):
    coef = np.array([1.0 + 0j])
    for gm in sch.gammas:
        coef = np.convolve(coef, [gm * gm / 2, gm, 1])     # numpy.polyval: старшие первыми
    if proj:
        coef = coef.real
    m = sch.stages
    x = np.linspace(-3.2 * m, 1.0 * m, 500); yy = np.linspace(-2.6 * m, 2.6 * m, 500)
    Z = x[None, :] + 1j * yy[:, None]
    R = np.abs(np.polyval(coef, Z))
    ax.contourf(x, yy, (R <= 1).astype(float), levels=[0.5, 1.5], colors=["#9ecae1"])
    ax.contour(x, yy, R, levels=[1], colors="k", linewidths=.8)
    ax.axhline(0, color="gray", lw=.5); ax.axvline(0, color="gray", lw=.5)
    ax.set_title(tag, fontsize=7); ax.set_aspect("equal"); ax.tick_params(labelsize=6)
    # интервалы устойчивости (до первого выхода |R| > 1 + 1e-9)
    def interval(dirn):
        r = np.linspace(0, 10 * m, 200001)[1:]
        bad = np.abs(np.polyval(coef, dirn * r)) > 1 + 1e-9
        return r[np.argmax(bad)] if bad.any() else np.inf
    a, bi = interval(-1.0), interval(1j)
    L.append(f"| {tag} | {m} | {a:.3f} | {a / (2 * m):.3f} | {bi:.3f} | {bi / (2 * m):.3f} |")
for ax in axs.ravel()[len(sel):]:
    ax.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_stability.png"), dpi=120)
open(os.path.join(HERE, "verify.md"), "w").write("\n".join(L) + "\n")
