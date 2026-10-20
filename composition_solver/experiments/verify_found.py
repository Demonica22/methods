"""Эксперимент 4б: проверка найденных коэффициентов на реальных ОДУ.

Читает results/coeff_search.json, строит схемы и меряет порядок для
средней точки (2 подшага) и метода Хойна (= средняя точка с 1 подшагом,
несимметричный метод общего вида). Результат: results/coeff_verify.md, fig_found_order.png
"""
import sys, os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
from csolver import integrate, problems, SCHEMES
from csolver.compositions import Composition
from csolver.base_methods import GraggMidpoint
from csolver.order_conditions import achieved_order, K_MIDPOINT, K_GENERAL, K_SYMMETRIC

OUT = os.path.join(ROOT, "results")
res = json.load(open(os.path.join(OUT, "coeff_search.json")))
NS = sorted({int(round(2 ** (k / 2))) for k in range(6, 21)})
PROBS = [problems.van_der_pol(), problems.kepler(), problems.lotka_volterra(), problems.forced_duffing()]
BASES = {"midpoint": GraggMidpoint(2), "general": GraggMidpoint(1)}


def order_fit(hs, es):
    hs, es = np.asarray(hs), np.asarray(es)
    ok = (es > 3e-13) & (es < 1e-2)
    for i in range(len(es) - 1):
        if es[i] < 1e-8 and not es[i + 1] < 0.7 * es[i]:
            ok[i + 1:] = False
            break
    idx = np.where(ok)[0][-4:]
    return np.polyfit(np.log(hs[idx]), np.log(es[idx]), 1)[0] if len(idx) >= 2 else np.nan


out = ["# Вопрос 4: найденные коэффициенты и проверка порядка\n"]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for r in res:
    if not r["stages"]:
        out.append(f"- {r['base']}, p = {r['order']}, палиндром = {r['palindromic']}, "
                   f"проекция Re = {r['project_real']}: решение не найдено")
        continue
    g = np.array([a + 1j * b for a, b in r["best"]])
    tag = f"{r['base']} p{r['order']} {'pal' if r['palindromic'] else 'gen'} {'Re' if r['project_real'] else 'C'}"
    sch = Composition(tag, g, r["order"])
    base = BASES[r["base"]]
    K = K_MIDPOINT if r["base"] == "midpoint" else K_GENERAL
    model_q = achieved_order(g, K, project_real=r["project_real"])
    qs = []
    for p in PROBS:
        hs, es = [], []
        for N in NS:
            y = integrate(p, base, sch, N, project_real=r["project_real"])
            hs.append(p.T / N); es.append(np.max(np.abs(y["y"] - p.reference())))
        qs.append(order_fit(hs, es))
        if p.name == "vanderpol":
            ax = axs[0 if r["base"] == "midpoint" else 1]
            e = np.array(es); ok = e > 1e-15
            ax.loglog(np.array(hs)[ok], e[ok], "-o", ms=3, label=f"p={r['order']}, m={r['stages']}, "
                      f"{'палиндром' if r['palindromic'] else 'общий'}, {'Re' if r['project_real'] else 'без Re'}")
    out.append(f"\n## {r['base']}: порядок {r['order']}, {'палиндром' if r['palindromic'] else 'без симметрии'}, "
               f"{'с проекцией Re' if r['project_real'] else 'без проекции'}\n")
    out.append(f"- стадий: **{r['stages']}**, найдено различных решений: {r['n_solutions']}")
    out.append(f"- порядок по модели: {model_q}; измеренный (Ван дер Поль, Кеплер, Лотка–Вольтерра, Дуффинг): "
               + ", ".join(f"{q:.2f}" for q in qs))
    out.append(f"- min Re γ = {g.real.min():.4f}, max|γ| = {np.abs(g).max():.4f}")
    out.append("- γ = [" + ", ".join(f"{z.real:+.15f}{z.imag:+.15f}j" for z in g) + "]")
# эталон: стандартные схемы на тех же опорах
for b, ax in zip(["midpoint", "general"], axs):
    p = PROBS[0]
    for sn, ls in [("P4C", "--"), ("P6C", ":")]:
        es = [np.max(np.abs(integrate(p, BASES[b], SCHEMES[sn], N)["y"] - p.reference())) for N in NS]
        e = np.array(es); ok = e > 1e-15
        ax.loglog((p.T / np.array(NS))[ok], e[ok], ls, color="#8a8a85", lw=1, label=f"{sn} (тройной прыжок)")
    ax.set_title("средняя точка, 2 подшага" if b == "midpoint" else "метод Хойна (несимметричный)")
    ax.set_xlabel("шаг h"); ax.legend(fontsize=6, frameon=False)
axs[0].set_ylabel("ошибка, Ван дер Поль, t = 10")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_found_order.png"), dpi=130)
open(os.path.join(OUT, "coeff_verify.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
