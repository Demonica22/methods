"""Эксперимент 3 (вопрос 3): области устойчивости композиций и A-устойчивость.

Запуск: python experiments/stability_regions.py
Результаты: results/stability_summary.md, results/fig_stab_*.png
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
from csolver import SCHEMES
from csolver.stability import amplification

OUT = os.path.join(ROOT, "results")
SCH = ["P2", "P3C", "P4C", "P4R", "P6C", "P6R", "P6SC"]
BASES = [("trapezoid", 1.0), ("KD", 1.0), ("KD", 0.0), ("midpoint", 1.0)]


def a_stability_table(project_real):
    """A-устойчивость и угол A(α) на полярной сетке левой полуплоскости, |z| до 1e5."""
    r = np.logspace(-3, 5, 1200)
    phi = np.linspace(0, np.pi / 2, 1441)
    R, P = np.meshgrid(r, phi)
    sig, om = -R * np.cos(P), R * np.sin(P)
    rows = []
    for b, k in BASES:
        for sn in SCH:
            rho = amplification(b, SCHEMES[sn].gammas, sig, om, k, project_real)
            bad = ~(rho <= 1 + 1e-9)
            badrow = bad.any(axis=1)
            alpha = np.degrees(phi[np.argmax(badrow)]) if badrow.any() else 90.0
            # максимум ρ на мнимой оси
            y = np.concatenate([np.linspace(1e-3, 60, 60001), np.logspace(np.log10(60), 5, 4000)])
            ri = amplification(b, SCHEMES[sn].gammas, 0 * y, y, k, project_real)
            i = np.nanargmax(np.where(np.isfinite(ri), ri, np.inf))
            rows.append((b, k, sn, not bad.any(), alpha, ri[i], y[i]))
    return rows


def plot_regions():
    lim = {"trapezoid": (-40, 10, 40), "KD": (-24, 4, 16), "midpoint": (-24, 4, 16)}
    for b, k in BASES:
        x0, x1, ym = lim[b]
        sig, om = np.meshgrid(np.linspace(x0, x1, 700), np.linspace(-ym, ym, 560))
        fig, axs = plt.subplots(2, 4, figsize=(14, 6.4), sharex=True, sharey=True)
        axs[1, 3].axis("off")
        for ax, sn in zip(axs.ravel(), SCH):
            rho = amplification(b, SCHEMES[sn].gammas, sig, om, k, True)
            stable = np.where(np.isfinite(rho), rho <= 1 + 1e-12, False)
            ax.contourf(sig, om, stable, levels=[0.5, 1.5], colors=["#9ec5f4"])
            ax.contour(sig, om, stable.astype(float), levels=[0.5], colors=["#1c5cab"], linewidths=0.8)
            ax.axhline(0, color="#8a8a85", lw=0.6); ax.axvline(0, color="#8a8a85", lw=0.6)
            ax.set_title(SCHEMES[sn].name, fontsize=9)
        for ax in axs[1]:
            ax.set_xlabel("Re hλ")
        for ax in axs[:, 0]:
            ax.set_ylabel("Im hλ")
        kk = f", асимметрия k = {k:g}" if b == "KD" else ""
        fig.suptitle(f"Области устойчивости ρ ≤ 1 (закрашено), опора {b}{kk}, с проекцией Re",
                     fontsize=10)
        fig.tight_layout()
        tag = f"{b}_k{int(k)}" if b == "KD" else b
        fig.savefig(os.path.join(OUT, f"fig_stab_{tag}.png"), dpi=120)
        plt.close(fig)


def main():
    lines = ["# Вопрос 3: A-устойчивость композиций\n",
             "Тестовая система 2x2 с собственными числами hλ = σ ± iω (для KD — с асимметрией k). "
             "A(α): наибольший угол сектора |arg(−hλ)| ≤ α, где ρ ≤ 1 (проверено до |hλ| = 1e5).\n"]
    for proj in (True, False):
        lines.append(f"\n## {'С проекцией Re после шага' if proj else 'Без проекции (комплексное решение)'}\n")
        lines.append("| опора | схема | A-устойчива | α, град | max ρ на мнимой оси | где (Im hλ) |")
        lines.append("|---|---|---|---|---|---|")
        for b, k, sn, ast, alpha, rmax, ymax in a_stability_table(proj):
            name = f"{b} (k={k:g})" if b == "KD" else b
            lines.append(f"| {name} | {sn} | {'да' if ast else 'нет'} | {alpha:.2f} | {rmax:.4g} | {ymax:.3g} |")
    open(os.path.join(OUT, "stability_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot_regions()


if __name__ == "__main__":
    main()
