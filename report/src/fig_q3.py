"""Вопрос 3: A-устойчивость композиций, области устойчивости, контрпример Гаусса, демонстрация."""
import numpy as np
from scipy.linalg import expm
import common as c
from common import plt
from csolver.stability import amplification
from csolver.problems import Problem
from matplotlib.colors import TwoSlopeNorm

CM = "RdBu_r"


def region_panel(ax, base, sch, X, Y, k=1.0, title=None, lim=1.5):
    R = amplification(base, c.scheme(sch).gammas, X, Y, k=k)
    L = np.log10(np.clip(R, 1e-3, 1e3))
    im = ax.pcolormesh(X, Y, L, cmap=CM, norm=TwoSlopeNorm(0, -lim, lim), shading="auto", rasterized=True)
    ax.contour(X, Y, R, levels=[1.0], colors="#0b0b0b", linewidths=1.0)
    ax.axvline(0, color="#52514e", lw=0.6)
    ax.set_title(title or sch, fontsize=10)
    ax.grid(False)
    return im


def fig_trap_imag():
    y = np.geomspace(0.05, 200, 3000)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for n in ["P2", "P3C", "P4C", "P6C", "P6SC"]:
        r = amplification("trapezoid", c.scheme(n).gammas, 0 * y, y, k=1.0)
        ax.semilogx(y, r, "-", color=c.COL[n], label=f"{n}, с проекцией Re")
    for n in ["P4C", "P6C"]:
        r = amplification("trapezoid", c.scheme(n).gammas, 0 * y, y, k=1.0, project_real=False)
        ax.semilogx(y, r, "--", color=c.COL[n], lw=1.2, label=f"{n}, без проекции")
    ax.axhline(1, color="#0b0b0b", lw=0.8)
    for n, yy, v in [("P4C", 10.1, 1.018), ("P6C", 26.3, 1.404)]:
        ax.annotate(f"{v} при y ≈ {yy}", (yy, v), textcoords="offset points", xytext=(8, 4), fontsize=8,
                    color=c.COL[n])
    ax.set_xlabel("y  (z = i·y на мнимой оси)"); ax.set_ylabel("ρ(z)")
    ax.set_ylim(0.9, 2.5)
    ax.set_title("Трапеции: множитель перехода на мнимой оси. A-устойчивость требует ρ ≤ 1", fontsize=10.5)
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    c.save(fig, "fig_q3_trap_imag.png")


def fig_regions(base, schemes, xr, yr, fname, suptitle, k=1.0, titles=None, lim=1.5):
    X, Y = np.meshgrid(np.linspace(*xr, 520), np.linspace(*yr, 420))
    nc = 3
    nr = int(np.ceil(len(schemes) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(12, 3.5 * nr), sharex=True, sharey=True)
    for i, (ax, s) in enumerate(zip(axs.ravel(), schemes)):
        im = region_panel(ax, base, s, X, Y, k, (titles or {}).get(s), lim)
    for ax in axs[-1]:
        ax.set_xlabel("Re hλ")
    for ax in axs[:, 0]:
        ax.set_ylabel("Im hλ")
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.015)
    cb.set_label("log₁₀ ρ  (синий: устойчиво, красный: рост)")
    fig.suptitle(suptitle, fontsize=11)
    c.save(fig, fname)


def gauss3(z):
    return (1 + z / 2 + z ** 2 / 10 + z ** 3 / 120) / (1 - z / 2 + z ** 2 / 10 - z ** 3 / 120)


def trap(z):
    return (1 + z / 2) / (1 - z / 2)


def fig_gauss():
    g = c.scheme("P6SC").gammas
    X, Y = np.meshgrid(np.linspace(-3, 1, 500), np.linspace(0, 30, 600))
    Z = X + 1j * Y
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for ax, r, ttl in [(axs[0], trap, "трапеции ∘ P6SC: полюсы только справа"),
                       (axs[1], gauss3, "Гаусс-3 ∘ P6SC: полюс −0.41 + 20.04i (×)")]:
        R = np.ones_like(Z)
        for gi in g:
            R = R * r(gi * Z)
        L = np.log10(np.clip(np.abs(R), 1e-3, 1e3))
        im = ax.pcolormesh(X, Y, L, cmap=CM, norm=TwoSlopeNorm(0, -1.5, 1.5), shading="auto", rasterized=True)
        ax.contour(X, Y, np.abs(R), levels=[1.0], colors="#0b0b0b", linewidths=0.8)
        ax.axvline(0, color="#52514e", lw=0.8)
        ax.set_title(ttl, fontsize=9.5); ax.set_xlabel("Re z"); ax.grid(False)
    axs[0].set_ylabel("Im z")
    axs[1].plot(-0.4118, 20.0418, "x", color="#0b0b0b", ms=9, mew=2)
    fig.colorbar(im, ax=axs[:2], shrink=0.85, pad=0.01).set_label("log₁₀ |R(z)|")
    ax = axs[2]
    yy = np.linspace(0, 30, 3000)
    for r, lab, col, ls in [(trap, "трапеции, Re z = 0", c.COL["P4C"], "-"),
                            (gauss3, "Гаусс-3, Re z = 0", c.COL["P6C"], "--"),
                            (gauss3, "Гаусс-3, Re z = −0.41", c.COL["P6C"], "-")]:
        sh = -0.4122 if "−0.41" in lab else 0.0
        R = np.ones_like(yy, dtype=complex)
        for gi in g:
            R = R * r(gi * (sh + 1j * yy))
        ax.semilogy(yy, np.abs(R), ls, color=col, label=lab)
    ax.axhline(1, color="#0b0b0b", lw=0.8)
    ax.set_xlabel("Im z"); ax.set_ylabel("|R(z)|"); ax.legend(fontsize=8)
    ax.set_title("Проверка «только по мнимой оси» не видит рост", fontsize=9.5)
    fig.suptitle("Контрпример ревью к общему «достаточному условию»: опора с комплексными полюсами (Гаусс–Лежандр, 3 стадии)",
                 fontsize=10.5)
    c.save(fig, "fig_q3_gauss.png")


def fig_demo():
    sig, om = -0.5, 26.0
    A = np.array([[sig, om], [-om, sig]])
    g = lambda t: np.array([np.sin(t), np.cos(t)])
    dg = lambda t: np.array([np.cos(t), -np.sin(t)])
    y0 = np.array([0.5, 1.0])
    T = 30.0
    exact = lambda t: g(t) + expm(A * t) @ (y0 - g(0.0))
    p = Problem("pr", lambda t, y: A @ (y - g(t)) + dg(t), lambda t, y: A.astype(complex), y0, T,
                exact=lambda: exact(T))
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.9))
    tt = np.linspace(0, T, 3000)
    ex = np.array([exact(t) for t in tt])
    for ax, h in zip(axs, [1.0, 0.5]):
        N = int(T / h); tn = np.linspace(0, T, N + 1)
        ax.plot(tt, ex[:, 0], color="#898781", lw=1, label="точное y₁(t)")
        for n in ["P6SC", "P6C", "P4C"]:
            with np.errstate(all="ignore"):
                r = c.integrate(p, c.BASES["trapezoid"], c.scheme(n), N, return_path=True)
            ax.plot(tn, np.clip(r["path"][:, 0], -3, 3), c.MRK[n] + "-", color=c.COL[n], ms=4, lw=1.1,
                    label=f"трапеции + {n}")
        ax.set_ylim(-3, 3); ax.set_xlabel("t")
        ax.set_title(f"h = {h}: hλ = {sig*h:+.2f} ± {om*h:.0f}i", fontsize=10)
    axs[0].set_ylabel("y₁"); axs[0].legend(fontsize=7.5, loc="lower left", ncol=2)
    fig.suptitle("Работающий случай A-устойчивости: y′ = A(y − g(t)) + g′(t), собств. числа −0.5 ± 26i, медленное решение g = (sin t, cos t)",
                 fontsize=10.5)
    fig.tight_layout()
    c.save(fig, "fig_q3_demo.png")


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    fig_trap_imag()
    fig_regions("trapezoid", ["P2", "P3C", "P6SC", "P4C", "P6C", "P4R"], (-3, 1.5), (-40, 40),
                "fig_q3_trap_regions.png",
                "Трапеции + композиции (с проекцией Re) у мнимой оси: красное слева от Re = 0 означает потерю A-устойчивости",
                lim=0.25,
                titles={"P2": "P2: A-устойчива", "P3C": "P3C: A-устойчива", "P6SC": "P6SC: A-устойчива",
                        "P4C": "P4C: ρ до 1.018 у оси (A(89.5°))", "P6C": "P6C: ρ до 1.40 у оси (A(87.4°))",
                        "P4R": "P4R: полюс в hλ ≈ −1.17"})
    fig_regions("KD", ["P2", "P4C", "P6C", "P6SC", "P4R", "P6R"], (-17, 2), (-14, 14),
                "fig_q3_kd_regions.png", "KD (s = 0.5), тестовая матрица в форме Фробениуса (k = 0)", k=0.0)
    fig_regions("midpoint", ["P2", "P4C", "P6C", "P6R", "midpoint_p6_gen_Re", "midpoint_p6_pal_Re"], (-26, 3), (-22, 22),
                "fig_q3_mid_regions.png", "Средняя точка Грэгга + сглаживание: области устойчивости композиций")
    fig_gauss(); fig_demo()
