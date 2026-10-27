"""Расчёты и рисунки для статьи об A-устойчивости композиций (paper/).

Все числа, которые приводятся в тексте статьи, считаются здесь и пишутся в
paper/data/numbers.json. Рисунки сохраняются в paper/figs/.
Пакет csolver используется без изменений.

Запуск из корня репозитория:  python3 paper/src/calc.py
"""
import json, os, sys
sys.dont_write_bytecode = True
import numpy as np
from scipy import ndimage, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "composition_solver"))
sys.path.insert(0, os.path.join(ROOT, "report", "src"))
import common as c                                           # noqa: E402  (уточнённые коэффициенты P3C, P6SC)
from csolver.stability import amplification                  # noqa: E402
from csolver.problems import Problem                         # noqa: E402

FIG = os.path.join(ROOT, "paper", "figs")
DATA = os.path.join(ROOT, "paper", "data")
os.makedirs(FIG, exist_ok=True); os.makedirs(DATA, exist_ok=True)

RC = {
    "font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "lines.linewidth": 1.4, "savefig.dpi": 200, "savefig.bbox": "tight",
    "mathtext.fontset": "dejavuserif", "axes.grid": False, "axes.facecolor": "white",
    "axes.spines.top": True, "axes.spines.right": True, "axes.edgecolor": "black",
    "xtick.color": "black", "ytick.color": "black",
}
plt.rcParams.update(RC)   # после common, который ставит свой стиль

NAMES = ["P2", "P3C", "P4C", "P4R", "P6C", "P6R", "P6SC"]
G = {n: np.asarray(c.scheme(n).gammas, dtype=complex) for n in NAMES}
OUT = {}


# ---------------------------------------------------------------- трапеции, аналитически
def r_trap(w):
    return (1 + w / 2) / (1 - w / 2)


def Rc(z, g, r=r_trap):
    out = np.ones_like(np.asarray(z, dtype=complex))
    for gi in g:
        out = out * r(gi * z)
    return out


def rho_trap(z, g, proj=True):
    """Спектральный радиус перехода для вещественной матрицы со с.ч. z, z̄."""
    if proj:
        return np.abs((Rc(z, g) + np.conj(Rc(np.conj(z), g))) / 2)
    return np.maximum(np.abs(Rc(z, g)), np.abs(Rc(np.conj(z), g)))


def sup_imag(g, proj):
    y = np.geomspace(1e-3, 1e6, 400001)
    v = rho_trap(1j * y, g, proj)
    i = int(np.argmax(v))
    if v[i] > 1 + 1e-12:   # уточнение максимума
        res = optimize.minimize_scalar(lambda t: -rho_trap(1j * np.exp(t), g, proj)[()],
                                       bracket=(np.log(y[max(i - 1, 0)]), np.log(y[i]), np.log(y[min(i + 1, len(y) - 1)])))
        return float(-res.fun), float(np.exp(res.x))
    return float(v.max()), float(y[i])


def alpha_deg(g, proj):
    """Наибольший α, при котором ρ ≤ 1 в секторе |arg(−z)| ≤ α (полюсов в C⁻ нет)."""
    rr = np.geomspace(1e-3, 1e6, 60001)

    def ok(a):
        z = rr * np.exp(1j * (np.pi - a))
        return np.all(rho_trap(z, g, proj) <= 1 + 1e-12) and np.all(rho_trap(np.conj(z), g, proj) <= 1 + 1e-12)
    if ok(np.pi / 2):
        return 90.0
    lo, hi = 0.0, np.pi / 2
    if not ok(1e-9):
        return 0.0
    for _ in range(50):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if ok(m) else (lo, m)
    return float(np.degrees(lo))


def poles_left(g):
    p = [2 / gi for gi in g if abs(gi) > 0]
    return sorted({round(z.real, 4) + 1j * round(z.imag, 4) for z in p if z.real < 0}, key=lambda z: z.real)


trap = {}
for n in NAMES:
    g = G[n]
    row = {"stages": len(g), "minRe": float(g.real.min()), "poles_left": [str(p) for p in poles_left(g)]}
    for proj in (True, False):
        key = "Re" if proj else "C"
        if row["poles_left"]:
            row[key] = {"sup": None, "y": None, "alpha": 0.0}
        else:
            s, y = sup_imag(g, proj)
            row[key] = {"sup": s, "y": y, "alpha": alpha_deg(g, proj)}
    trap[n] = row
OUT["trapezoid"] = trap

# множители стадий P4C при y = ±10.1 (без проекции)
yP = trap["P4C"]["C"]["y"]
fac = {f"{s:+.0f}": [float(abs(r_trap(gi * 1j * s * yP))) for gi in G["P4C"]] for s in (1, -1)}
OUT["P4C_factors"] = {"y": yP, "factors": fac,
                      "prod": {k: float(np.prod(v)) for k, v in fac.items()}}

# проверка: сеточный расчёт csolver совпадает с аналитическим
yy = np.linspace(0.1, 60, 2000)
OUT["check_csolver_vs_analytic"] = {n: float(np.max(np.abs(
    amplification("trapezoid", G[n], 0 * yy, yy, k=1.0) - rho_trap(1j * yy, G[n])))) for n in NAMES}

# «усы»: где граница ρ = 1 у P4C/P6C при |Im| = 40
whisk = {}
for n in ["P4C", "P6C"]:
    f = lambda x: rho_trap(np.array(x + 40j), G[n])[()] - 1
    whisk[n] = float(optimize.brentq(f, -5, -1e-9))
OUT["whisker_at_Im40"] = whisk


# ---------------------------------------------------------------- контрпример: Гаусс-3 ∘ P6SC
def r_gauss3(w):
    return (1 + w / 2 + w ** 2 / 10 + w ** 3 / 120) / (1 - w / 2 + w ** 2 / 10 - w ** 3 / 120)


wp = np.roots([-1 / 120, 1 / 10, -1 / 2, 1])
polesR = [complex(w / gi) for w in wp for gi in G["P6SC"]]
left = sorted([p for p in polesR if p.real < 0], key=lambda z: z.imag)
pz = left[-1]
# максимум |R| рядом с полюсом на вертикали Re z = Re pz чуть в стороне
zz = pz.real + 1j * np.linspace(pz.imag - 1, pz.imag + 1, 200001)
zz = zz[np.abs(zz - pz) > 0.02]
yI = np.linspace(-200, 200, 400001)
OUT["gauss3_P6SC"] = {
    "poles_r": [str(np.round(w, 4)) for w in wp],
    "arg_poles_deg": [float(np.degrees(abs(np.angle(w)))) for w in wp],
    "arg_gamma_deg": [float(np.degrees(np.angle(gi))) for gi in G["P6SC"]],
    "poles_R_left": [str(np.round(p, 4)) for p in left],
    "sup_imag": float(np.max(np.abs(Rc(1j * yI, G["P6SC"], r_gauss3)))),
    "R_at_dist_0.02": float(np.max(np.abs(Rc(zz, G["P6SC"], r_gauss3)))),
}


# ---------------------------------------------------------------- KD и Грэгг: габариты областей
def component(base, g, k, xr, yr, nx, ny):
    X, Y = np.meshgrid(np.linspace(*xr, nx), np.linspace(*yr, ny))
    with np.errstate(all="ignore"):
        R = amplification(base, g, X, Y, k=k)
    st = np.nan_to_num(R, nan=np.inf) <= 1
    lab, _ = ndimage.label(st)
    i0 = np.argmin(np.abs(Y[:, 0])); j0 = np.argmin(np.abs(X[0] + 0.05))
    L = lab[i0, j0]
    if L == 0:
        return None
    m = lab == L
    dA = (X[0, 1] - X[0, 0]) * (Y[1, 0] - Y[0, 0])
    other = int(st.sum() - m.sum())
    return {"re_min": float(X[m].min()), "im_max": float(np.abs(Y[m]).max()),
            "area": float(m.sum() * dA), "area_per_stage": float(m.sum() * dA / len(g)),
            "other_stable_cells": other, "n_components": int(lab.max()),
            "touches_left_edge": bool(m[:, 0].any())}


regions = {}
for k in (0.0, 1.0):
    for n in NAMES:
        regions[f"KD_k{k:g}_{n}"] = component("KD", G[n], k, (-40, 3), (-20, 20), 1721, 801)
for n in NAMES:
    regions[f"mid_{n}"] = component("midpoint", G[n], 1.0, (-30, 3), (-25, 25), 1321, 1001)
OUT["regions"] = regions

# ρ на мнимой оси для KD (k = 0, 1) и Грэгга: неограниченный рост
OUT["imag_axis_growth"] = {
    lab: [float(amplification(b, G["P2"], np.array([0.0]), np.array([y]), k=k)[0]) for y in (5, 50, 500)]
    for lab, b, k in [("KD_k0", "KD", 0.0), ("KD_k1", "KD", 1.0), ("midpoint", "midpoint", 1.0)]}


# ---------------------------------------------------------------- пример: быстрая мода −0.5 ± 26i
def demo_run():
    from scipy.linalg import expm
    sig, om = -0.5, 26.0
    A = np.array([[sig, om], [-om, sig]])
    gfun = lambda t: np.array([np.sin(t), np.cos(t)])
    dg = lambda t: np.array([np.cos(t), -np.sin(t)])
    y0 = np.array([0.5, 1.0]); T = 30.0
    exact = lambda t: gfun(t) + expm(A * t) @ (y0 - gfun(0.0))
    p = Problem("demo", lambda t, y: A @ (y - gfun(t)) + dg(t), lambda t, y: A.astype(complex), y0, T,
                exact=lambda: exact(T))
    res, paths = {}, {}
    for h in (1.0, 0.5):
        N = int(round(T / h))
        for n in ["P2", "P3C", "P4C", "P6C", "P6SC"]:
            with np.errstate(all="ignore"):
                r = c.integrate(p, c.BASES["trapezoid"], c.scheme(n), N, return_path=True)
            tn = np.linspace(0, T, N + 1)
            ex = np.array([exact(t) for t in tn])
            e = np.abs(r["path"] - ex).max(axis=1)
            res[f"h{h:g}_{n}"] = {"err_T": float(e[-1]), "err_max": float(np.nanmax(e))}
            paths[(h, n)] = (tn, r["path"][:, 0].real)
    tt = np.linspace(0, T, 3000)
    return res, paths, tt, np.array([exact(t)[0] for t in tt])


demo, demo_paths, demo_tt, demo_ex = demo_run()
OUT["demo"] = demo
# проекция Re в контрпримере: μ = (R(z) + conj R(z̄))/2 тоже имеет полюс
mu_g = np.abs((Rc(zz, G["P6SC"], r_gauss3) + np.conj(Rc(np.conj(zz), G["P6SC"], r_gauss3))) / 2)
OUT["gauss3_P6SC"]["mu_at_dist_0.02"] = float(mu_g.max())
# ρ в точке примера hλ = −0.5 + 26i (h = 1) и −0.25 + 13i (h = 0.5)
OUT["demo_rho"] = {f"{z}": {n: float(rho_trap(np.array(z), G[n])) for n in ["P2", "P3C", "P4C", "P6C", "P6SC"]}
                   for z in (-0.5 + 26j, -0.25 + 13j)}
# функция устойчивости Грэгга (2 подшага + сглаживание): сравнение с 1 + z + z²/2 + z³/8
zt = np.array([-1.3 + 0.7j, 0.4 - 2.1j])
OUT["gragg_poly_check"] = float(np.max(np.abs(
    amplification("midpoint", [1.0], zt.real, zt.imag, k=1.0, project_real=False)
    - np.maximum(np.abs(1 + zt + zt**2 / 2 + zt**3 / 8), np.abs(1 + zt.conj() + zt.conj()**2 / 2 + zt.conj()**3 / 8)))))
# P4R/P6R на трапециях: множество {Re z < 0, ρ > 1} локализовано около полюсов
loc = {}
X, Y = np.meshgrid(np.linspace(-1.5, -0.6, 1801), np.linspace(-0.3, 0.3, 1201))
Xw, Yw = np.meshgrid(np.linspace(-200, -0.001, 2000), np.linspace(-200, 200, 2001))
for n in ["P4R", "P6R"]:
    with np.errstate(all="ignore"):
        bad = rho_trap(X + 1j * Y, G[n]) > 1
        wide = rho_trap(Xw + 1j * Yw, G[n])
    far = (np.abs(Xw + 0.0 + 1j * Yw - 0) > 0) & ~((Xw > -1.5) & (Xw < -0.6) & (np.abs(Yw) < 0.3))
    loc[n] = {"re_min": float(X[bad].min()), "re_max": float(X[bad].max()), "im_max": float(np.abs(Y[bad]).max()),
              "area": float(bad.sum() * (X[0, 1] - X[0, 0]) * (Y[1, 0] - Y[0, 0])),
              "max_rho_outside_box": float(np.nanmax(wide[far]))}
OUT["real_schemes_unstable_set"] = loc
OUT["gammas"] = {n: [[float(g.real), float(g.imag)] for g in G[n]] for n in NAMES}
json.dump(OUT, open(os.path.join(DATA, "numbers.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: OUT[k] for k in ["trapezoid", "P4C_factors", "check_csolver_vs_analytic", "whisker_at_Im40",
                                      "gauss3_P6SC", "imag_axis_growth", "demo"]}, indent=1, ensure_ascii=False))
for k, v in regions.items():
    print(k, v)


# ================================================================ рисунки
GRAY = ListedColormap(plt.cm.gray(np.linspace(0.0, 0.88, 256)))   # как colormap gray в статье, без чисто белого
VIOL = "#c0392b"


def galchenko_panel(ax, X, Y, R, title, mark_violation=False, gamma=1.0):
    """Область устойчивости в стиле Galchenko et al.: заливка 1 − ρ в оттенках серого
    (чёрный: ρ ≈ 1, светлый: ρ ≈ 0), неустойчивая часть не закрашена.
    gamma < 1 растягивает шкалу около ρ = 1 (заливка (1 − ρ)^gamma)."""
    F = np.where(R <= 1, np.clip(1 - R, 0, 1) ** gamma, np.nan)
    ax.contourf(X, Y, F, levels=np.linspace(0, 1, 41), cmap=GRAY)
    ax.contour(X, Y, R, levels=[1.0], colors="k", linewidths=0.7)
    if mark_violation:
        bad = ((X < 0) & (R > 1 + 1e-9)).astype(float)
        if bad.any():
            ax.contourf(X, Y, bad, levels=[0.5, 1.5], colors=[VIOL], alpha=0.9)
    ax.axhline(0, color="0.55", lw=0.4); ax.axvline(0, color="0.55", lw=0.4)
    ax.set_title(title)


def fig_coeffs():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    sty = {"P3C": ("o", "#7d3c98"), "P4C": ("s", "#1f5fa8"), "P6C": ("^", "#d35400"),
           "P6SC": ("P", "#117a65"), "P4R": ("D", "#1f5fa8"), "P6R": ("v", "#d35400")}
    for n in ["P3C", "P4C", "P6C", "P6SC"]:
        m, col = sty[n]
        axs[0].plot(G[n].real, G[n].imag, m, color=col, ms=5, label=n, mfc="none" if n == "P6C" else col)
    axs[0].axvline(0, color="0.4", lw=0.6); axs[0].axhline(0, color="0.4", lw=0.6)
    axs[0].set_title("(a) комплексные схемы"); axs[0].set_xlabel(r"Re $\gamma_i$"); axs[0].set_ylabel(r"Im $\gamma_i$")
    axs[0].legend(loc="upper left"); axs[0].set_xlim(-0.05, 0.6)
    for n in ["P4R", "P6R"]:
        m, col = sty[n]
        axs[1].plot(G[n].real, 0 * G[n].real + (0.05 if n == "P4R" else -0.05), m, color=col, ms=5, label=n)
    axs[1].axvline(0, color="0.4", lw=0.6)
    axs[1].set_ylim(-0.5, 0.5); axs[1].set_yticks([])
    axs[1].set_title("(b) вещественные схемы (Йошида)"); axs[1].set_xlabel(r"$\gamma_i$")
    axs[1].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_coeffs.png")); plt.close(fig)


def fig_imag_axis():
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
    y = np.linspace(-40, 40, 4001)
    ax = axs[0]
    g = G["P4C"]
    ax.plot(y, np.abs(r_trap(g[0] * 1j * y)), color="#1f5fa8", label=r"$|r(\gamma_1 iy)|=|r(\gamma_3 iy)|$")
    ax.plot(y, np.abs(r_trap(g[1] * 1j * y)), color="#d35400", label=r"$|r(\gamma_2 iy)|$")
    ax.plot(y, np.abs(Rc(1j * y, g)), "k--", label=r"$|R(iy)|$, без проекции")
    ax.axhline(1, color="0.5", lw=0.6)
    ax.set_xlabel("y"); ax.set_title("(a) P4C: множители стадий, z = iy"); ax.legend(loc="upper left", fontsize=6.8)
    ax.set_ylim(0.4, 2.35)
    ax = axs[1]
    y = np.geomspace(0.1, 300, 3000)
    for n, col, ls in [("P4C", "#1f5fa8", "-"), ("P6C", "#d35400", "-")]:
        ax.semilogx(y, rho_trap(1j * y, G[n]), ls, color=col, label=f"{n}, проекция Re")
        ax.semilogx(y, rho_trap(1j * y, G[n], proj=False), "--", color=col, lw=1.0, label=f"{n}, без проекции")
    for n, col in [("P3C", "#7d3c98"), ("P6SC", "#117a65")]:
        ax.semilogx(y, rho_trap(1j * y, G[n]), ":", color=col, lw=1.6, label=f"{n}: ρ ≡ 1")
    ax.axhline(1, color="0.5", lw=0.6)
    ax.set_xlabel("y"); ax.set_ylabel(r"$\rho(iy)$"); ax.set_title("(b) спектральный радиус на мнимой оси")
    ax.legend(fontsize=6.5, loc="upper left"); ax.set_ylim(0.95, 2.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_imag_axis.png")); plt.close(fig)


def fig_trap_regions():
    X, Y = np.meshgrid(np.linspace(-3, 1.5, 541), np.linspace(-40, 40, 801))   # строка Im = 0 входит в сетку
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.0), sharex=True, sharey=True)
    for ax, n in zip(axs.ravel(), ["P2", "P3C", "P6SC", "P4C", "P6C", "P4R"]):
        with np.errstate(all="ignore"):
            R = amplification("trapezoid", G[n], X, Y, k=1.0)
        galchenko_panel(ax, X, Y, np.nan_to_num(R, nan=1e9), n, mark_violation=True, gamma=0.3)
    for ax in axs[-1]: ax.set_xlabel(r"Re $h\lambda$")
    for ax in axs[:, 0]: ax.set_ylabel(r"Im $h\lambda$")
    ax = axs[1, 2]
    ax.annotate(r"полюс $2/\gamma_2\approx-1.17$", xy=(-1.175, 0), xytext=(-2.9, 24), fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", lw=0.5),
                arrowprops=dict(arrowstyle="->", lw=0.7, color="white"))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_trap_regions.png")); plt.close(fig)


def fig_gauss():
    g = G["P6SC"]
    X, Y = np.meshgrid(np.linspace(-1.2, 0.4, 481), np.linspace(18.5, 21.5, 601))
    Z = X + 1j * Y
    with np.errstate(all="ignore"):
        R = np.abs(Rc(Z, g, r_gauss3))
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
    galchenko_panel(axs[0], X, Y, R, "(a) Гаусс-3 ∘ P6SC около полюса", mark_violation=True)
    axs[0].plot(pz.real, pz.imag, "x", color="k", ms=7, mew=1.5)
    axs[0].set_ylim(18.5, 21.5)
    axs[0].set_xlabel("Re z"); axs[0].set_ylabel("Im z")
    ax = axs[1]
    yy = np.linspace(0, 40, 4001)
    ax.semilogy(yy, np.abs(Rc(1j * yy, g, r_gauss3)), "k-", label="Re z = 0")
    ax.semilogy(yy, np.abs(Rc(pz.real + 0.002 + 1j * yy, g, r_gauss3)), color=VIOL, label=f"Re z = {pz.real + 0.002:.3f}")
    ax.axhline(1, color="0.5", lw=0.6)
    ax.set_xlabel("Im z"); ax.set_ylabel("|R(z)|"); ax.set_title("(b) |R| на двух вертикалях"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_gauss.png")); plt.close(fig)


def fig_regions(base, schemes, xr, yr, k, fname, nx=700, ny=561, figsize=(7.4, 5.0), rows=2):
    X, Y = np.meshgrid(np.linspace(*xr, nx), np.linspace(*yr, ny))
    cols = int(np.ceil(len(schemes) / rows))
    fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for ax, n in zip(axs.ravel(), schemes):
        with np.errstate(all="ignore"):
            R = amplification(base, G[n], X, Y, k=k)
        galchenko_panel(ax, X, Y, np.nan_to_num(R, nan=1e9), n)
    for ax in axs[-1]: ax.set_xlabel(r"Re $h\lambda$")
    for ax in axs[:, 0]: ax.set_ylabel(r"Im $h\lambda$")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)


def fig_demo():
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7), sharey=True)
    sty = {"P4C": ("s", "#1f5fa8"), "P6C": ("^", "#d35400"), "P6SC": ("o", "#117a65")}
    for ax, h in zip(axs, (1.0, 0.5)):
        ax.plot(demo_tt, demo_ex, color="0.6", lw=0.9, label="точное")
        for n in ["P4C", "P6C", "P6SC"]:
            tn, yv = demo_paths[(h, n)]
            m, col = sty[n]
            ax.plot(tn, np.clip(yv, -3, 3), m + "-", color=col, ms=2.8, lw=0.8, label=n)
        ax.set_ylim(-3, 3); ax.set_xlabel("t"); ax.set_title(f"h = {h:g}:  hλ = {-0.5*h:g} ± {26*h:g}i")
    axs[0].set_ylabel(r"$y_1$"); axs[0].legend(ncol=2, loc="lower left", fontsize=6.8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig8_demo.png")); plt.close(fig)


fig_coeffs(); fig_imag_axis(); fig_trap_regions(); fig_gauss()
fig_regions("KD", ["P2", "P4C", "P6C", "P6SC", "P4R", "P6R"], (-17, 2), (-14, 14), 0.0, "fig5_kd_k0.png")
fig_regions("KD", ["P2", "P4C", "P4R", "P6R"], (-12, 2), (-8, 8), 1.0, "fig6_kd_k1.png", figsize=(7.4, 4.2), rows=2)
fig_regions("midpoint", ["P2", "P4C", "P6C", "P6SC", "P4R", "P6R"], (-26, 3), (-22, 22), 1.0, "fig7_mid.png")
fig_demo()
print("figures saved")
