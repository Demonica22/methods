"""Воспроизведение 2D-экспериментов статьи
Galchenko et al., "New Approaches to Stability Analysis for Semi-Implicit
Numerical Integration Methods", Mathematics 2026, 14, 231.

Рис. 1   : область устойчивости явного Эйлера, k = 0, 0.5, 1.
Рис. 2   : область устойчивости CD (s = 0.5), k = 0 .. 1 шаг 0.125.
Рис. 3-5 : область устойчивости CD, s = 0 .. 1 шаг 0.125, k = 1, 0.5, 0.
Рис. 6-8 : движение корней F = (R - I)/h, k = 1, 0.5, 0.
Рис. 9   : движение корней в крупном масштабе, k = 1 и k = 0.

Формулы и параметры взяты из статьи: тестовая матрица (14)-(16) с r = -1,
матричная функция устойчивости по (23)-(25), h = 1 для областей,
оттенки серого по 1 - max|eig R| (как contourf(..., 1-Rh) + colormap gray
в приложении A). Движение корней повторяет код приложения A дословно.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
R_COEF = -1.0  # "non-influencing coefficient" r из приложения A


def test_matrix(sig, om, k, r=R_COEF):
    """Элементы a, b, c, d тестовой матрицы (14)-(16), как в приложении A."""
    sig = np.asarray(sig, float)
    om = np.asarray(om, float)
    d = 2 * sig / (1 + k)
    c = -np.sqrt(-1 / r * (sig**2 * (k - 1) ** 2 / (k + 1) ** 2 + om**2))
    b = r * c
    a = k * d
    return a, b, c, d


def R_cd(a, b, c, d, h, s):
    """Матрица R(h) одного шага CD с коэффициентом симметрии s, формулы (23), (25)."""
    shape = np.broadcast(a, b, c, d, h, s).shape
    r11, r12, r21, r22 = (np.ones(shape), np.zeros(shape), np.zeros(shape), np.ones(shape))
    h1 = s * h
    h2 = (1 - s) * h
    # полушаг Эйлера-Кромера (23)
    r11 = r11 + h1 * (a * r11 + b * r21)
    r12 = r12 + h1 * (a * r12 + b * r22)
    r21 = r21 + h1 * (c * r11 + d * r21)
    r22 = r22 + h1 * (c * r12 + d * r22)
    # полушаг D-метода в обратном порядке (25)
    with np.errstate(divide="ignore", invalid="ignore"):
        r21 = (r21 + h2 * c * r11) / (1 - h2 * d)
        r22 = (r22 + h2 * c * r12) / (1 - h2 * d)
        r11 = (r11 + h2 * b * r21) / (1 - h2 * a)
        r12 = (r12 + h2 * b * r22) / (1 - h2 * a)
    return np.stack([np.stack([r11, r12], -1), np.stack([r21, r22], -1)], -2)


def R_euler(a, b, c, d, h):
    """Явный Эйлер (22): R = I + hA."""
    A = np.stack([np.stack([a, b], -1), np.stack([c, d], -1)], -2)
    return np.eye(2) + h * A


def spectral_radius(R):
    with np.errstate(invalid="ignore", over="ignore"):
        bad = ~np.all(np.isfinite(R), axis=(-2, -1))
        Rs = np.where(bad[..., None, None], 0.0, R)
        m = np.max(np.abs(np.linalg.eigvals(Rs)), axis=-1)
    m[bad] = np.inf
    return m


def region(method, k, s, xlim, ylim, n=601):
    sp = np.linspace(*xlim, n)
    wp = np.linspace(*ylim, n)
    X, Y = np.meshgrid(sp, wp)
    a, b, c, d = test_matrix(X, Y, k)
    R = R_euler(a, b, c, d, 1.0) if method == "euler" else R_cd(a, b, c, d, 1.0, s)
    return X, Y, spectral_radius(R)


def draw_region(ax, X, Y, Rh):
    """Аналог contourf(X,Y,1-Rh,[1 0:0.01:1]); colormap gray: |R|>1 не закрашивается."""
    Z = 1 - Rh
    ax.contourf(X, Y, np.where(np.isfinite(Z), Z, -1), levels=np.linspace(0, 1, 101),
                cmap="gray", vmin=0, vmax=1)
    ax.grid(True, color="0.85", lw=0.5)
    ax.set_axisbelow(True)


def grid_figure(panels, fname, xlim, ylim, xticks=None, rot=0, title=None):
    fig, axs = plt.subplots(len(panels) // 3, 3, figsize=(9, 3.0 * len(panels) // 3 + 0.4),
                            squeeze=False)
    for ax, (label, X, Y, Rh) in zip(axs.flat, panels):
        draw_region(ax, X, Y, Rh)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_box_aspect(1)
        if xticks is not None:
            ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelrotation=rot)
        ax.set_xlabel(f"$Re$   {label}")
        ax.set_ylabel("$Im$")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- области
def fig1():
    xl, yl = (-2.2, 0.2), (-1.2, 1.2)
    panels = [(f"$k={k:g}$", *region("euler", k, None, xl, yl)) for k in (0, 0.5, 1)]
    grid_figure(panels, "fig01_euler_k.png", xl, yl,
                title="Рис. 1. Явный Эйлер, разные k")


def fig2():
    xl, yl = (-9, 0.3), (-2, 2)
    ks = np.arange(9) * 0.125
    panels = [(f"$k={k:g}$", *region("cd", k, 0.5, xl, yl)) for k in ks]
    grid_figure(panels, "fig02_cd_s05_k.png", xl, yl, xticks=[-8, -6, -4, -2, 0], rot=45,
                title="Рис. 2. CD, s = 0.5, разные k")


def fig3_5():
    ss = np.arange(9) * 0.125
    for num, k, xl, yl, xt in [(3, 1.0, (-10, 10), (-4, 4), [-10, 0, 10]),
                               (4, 0.5, (-25, 25), (-4, 4), [-20, 0, 20]),
                               (5, 0.0, (-5, 0.3), (-5, 5), [-4, -2, 0])]:
        panels = [(f"$s={s:g}$", *region("cd", k, s, xl, yl)) for s in ss]
        grid_figure(panels, f"fig{num:02d}_cd_k{k:g}_s.png", xl, yl, xticks=xt,
                    title=f"Рис. {num}. CD, k = {k:g}, разные s")


# ---------------------------------------------------------- движение корней
H_LIST = np.logspace(-3, -1, 20)
S_LIST = np.linspace(0, 1, 1001)
I_HALF = 500  # S_LIST[500] == 0.5 ровно


def root_paths(sig, om, k):
    """Траектории корня F=(R-I)/h по s для каждого h (приложение A).

    Как в MATLAB-коде: om > 0 -> lams(1), иначе lams(2). numpy.linalg.eig и MATLAB eig
    для вещественной несимметричной матрицы оба вызывают LAPACK dgeev, поэтому
    порядок собственных чисел совпадает.
    """
    a, b, c, d = test_matrix(sig, om, k)
    Hg, Sg = np.meshgrid(H_LIST, S_LIST, indexing="ij")
    R = R_cd(a, b, c, d, Hg, Sg)
    F = (R - np.eye(2)) / Hg[..., None, None]
    lams = np.linalg.eigvals(F)
    if om > 0:
        return [lams[..., 0]]
    if om < 0:
        return [lams[..., 1]]
    # om == 0: по коду приложения A берётся только lams(2), но на рис. 7-8 статьи дуги
    # у корней на вещественной оси симметричны относительно оси, т.е. нарисованы оба
    # собственных числа. Рисуем оба, чтобы совпасть с опубликованными рисунками.
    return [lams[..., 0], lams[..., 1]]  # каждый shape (len(H), len(S))


def draw_roots(ax, pts, k, lw=0.9, ms=1.5):
    cmap = plt.get_cmap("jet")
    for sig, om in pts:
        for lam in root_paths(sig, om, k):
            for j in range(len(H_LIST)):
                z = lam[j]
                xy = np.column_stack([z.real, z.imag])
                seg = np.stack([xy[:-1], xy[1:]], 1)
                lc = LineCollection(seg, cmap=cmap, norm=plt.Normalize(0, 1), lw=lw, zorder=3)
                lc.set_array(S_LIST[:-1])
                ax.add_collection(lc)
                ax.scatter(z.real[::25], z.imag[::25], c=S_LIST[::25], cmap=cmap,
                           vmin=0, vmax=1, s=ms, zorder=3)
                ax.plot(z.real[I_HALF], z.imag[I_HALF], "m*", ms=4, zorder=4)
        ax.plot(sig, om, "o", mfc="none", mec="r", mew=1.5, ms=5, zorder=5)


def roots_figure(num, k, xlim, ylim, pts, region_lim, fname, title, figsize=(10, 6.5)):
    fig, ax = plt.subplots(figsize=figsize)
    X, Y, Rh = region("cd", k, 0.5, region_lim[0], region_lim[1], n=801)
    draw_region(ax, X, Y, Rh)
    draw_roots(ax, pts, k)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("$Re$")
    ax.set_ylabel("$Im$")
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, label="коэффициент симметрии s", fraction=0.03, pad=0.02)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


GRID = [(sg, o) for sg in range(-3, 4) for o in range(-3, 4)]


def fig6_9(only9=False):
    if only9:
        return fig9()
    roots_figure(6, 1.0, (-3.5, 4.3), (-5, 5), GRID, ((-3.6, 0.0), (-5, 5)),
                 "fig06_roots_k1.png", "Рис. 6. Движение корней, k = 1")
    roots_figure(7, 0.5, (-3.7, 5.6), (-4.5, 5.1), GRID, ((-3.8, 0.0), (-5, 5)),
                 "fig07_roots_k05.png", "Рис. 7. Движение корней, k = 0.5")
    roots_figure(8, 0.0, (-5, 12), (-4.3, 4.3), GRID, ((-5, 0.0), (-4.3, 4.3)),
                 "fig08_roots_k0.png", "Рис. 8. Движение корней, k = 0")
    fig9()


def fig9():
    # Рис. 9a: корни -2 +- 1e-3 i (красные круги на рисунке статьи)
    roots_figure(9, 1.0, (-2.0, -1.7), (-4e-3, 4e-3), [(-2, 1e-3), (-2, -1e-3)],
                 ((-2.0, -1.7), (-4e-3, 4e-3)), "fig09a_roots_k1_zoom.png",
                 "Рис. 9a. Крупный масштаб, k = 1", figsize=(10, 5.5))
    # Рис. 9b: тестовые точки сняты с рисунка статьи по положению красных кругов.
    # Точки "на оси" взяты со сдвигом +-1e-3 по Im (как в тексте к рис. 7/9): на рисунке
    # статьи у -2 и -0.25 видна только нижняя ветвь, у -1 только верхняя.
    pts9b = [(-2, -1e-3), (-1.75, 0.2), (-1.5, -0.68), (-1.25, 0.6), (-1, 1e-3),
             (-0.75, -0.58), (-0.5, 0.2), (-0.25, -1e-3), (0, 0.6)]
    roots_figure(9, 0.0, (-3.1, 0.1), (-0.7, 0.65), pts9b, ((-3.1, 0.0), (-0.7, 0.65)),
                 "fig09b_roots_k0_zoom.png", "Рис. 9b. Крупный масштаб, k = 0",
                 figsize=(10, 5.5))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    import sys
    which = sys.argv[1:] or ["regions", "roots"]
    if "regions" in which:
        fig1(); fig2(); fig3_5()
    if "fig9" in which:
        fig9()
    if "roots" in which:
        fig6_9()
    print("done")
