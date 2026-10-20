"""Введение и вопрос 1: коэффициенты, симметрия опор, решения, сходимость, стоимость."""
import json
import numpy as np
from scipy.integrate import solve_ivp
from multiprocessing import Pool
import common as c
from common import plt, np

STD = ["P2", "P3C", "P4C", "P4R", "P6C", "P6R", "P6SC"]
NS = np.unique(np.round(np.geomspace(6, 2500, 26)).astype(int))


# ---------- 1. коэффициенты на комплексной плоскости ----------
def fig_coeffs():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 0.8]})
    ax = axs[0]
    for n in ["P3C", "P4C", "P4R", "P6C", "P6SC"]:
        g = c.scheme(n).gammas.astype(complex)
        ax.plot(g.real, g.imag, "-", color=c.COL[n], lw=0.8, alpha=0.5)
        ax.plot(g.real, g.imag, c.MRK[n], color=c.COL[n], ms=7, label=f"{n} ({len(g)} ст.)")
    ax.axvline(0, color="#898781", lw=0.8)
    ax.axhline(0, color="#898781", lw=0.8)
    ax.set_title("Симметричные схемы: стадии γᵢ")
    ax.set_xlabel("Re γ"); ax.set_ylabel("Im γ"); ax.legend(loc="lower left")
    ax.annotate("P4R: γ₂ ≈ −1.70\n(отрицательный шаг)", xy=(-1.70, 0), xytext=(-1.6, 0.17),
                arrowprops=dict(arrowstyle="->", color="#52514e"), fontsize=8)
    ax = axs[1]
    for n, k in [("midpoint_p6_pal_Re", "F1"), ("midpoint_p6_gen_Re", "F2"), ("midpoint_p6_pal_C", "F3"),
                 ("general_p6_gen_Re", "P6C"), ("general_p4_pal_Re", "P4C")]:
        g = c.scheme(n).gammas.astype(complex)
        ax.plot(g.real, g.imag, "-", color=c.COL[k], lw=0.8, alpha=0.5)
        ax.plot(g.real, g.imag, c.MRK[k], color=c.COL[k], ms=7, label=f"{n} ({len(g)} ст.)")
    ax.axvline(0, color="#898781", lw=0.8); ax.axhline(0, color="#898781", lw=0.8)
    ax.set_title("Найденные схемы для несимметричных опор (вопрос 4)")
    ax.set_xlabel("Re γ"); ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7.5, frameon=False)
    c.save(fig, "fig_coeffs.png")


# ---------- 2. дефект симметрии опор ----------
def fig_symdefect():
    p = c.problems.van_der_pol()
    y0 = np.array([2.0, 0.0], complex)
    hs = np.geomspace(0.4, 0.004, 14)
    meths = [("KD, s = 0.5: 0 (симметричен)", c.BASES["KD"], "P4C"),
             ("трапеции: 0 (симметричен)", c.BASES["trapezoid"], "P2"),
             ("KD, s = 0.3: ∝ h²", c.KD(0.3), "P4R"), ("Хойн (1 подшаг): ∝ h⁴", c.BASES["heun"], "P6SC"),
             ("Грэгг, 2 подшага: ∝ h⁶", c.BASES["midpoint"], "P6C"),
             ("Грэгг, 4 подшага: ∝ h⁶", c.GraggMidpoint(4), "P6R")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for lab, m, k in meths:
        d = [np.max(np.abs(m.step(p, -h, m.step(p, 0.0, y0, h), -h) - y0)) for h in hs]
        d = np.maximum(np.array(d), 1e-17)
        ax.loglog(hs, d, c.MRK[k] + ("--" if k == "P2" else "-"), color=c.COL[k], ms=5, label=lab)
    ax.set_xlabel("шаг h"); ax.set_ylabel("‖Φ₋ₕ(Φₕ(y₀)) − y₀‖")
    ax.set_title("Дефект симметрии опорных методов (Ван дер Поль)")
    ax.set_ylim(3e-18, 10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    c.save(fig, "fig_symdefect.png")


# ---------- 3. галерея решений ----------
def dense_ref(p, T=None, n=1500):
    T = p.T if T is None else T
    s = solve_ivp(lambda t, y: np.real(p.f(t, y)), (p.t0, p.t0 + T), p.y0, method="DOP853",
                  rtol=1e-13, atol=1e-14, dense_output=True)
    tt = np.linspace(p.t0, p.t0 + T, n)
    return s, tt, s.sol(tt)


def fig_gallery():
    cases = [(c.problems.van_der_pol(), "KD", "P6C", 40, "Ван дер Поль μ = 1: фазовый портрет", (0, 1)),
             (c.problems.kepler(), "trapezoid", "P6SC", 30, "Кеплер e = 0.5: орбита за период", (0, 1)),
             (c.problems.lotka_volterra(), "KD", "P6SC", 30, "Лотка–Вольтерра: фазовый портрет", (0, 1)),
             (c.problems.lorenz(), "trapezoid", "P6C", 40, "Лоренц, T = 1: проекция (x, z)", (0, 2)),
             (c.problems.forced_duffing(), "midpoint", "P4C", 40, "Дуффинг (неавтономная): x(t)", None),
             (c.problems.linear_companion(), "KD", "P4C", 40, "Линейная, форма Фробениуса: x(t)", None)]
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.2))
    for ax, (p, b, s, N, title, ij) in zip(axs.ravel(), cases):
        r = c.integrate(p, c.BASES[b], c.scheme(s), N, return_path=True)
        _, tt, Y = dense_ref(p)
        tn = np.linspace(p.t0, p.t0 + p.T, N + 1)
        if ij:
            ax.plot(Y[ij[0]], Y[ij[1]], "-", color="#898781", lw=1.2, label="эталон DOP853")
            ax.plot(r["path"][:, ij[0]], r["path"][:, ij[1]], "o", color=c.COL[s], ms=4,
                    label=f"{c.BASE_RU[b].split(' (')[0]} + {s}, N = {N}")
            ax.set_xlabel(["x", "y", "z"][ij[0]] if p.name == "lorenz" else "y₁")
            ax.set_ylabel(["x", "y", "z"][ij[1]] if p.name == "lorenz" else "y₂")
            if p.name == "kepler":
                ax.set_aspect("equal", adjustable="datalim")
        else:
            ax.plot(tt, Y[0], "-", color="#898781", lw=1.2, label="эталон DOP853")
            ax.plot(tn, r["path"][:, 0], "o", color=c.COL[s], ms=4,
                    label=f"{c.BASE_RU[b].split(' (')[0].split(' +')[0]} + {s}, N = {N}")
            ax.set_xlabel("t"); ax.set_ylabel("y₁")
        e = np.max(np.abs(r["y"] - p.reference()))
        ax.set_title(title, fontsize=10)
        ax.legend(loc="best", fontsize=7.5, title=f"ошибка в T: {e:.1e}", title_fontsize=7.5)
    fig.tight_layout()
    c.save(fig, "fig_gallery.png")


# ---------- 4. ошибка вдоль траектории ----------
def fig_err_time():
    p = c.problems.van_der_pol()
    s, _, _ = dense_ref(p)
    N = 100
    tn = np.linspace(0, p.T, N + 1)
    Yr = s.sol(tn).T
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, b in zip(axs, ["trapezoid", "KD", "midpoint"]):
        for n in ["P2", "P4C", "P4R", "P6C", "P6SC"]:
            r = c.integrate(p, c.BASES[b], c.scheme(n), N, return_path=True)
            e = np.maximum(np.max(np.abs(r["path"] - Yr), axis=1), 1e-16)
            ax.semilogy(tn[1:], e[1:], "-", color=c.COL[n], label=n)
        ax.set_title(f"{c.BASE_RU[b]}", fontsize=10)
        ax.set_xlabel("t")
    axs[0].set_ylabel("‖y_n − y(t_n)‖")
    axs[0].legend(ncol=2)
    fig.suptitle("Ван дер Поль, N = 100 шагов (h = 0.1): глобальная ошибка вдоль траектории", fontsize=11)
    fig.tight_layout()
    c.save(fig, "fig_err_time.png")


# ---------- 5. сходимость (двойная точность) ----------
def _job(args):
    pname, b, n, pr = args
    p = getattr(c.problems, pname)()
    return args, c.conv(p, c.BASES[b], c.scheme(n), NS, project_real=pr).tolist()


def run_conv(pnames, bases, schemes, pr=True):
    jobs = [(pn, b, n, pr) for pn in pnames for b in bases for n in schemes]
    with Pool(4) as pool:
        res = pool.map(_job, jobs)
    return {f"{a[0]}|{a[1]}|{a[2]}|{a[3]}": v for a, v in res}


# асимптотический порядок по высокоточной проверке ревью (локальная ошибка, mpmath)
ORD = {"sym": {"P2": 2, "P3C": 4, "P4C": 4, "P4R": 4, "P6C": 6, "P6R": 6, "P6SC": 6},
       "midpoint": {"P2": 2, "P3C": 4, "P4C": 4, "P4R": 4, "P6C": 5, "P6R": 5, "P6SC": 5,
                    "midpoint_p6_pal_Re": 6, "midpoint_p6_gen_Re": 6, "midpoint_p6_pal_C": 6},
       "heun": {"P2": 2, "P3C": 3, "P4C": 3, "P4R": 3, "P6C": 3, "P6R": 3, "P6SC": 3,
                "general_p4_gen_Re": 4, "general_p4_pal_Re": 4, "general_p6_gen_Re": 6, "general_p6_pal_Re": 6}}
KEY = {"midpoint_p6_pal_Re": "F1", "midpoint_p6_gen_Re": "F2", "midpoint_p6_pal_C": "F3",
       "general_p4_gen_Re": "P4R", "general_p6_gen_Re": "F1", "general_p4_pal_Re": "F2", "general_p6_pal_Re": "F3"}


def plot_conv(data, pname, bases, schemes, fname, title, xcost=False, pr=True):
    fig, axs = plt.subplots(1, len(bases), figsize=(3.6 * len(bases) + 1.6, 3.8), sharey=True)
    axs = np.atleast_1d(axs)
    slopes = {}
    for ax, b in zip(axs, bases):
        od = ORD["sym"] if b in ("trapezoid", "KD") else ORD[b]
        for n in schemes:
            d = np.array(data[f"{pname}|{b}|{n}|{pr}"])
            if len(d) == 0:
                continue
            k = n if n in c.COL else KEY[n]
            slopes[f"{b}|{n}"] = c.slope(d[:, 0], d[:, 1])
            x = d[:, 2] if xcost else d[:, 0]
            ax.loglog(x, d[:, 1], c.MRK[k] + "-", color=c.COL[k], ms=4, lw=1.4,
                      label=f"{n}" + (f": порядок {od[n]}" if n in od else ""))
        ax.set_xlabel("число вычислений f" if xcost else "шаг h")
        ax.axhspan(1e-17, 3e-13, color="#e1e0d9", alpha=0.6, lw=0)
        ax.set_ylim(1e-16, 10)
        ax.set_title(c.BASE_RU[b], fontsize=10)
    axs[0].set_ylabel("ошибка в t = T")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8.5)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    c.save(fig, fname)
    return slopes


# ---------- 6. локальная ошибка в высокой точности (данные ревью) ----------
def fig_local_hp():
    d = json.load(open(c.os.path.join(c.ROOT, "review/data/r3_local.json")))
    hs = 0.4 * 2.0 ** -np.arange(3, 9)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, b in zip(axs, ["trapezoid", "KD", "gragg2"]):
        for prob, base, sch, errs, sl in d:
            if prob == "duffing" and base == b and sch in ["P2", "P3C", "P4C", "P4R", "P6C", "P6R", "P6SC*"]:
                k = sch.rstrip("*")
                ax.loglog(hs[:len(errs)], errs, c.MRK[k] + "-", color=c.COL[k], ms=4,
                          label=f"{k}: порядок {sl[-1]:.2f}")
        ax.set_title({"gragg2": c.BASE_RU["midpoint"]}.get(b, c.BASE_RU.get(b)), fontsize=10)
        ax.set_xlabel("шаг h"); ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.legend(fontsize=7, loc="lower right")
    axs[0].set_ylabel("локальная ошибка за 1 шаг")
    fig.suptitle("Локальная ошибка в 50-значной арифметике (ревью, Дуффинг): порядок по наклону локальной ошибки", fontsize=11)
    fig.tight_layout()
    c.save(fig, "fig_local_hp.png")


if __name__ == "__main__":
    import sys
    if "plots" not in sys.argv:
        fig_coeffs(); fig_symdefect(); fig_gallery(); fig_err_time(); fig_local_hp()
        data = run_conv(["van_der_pol", "forced_duffing", "kepler"], ["trapezoid", "KD", "midpoint"], STD)
        json.dump(data, open(c.os.path.join(c.DATA, "conv_q1.json"), "w"))
    data = json.load(open(c.os.path.join(c.DATA, "conv_q1.json")))
    sl = {}
    for pn, ru in [("van_der_pol", "Ван дер Поль"), ("forced_duffing", "Дуффинг (неавтономная)"),
                   ("kepler", "Кеплер e = 0.5")]:
        sl[pn] = plot_conv(data, pn, ["trapezoid", "KD", "midpoint"], STD, f"fig_conv_{pn}.png",
                           f"{ru}: глобальная ошибка от шага (серая полоса: уровень округления)")
    plot_conv(data, "van_der_pol", ["trapezoid", "KD", "midpoint"], STD, "fig_cost_van_der_pol.png",
              "Ван дер Поль: точность от стоимости (число вычислений f)", xcost=True)
    json.dump(sl, open(c.os.path.join(c.DATA, "slopes_q1.json"), "w"), indent=1)
