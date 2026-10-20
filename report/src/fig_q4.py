"""Вопрос 4: коэффициенты для несимметричной (почти симметричной) опоры."""
import json
import numpy as np
from multiprocessing import Pool
import common as c
from common import plt
from fig_q1 import plot_conv, dense_ref, KEY, ORD

NS = np.unique(np.round(np.geomspace(6, 2500, 26)).astype(int))
MID = ["P2", "P4C", "P6C", "P6R", "midpoint_p6_gen_Re", "midpoint_p6_pal_Re", "midpoint_p6_pal_C"]
HEUN = ["P2", "P4C", "P6C", "general_p4_gen_Re", "general_p4_pal_Re", "general_p6_gen_Re", "general_p6_pal_Re"]


def _job(a):
    pn, b, n = a
    pr = not n.endswith("_C")          # схемы *_C рассчитаны без проекции Re после шага
    p = getattr(c.problems, pn)()
    return a, c.conv(p, c.BASES[b], c.scheme(n), NS, project_real=pr).tolist()


def run():
    jobs = [(pn, "midpoint", n) for pn in ["van_der_pol", "forced_duffing", "lotka_volterra"] for n in MID]
    jobs += [(pn, "heun", n) for pn in ["van_der_pol", "forced_duffing"] for n in HEUN]
    with Pool(4) as pool:
        res = pool.map(_job, jobs)
    return {f"{a[0]}|{a[1]}|{a[2]}|True": v for a, v in res}


def fig_local():
    loc = json.load(open(c.os.path.join(c.ROOT, "review/data/r3_local.json")))
    fnd = json.load(open(c.os.path.join(c.ROOT, "review/data/r3_local_found.json")))
    hs = 0.4 * 2.0 ** -np.arange(3, 9)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
    for ax, b, names in [(axs[0], "gragg2", ["P4C", "P6C", "P6R", "midpoint_p6_gen_Re*", "midpoint_p6_pal_Re*", "midpoint_p6_pal_C*"]),
                         (axs[1], "heun", ["P4C", "P6C", "general_p4_gen_Re*", "general_p4_pal_Re_exact", "general_p6_gen_Re*", "general_p6_pal_Re*"])]:
        for prob, base, sch, errs, sl in loc + fnd:
            if prob == "duffing" and base == b and sch in names:
                n = sch.rstrip("*").replace("_exact", "")
                k = n if n in c.COL else KEY[n]
                ax.loglog(hs[:len(errs)], errs, c.MRK[k] + "-", color=c.COL[k], ms=4,
                          label=f"{n}: порядок {sl[-1]:.2f}")
        ax.set_xlabel("шаг h"); ax.legend(fontsize=7.5, loc="lower right"); ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_title({"gragg2": c.BASE_RU["midpoint"], "heun": c.BASE_RU["heun"]}[b], fontsize=10)
    axs[0].set_ylabel("локальная ошибка за 1 шаг")
    fig.suptitle("Локальная ошибка в 50-значной арифметике (ревью, Дуффинг): порядок по наклону локальной ошибки", fontsize=11)
    fig.tight_layout()
    c.save(fig, "fig_q4_local_hp.png")


def fig_solution():
    p = c.problems.forced_duffing()
    s, tt, Y = dense_ref(p)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.9))
    ax = axs[0]
    N = 25
    r = c.integrate(p, c.BASES["midpoint"], c.scheme("midpoint_p6_gen_Re"), N, return_path=True)
    ax.plot(Y[0], Y[1], color="#898781", lw=1.2, label="эталон DOP853")
    ax.plot(r["path"][:, 0], r["path"][:, 1], "h", color=c.COL["F2"], ms=6,
            label=f"Грэгг + midpoint_p6_gen_Re, N = {N} (h = 0.4)")
    ax.set_xlabel("x"); ax.set_ylabel("x′"); ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    ax.set_title(f"Дуффинг, фазовая плоскость; ошибка в T: {np.max(np.abs(r['y'] - p.reference())):.1e}", fontsize=10)
    ax = axs[1]
    N = 200
    tn = np.linspace(0, p.T, N + 1)
    Yr = s.sol(tn).T
    for n, k in [("P4C", "P4C"), ("P6C", "P6C"), ("midpoint_p6_gen_Re", "F2"), ("midpoint_p6_pal_Re", "F1")]:
        r = c.integrate(p, c.BASES["midpoint"], c.scheme(n), N, return_path=True)
        e = np.maximum(np.max(np.abs(r["path"] - Yr), axis=1), 1e-17)
        ax.semilogy(tn[1:], e[1:], "-", color=c.COL[k], label=f"{n} ({c.scheme(n).stages} ст.)")
    ax.set_xlabel("t"); ax.set_ylabel("‖y_n − y(t_n)‖"); ax.legend(fontsize=7.5)
    ax.set_title("Ошибка вдоль траектории, N = 200 (h = 0.05), опора: Грэгг", fontsize=10)
    fig.tight_layout()
    c.save(fig, "fig_q4_solution.png")


if __name__ == "__main__":
    import sys, warnings; warnings.filterwarnings("ignore")
    fig_local(); fig_solution()
    if "plots" not in sys.argv:
        json.dump(run(), open(c.os.path.join(c.DATA, "conv_q4.json"), "w"))
    data = json.load(open(c.os.path.join(c.DATA, "conv_q4.json")))
    sl = {}
    for pn, ru in [("van_der_pol", "Ван дер Поль"), ("forced_duffing", "Дуффинг"), ("lotka_volterra", "Лотка–Вольтерра")]:
        sl[pn] = plot_conv(data, pn, ["midpoint"], MID, f"fig_q4_conv_mid_{pn}.png", f"{ru}: опора средняя точка Грэгга")
    for pn, ru in [("van_der_pol", "Ван дер Поль"), ("forced_duffing", "Дуффинг")]:
        sl[pn + "_heun"] = plot_conv(data, pn, ["heun"], HEUN, f"fig_q4_conv_heun_{pn}.png", f"{ru}: опора метод Хойна")
    plot_conv(data, "van_der_pol", ["midpoint"], MID, "fig_q4_cost_mid.png",
              "Ван дер Поль, Грэгг: точность от числа вычислений f", xcost=True)
    json.dump(sl, open(c.os.path.join(c.DATA, "slopes_q4.json"), "w"), indent=1)
