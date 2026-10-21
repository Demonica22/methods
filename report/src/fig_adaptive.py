"""Графики и данные для отчёта по адаптивному шагу (report/adaptive_report.pdf).

Запуск из корня репозитория:  python3 report/src/fig_adaptive.py   (~6 мин на 4 ядрах)
Данные кэшируются в report/data/adaptive_*.json; удалите их, чтобы пересчитать.
"""
import json, os, warnings
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from common import FIG, DATA, BASES, integrate, SCHEMES, problems, save
from csolver import PAIRS, integrate_adaptive

warnings.filterwarnings("ignore")
FIG_AD = os.path.join(FIG, "adaptive")
os.makedirs(FIG_AD, exist_ok=True)

BASE_KEYS = ["trapezoid", "KD", "midpoint"]
PAIR_KEYS = ["P4C/P2", "P6C/P4C", "P6SC/P4C"]
PAIR_COL = {"P4C/P2": "#2a78d6", "P6C/P4C": "#eb6834", "P6SC/P4C": "#6250d6"}
TOLS = 10.0 ** -np.arange(3, 12.5, 0.5)
TOL_MIN = {"P4C/P2": 1e-10}      # ниже пара 2-го порядка делает слишком много шагов
NS = sorted({int(round(2 ** (k / 2))) for k in range(4, 25)})   # постоянный шаг: 4 … 4096
LEVELS = [1e-4, 1e-6, 1e-8, 1e-10]


def all_problems():
    ps = [P() for P in problems.ALL]
    k = problems.kepler(0.9); k.name, k.descr = "kepler_e09", "Кеплер, e = 0.9, один период"
    v = problems.van_der_pol(5.0); v.name, v.descr = "vanderpol_mu5", "Ван дер Поль, μ = 5, T = 10"
    return ps + [k, v]


PROBS = {p.name: p for p in all_problems()}
PROB_RU = {"oscillator": "осциллятор", "companion": "форма Фробениуса", "logistic": "логистическая",
           "vanderpol": "Ван дер Поль μ=1", "lotka": "Лотка–Вольтерра", "kepler": "Кеплер e=0.5",
           "lorenz": "Лоренц", "duffing": "Дуффинг", "kepler_e09": "Кеплер e=0.9",
           "vanderpol_mu5": "Ван дер Поль μ=5"}
PROB_COL = dict(zip(PROBS, ["#52514e", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#2a78d6",
                            "#b3261e", "#8a6d3b", "#6250d6", "#eb6834"]))


def rel_err(p, y):
    ref = p.reference()
    return float(np.max(np.abs(y - ref)) / max(1.0, np.max(np.abs(ref))))


# ------------------------------------------------------------------ расчёт
def job(args):
    pn, bn = args
    p, base = PROBS[pn], BASES[bn]
    p.reference()
    ad, fx = [], []
    for pa in PAIR_KEYS:
        for tol in TOLS[TOLS >= TOL_MIN.get(pa, 0)]:
            try:
                r = integrate_adaptive(p, base, pa, rtol=tol)
                ad.append(dict(problem=pn, base=bn, pair=pa, tol=tol, err=rel_err(p, r["y"]), nf=r["nf"],
                               acc=r["n_accepted"], rej=r["n_rejected"]))
            except RuntimeError:
                pass
        sch = PAIRS[pa].high
        for N in NS:
            r = integrate(p, base, sch, N)
            e = rel_err(p, r["y"])
            fx.append(dict(problem=pn, base=bn, scheme=pa.split("/")[0], N=N,
                           err=e if np.isfinite(e) else None, nf=r["nf"]))
    return ad, fx


def load_data():
    fa, ff = os.path.join(DATA, "adaptive_sweep.json"), os.path.join(DATA, "adaptive_fixed.json")
    if os.path.exists(fa) and os.path.exists(ff):
        return json.load(open(fa)), json.load(open(ff))
    with Pool(4) as pool:
        res = pool.map(job, [(pn, bn) for pn in PROBS for bn in BASE_KEYS])
    ad = [r for a, _ in res for r in a]
    fx = [r for _, f in res for r in f]
    json.dump(ad, open(fa, "w"), indent=0)
    json.dump(fx, open(ff, "w"), indent=0)
    return ad, fx


def nf_at(nf, e, eps):
    nf, e = np.asarray(nf, float), np.asarray([np.nan if x is None else x for x in e], float)
    ok = np.isfinite(e) & (e > 3e-14) & (e < 1)
    nf, e = nf[ok], e[ok]
    if len(e) < 2 or not (e.min() <= eps <= e.max()):
        return np.nan
    o = np.argsort(e)
    return float(np.exp(np.interp(np.log(eps), np.log(e[o]), np.log(nf[o]))))


def gains(ad, fx):
    out = {}
    for pn in PROBS:
        for bn in BASE_KEYS:
            for pa in PAIR_KEYS:
                a = [r for r in ad if r["problem"] == pn and r["base"] == bn and r["pair"] == pa]
                f = [r for r in fx if r["problem"] == pn and r["base"] == bn and r["scheme"] == pa.split("/")[0]]
                out[(pn, bn, pa)] = [nf_at([r["nf"] for r in f], [r["err"] for r in f], eps) /
                                     nf_at([r["nf"] for r in a], [r["err"] for r in a], eps) for eps in LEVELS]
    return out


# ------------------------------------------------------------------ графики
def fig_stages():
    """Узлы стадий в комплексной плоскости времени: у P4C и P6C на шаге h нет общих узлов."""
    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    for name, col, mk in [("P6C", "#eb6834", "o"), ("P6SC", "#6250d6", "D"), ("P4C", "#2a78d6", "s")]:
        g = SCHEMES[name].gammas
        z = np.concatenate([[0], np.cumsum(g)])
        ax.plot(z.real, z.imag, "-", color=col, marker=mk, ms=5, lw=1.4,
                label=f"{name}: {len(g)} стадий")
    ax.axhline(0, color="#898781", lw=0.8)
    ax.plot([0, 1], [0, 0], "o", color="#0b0b0b", ms=6)
    ax.text(0, -0.03, "t", ha="center", va="top"); ax.text(1, -0.03, "t + h", ha="center", va="top")
    ax.set_xlabel("Re (τ − t) / h"); ax.set_ylabel("Im (τ − t) / h")
    ax.set_aspect("equal"); ax.legend(loc="upper right", frameon=False)
    ax.set_title("Путь по комплексному времени за один шаг h")
    save(fig, "adaptive/fig_ad_stages.png")


def fig_tol(ad):
    fig, axs = plt.subplots(3, 3, figsize=(12, 10.2), sharex=True, sharey=True)
    for i, bn in enumerate(BASE_KEYS):
        for j, pa in enumerate(PAIR_KEYS):
            ax = axs[i, j]
            for pn in PROBS:
                rr = [r for r in ad if r["problem"] == pn and r["base"] == bn and r["pair"] == pa]
                ax.loglog([r["tol"] for r in rr], [max(r["err"], 1e-16) for r in rr], "-o",
                          color=PROB_COL[pn], ms=2.2, lw=1.1, label=PROB_RU[pn])
            ax.loglog(TOLS, TOLS, ":", color="#0b0b0b", lw=1)
            q = PAIRS[pa].q
            p = 5 if (bn == "midpoint" and pa != "P4C/P2") else (4 if pa == "P4C/P2" else 6)
            ax.loglog(TOLS, 1e-5 * (TOLS / 1e-3) ** (p / (q + 1)), "--", color="#898781", lw=1)
            ax.text(TOLS[-1], 1e-5 * (TOLS[-1] / 1e-3) ** (p / (q + 1)), f" tol^{p}/{q + 1}",
                    color="#52514e", fontsize=7, va="top")
            ax.set_title(f"{BASES[bn].name}, {pa}", fontsize=10)
            ax.set_ylim(1e-16, 1e-1)
            if i == 2: ax.set_xlabel("допуск tol (rtol = atol)")
            if j == 0: ax.set_ylabel("ошибка в t = T")
    axs[0, 0].legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "adaptive/fig_ad_tol.png")


def tol_table(ad):
    rows = []
    for bn in BASE_KEYS:
        for pa in PAIR_KEYS:
            sl, rat, rej = [], [], []
            for pn in PROBS:
                rr = [r for r in ad if r["problem"] == pn and r["base"] == bn and r["pair"] == pa]
                t = np.array([r["tol"] for r in rr]); e = np.array([r["err"] for r in rr])
                ok = np.isfinite(e) & (e > 1e-13)
                if ok.sum() >= 3:
                    sl.append(np.polyfit(np.log(t[ok]), np.log(e[ok]), 1)[0])
                rat += list(e[ok] / t[ok])
                rej += [r["rej"] / (r["acc"] + r["rej"]) for r in rr]
            rows.append(dict(base=bn, pair=pa, slope=float(np.median(sl)), ratio_med=float(np.median(rat)),
                             ratio_max=float(np.max(rat)), rej=float(100 * np.mean(rej)),
                             above=int(np.sum(np.array(rat) > 1))))
    return rows


def fig_ratio(ad):
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    pos, data, cols, labels = [], [], [], []
    for i, bn in enumerate(BASE_KEYS):
        for j, pa in enumerate(PAIR_KEYS):
            rr = [r for r in ad if r["base"] == bn and r["pair"] == pa and r["err"] > 1e-13]
            data.append(np.log10([r["err"] / r["tol"] for r in rr]))
            pos.append(i * 4 + j); cols.append(PAIR_COL[pa])
    bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", ms=3, mec="#898781"), medianprops=dict(color="#0b0b0b"))
    for b, c in zip(bp["boxes"], cols):
        b.set_facecolor(c); b.set_alpha(0.55)
    ax.axhline(0, color="#b3261e", lw=1, ls=":")
    ax.text(10.6, 0.15, "ошибка = tol", color="#b3261e", fontsize=8, ha="right")
    ax.set_xticks([1, 5, 9]); ax.set_xticklabels([BASES[b].name for b in BASE_KEYS])
    ax.set_ylabel("log10(ошибка / tol)")
    for pa in PAIR_KEYS:
        ax.bar(0, 0, color=PAIR_COL[pa], alpha=0.55, label=pa)
    ax.legend(frameon=False, ncol=3, loc="lower center")
    ax.set_ylim(-7, 2.5)
    save(fig, "adaptive/fig_ad_ratio.png")


def fig_cost(ad, fx, pn):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True)
    for ax, bn in zip(axs, BASE_KEYS):
        for pa in PAIR_KEYS:
            c = PAIR_COL[pa]
            f = [r for r in fx if r["problem"] == pn and r["base"] == bn and r["scheme"] == pa.split("/")[0]
                 and r["err"] is not None and 1e-15 < r["err"] < 10]
            a = [r for r in ad if r["problem"] == pn and r["base"] == bn and r["pair"] == pa and r["err"] > 1e-15]
            ax.loglog([r["nf"] for r in f], [r["err"] for r in f], "--", color=c, marker="s", ms=2.8, lw=1.2,
                      alpha=0.8, label=f"{pa.split('/')[0]}, постоянный шаг")
            ax.loglog([r["nf"] for r in a], [r["err"] for r in a], "-", color=c, marker="o", ms=3,
                      label=f"{pa}, адаптивный")
        ax.set_title(BASES[bn].name); ax.set_xlabel("вычислений f")
        ax.set_ylim(1e-14, 1e1)
    axs[0].set_ylabel("ошибка в t = T"); axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(PROBS[pn].descr, fontsize=10)
    fig.tight_layout()
    save(fig, f"adaptive/fig_ad_cost_{pn}.png")


def fig_gain(G, k=2):
    pnames = list(PROBS)
    cols = [(bn, pa) for bn in BASE_KEYS for pa in PAIR_KEYS]
    M = np.array([[G[(pn, bn, pa)][k] for bn, pa in cols] for pn in pnames])
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    L = np.log2(np.where(np.isfinite(M), M, np.nan))
    im = ax.imshow(L, cmap="RdBu", norm=TwoSlopeNorm(0, -2.5, 4), aspect="auto")
    for i in range(len(pnames)):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, "—" if not np.isfinite(v) else f"{v:.2g}×", ha="center", va="center", fontsize=7.5,
                    color="white" if np.isfinite(v) and (v > 6 or v < 0.3) else "#0b0b0b")
    ax.set_yticks(range(len(pnames))); ax.set_yticklabels([PROB_RU[p] for p in pnames])
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{BASES[b].name}\n{p}" for b, p in cols], fontsize=7.5)
    ax.grid(False)
    for x in [2.5, 5.5]:
        ax.axvline(x, color="white", lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_ticks(np.log2([0.25, 0.5, 1, 2, 4, 8, 16])); cb.set_ticklabels(["¼", "½", "1", "2", "4", "8", "16"])
    cb.set_label("выигрыш по f, раз")
    ax.set_title(f"Во сколько раз адаптивный шаг дешевле постоянного при ошибке {LEVELS[k]:.0e}", fontsize=10)
    save(fig, "adaptive/fig_ad_gain.png")


def fig_steps():
    fig, axs = plt.subplots(2, 2, figsize=(12, 5.6), sharex="col")
    for j, (pn, comp, lab) in enumerate([("kepler_e09", None, "расстояние r(t)"),
                                         ("vanderpol_mu5", 0, "x(t)")]):
        p = PROBS[pn]
        for pa in PAIR_KEYS:
            r = integrate_adaptive(p, BASES["KD"], pa, rtol=1e-8)
            dashed = pa == "P6C/P4C"
            axs[0, j].semilogy(r["t"][1:], r["h"], "--" if dashed else "-", color=PAIR_COL[pa],
                               lw=2.6 if dashed else 1.3,
                               label=f"{pa}: {r['n_accepted']} шагов, {r['n_rejected']} отброшено")
        path = integrate(p, BASES["KD"], SCHEMES["P6C"], 4000, return_path=True)["path"]
        tt = np.linspace(p.t0, p.t0 + p.T, len(path))
        y = np.hypot(path[:, 0], path[:, 1]) if comp is None else path[:, comp]
        axs[1, j].plot(tt, y, color="#0b0b0b", lw=1.4)
        axs[0, j].set_title(p.descr + ", KD, tol = 1e-8", fontsize=10)
        axs[0, j].set_ylim(1e-5, 0.5)
        axs[0, j].legend(fontsize=7.5, frameon=False, loc="lower center")
        axs[1, j].set_ylabel(lab); axs[1, j].set_xlabel("t")
    axs[0, 0].set_ylabel("шаг h")
    fig.tight_layout()
    save(fig, "adaptive/fig_ad_steps.png")


def fig_work(ad):
    """Структура затрат в стадиях опоры при tol = 1e-8, медиана по задачам и опорам."""
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    for i, pa in enumerate(PAIR_KEYS):
        sh, sl = PAIRS[pa].high.stages, PAIRS[pa].low.stages
        rr = [r for r in ad if r["pair"] == pa and abs(np.log10(r["tol"]) + 8) < 1e-9]
        parts = np.median([[r["acc"] * sh, r["acc"] * sl, r["rej"] * (sh + sl)] for r in rr], axis=0)
        tot = parts.sum()
        left = 0
        for v, c, lab in zip(parts, ["#52514e", "#eb6834", "#b3261e"],
                             ["решение (схема высокого порядка)", "оценщик", "отброшенные шаги"]):
            ax.barh(i, v / tot * 100, left=left, color=c, alpha=0.8, label=lab if i == 0 else None)
            if v / tot > 0.04:
                ax.text(left + v / tot * 50, i, f"{100 * v / tot:.0f}%", ha="center", va="center",
                        color="white", fontsize=8.5)
            left += v / tot * 100
    ax.set_yticks(range(3)); ax.set_yticklabels(PAIR_KEYS); ax.invert_yaxis()
    ax.set_xlabel("доля стадий опоры, %"); ax.set_xlim(0, 100); ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=8)
    save(fig, "adaptive/fig_ad_work.png")


def main():
    ad, fx = load_data()
    G = gains(ad, fx)
    fig_stages(); fig_tol(ad); fig_ratio(ad); fig_work(ad); fig_steps()
    for pn in ["kepler_e09", "vanderpol_mu5", "kepler", "vanderpol"]:
        fig_cost(ad, fx, pn)
    fig_gain(G, 2)
    T = tol_table(ad)
    summary = dict(tol=T, gains={"|".join(k): v for k, v in G.items()}, levels=LEVELS)
    json.dump(summary, open(os.path.join(DATA, "adaptive_summary.json"), "w"), indent=1)
    for r in T:
        print(r)
    for pn in PROBS:
        print(pn, {f"{b}|{p}": [f"{x:.2f}" for x in G[(pn, b, p)]] for b in BASE_KEYS for p in PAIR_KEYS})


if __name__ == "__main__":
    main()
