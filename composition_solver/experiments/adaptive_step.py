"""Эксперимент 3: адаптивный шаг с оценкой ошибки по вложенной паре композиций.

1) Допуск → достигнутая ошибка: все 8 задач, 3 опоры, пары P4C/P2, P6C/P4C, P6SC/P4C,
   rtol = atol = 1e-3 … 1e-12.
2) Стоимость: ошибка от числа вычислений f, адаптивный шаг против постоянного
   (та же схема высокого порядка), на задачах с неравномерной динамикой.
3) История шага h(t) на Кеплере e = 0.9.

Запуск:  python experiments/adaptive_step.py   (~3 мин)
Результаты: results/adaptive.csv, results/adaptive_summary.md, results/fig_adapt_*.png
"""
import sys, os, csv, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
from csolver import BASE_METHODS, SCHEMES, PAIRS, integrate, integrate_adaptive, problems

OUT = os.path.join(ROOT, "results")
TOLS = 10.0 ** -np.arange(3, 12.5, 0.5)
TOL_MIN = {"P4C/P2": 1e-10}  # у пары 2-го порядка ниже слишком много шагов
NS = sorted({int(round(2 ** (k / 2))) for k in range(4, 25)})  # постоянный шаг: 4 … 4096
PAIR_ORDER = ["P4C/P2", "P6C/P4C", "P6SC/P4C"]
PAIR_COLOR = {"P4C/P2": "#eb6834", "P6C/P4C": "#e87ba4", "P6SC/P4C": "#008300"}
PROB_COLOR = ["#2a78d6", "#1baf7a", "#eb6834", "#e87ba4", "#008300", "#8b5cf6", "#c0392b", "#6b6b66"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e4e0", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.6, "lines.markersize": 4})


def rel_err(p, y):
    ref = p.reference()
    return np.max(np.abs(y - ref)) / max(1.0, np.max(np.abs(ref)))


def hard_problems():
    """Задачи, где выгода от адаптивного шага ожидаема: быстрые и медленные участки."""
    k = problems.kepler(0.9)
    k.name, k.descr = "kepler_e09", "Кеплер, e = 0.9, один период"
    v = problems.van_der_pol(5.0)
    v.name, v.descr = "vanderpol_mu5", "Ван дер Поль, μ = 5, T = 10"
    return [k, v, problems.kepler(), problems.van_der_pol()]


def tol_sweep(probs):
    rows = []
    for p in probs:
        for bn, base in BASE_METHODS.items():
            for pn in PAIR_ORDER:
                for tol in TOLS[TOLS >= TOL_MIN.get(pn, 0)]:
                    try:
                        r = integrate_adaptive(p, base, pn, rtol=tol)
                        e = rel_err(p, r["y"])
                    except RuntimeError:
                        r, e = dict(nf=np.nan, n_accepted=np.nan, n_rejected=np.nan), np.nan
                    rows.append(dict(problem=p.name, base=bn, pair=pn, tol=tol, error=e, nf=r["nf"],
                                     accepted=r["n_accepted"], rejected=r["n_rejected"]))
            print(f"{p.name:13s} {bn:9s} готово", flush=True)
    return rows


def fixed_sweep(p, base, scheme):
    out = []
    for N in NS:
        r = integrate(p, base, scheme, N)
        out.append((r["nf"], rel_err(p, r["y"])))
    return np.array(out)


def gain_at(fixed, adapt, levels):
    """Во сколько раз меньше вычислений f нужно адаптивному решателю на уровне ошибки ε."""
    def nf_at(data, eps):
        nf, e = data[:, 0], data[:, 1]
        ok = np.isfinite(e) & (e > 3e-14) & (e < 1)
        nf, e = nf[ok], e[ok]
        if len(e) < 2 or not (e.min() <= eps <= e.max()):
            return np.nan
        o = np.argsort(e)
        return float(np.exp(np.interp(np.log(eps), np.log(e[o]), np.log(nf[o]))))
    return [nf_at(fixed, eps) / nf_at(adapt, eps) for eps in levels]


def slope(tols, errs):
    ok = np.isfinite(errs) & (errs > 1e-13)
    if ok.sum() < 3:
        return np.nan
    return np.polyfit(np.log(tols[ok]), np.log(errs[ok]), 1)[0]


def main():
    t0 = time.time()
    probs = [P() for P in problems.ALL]
    hard = hard_problems()
    for p in probs + hard:
        p.reference()

    # --- 1. допуск -> ошибка
    rows = tol_sweep(probs + hard[:2])
    with open(os.path.join(OUT, "adaptive.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    sel = lambda pn, bn, pa: [r for r in rows if r["problem"] == pn and r["base"] == bn and r["pair"] == pa]

    fig, axs = plt.subplots(3, 3, figsize=(12, 10.5), sharex=True, sharey=True)
    for i, bn in enumerate(BASE_METHODS):
        for j, pa in enumerate(PAIR_ORDER):
            ax = axs[i, j]
            for c, p in zip(PROB_COLOR, probs):
                rr = sel(p.name, bn, pa)
                ax.loglog([r["tol"] for r in rr], [max(r["error"], 1e-16) for r in rr], "-o",
                          color=c, ms=2.5, lw=1.2, label=p.name)
            ax.loglog(TOLS, TOLS, ":", color="#8a8a85", lw=1)
            ax.text(TOLS[2], TOLS[2] * 3, "ошибка = tol", color="#6b6b66", fontsize=7, rotation=33)
            ax.set_title(f"{bn}, пара {pa}")
            ax.set_ylim(1e-16, 1e-1)
            if i == 2: ax.set_xlabel("допуск tol (rtol = atol)")
            if j == 0: ax.set_ylabel("ошибка в t = T")
    axs[0, 0].legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    fig.suptitle("Адаптивный шаг: заданный допуск и достигнутая глобальная ошибка", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_adapt_tol.png"), dpi=130); plt.close(fig)

    # сводка: наклон log err / log tol и диапазон err/tol
    summ = ["| опора | пара | медиана наклона err(tol) | медиана err/tol | max err/tol | доля отброшенных шагов |",
            "|---|---|---|---|---|---|"]
    for bn in BASE_METHODS:
        for pa in PAIR_ORDER:
            sl, ratios, rej = [], [], []
            for p in probs:
                rr = sel(p.name, bn, pa)
                tt = np.array([r["tol"] for r in rr]); ee = np.array([r["error"] for r in rr])
                sl.append(slope(tt, ee))
                ok = np.isfinite(ee) & (ee > 1e-13)
                ratios += list(ee[ok] / tt[ok])
                rej += [r["rejected"] / (r["accepted"] + r["rejected"]) for r in rr if np.isfinite(r["error"])]
            summ.append(f"| {bn} | {pa} | {np.nanmedian(sl):.2f} | {np.median(ratios):.1e} | "
                        f"{np.max(ratios):.1e} | {100 * np.mean(rej):.1f}% |")

    # --- 2. стоимость: адаптивный против постоянного шага
    gains = ["| задача | опора | пара | выигрыш по f при ε = 1e-4 | ε = 1e-6 | ε = 1e-9 |", "|---|---|---|---|---|---|"]
    for p in hard:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        for ax, (bn, base) in zip(axs, BASE_METHODS.items()):
            for pa in PAIR_ORDER:
                c = PAIR_COLOR[pa]
                hi = PAIRS[pa].high
                fx = fixed_sweep(p, base, hi)
                ad = np.array([(r["nf"], r["error"]) for r in
                               (sel(p.name, bn, pa) if p in hard[:2] else
                                [dict(nf=x["nf"], error=rel_err(p, x["y"])) for x in
                                 [integrate_adaptive(p, base, pa, rtol=t) for t in TOLS[TOLS >= TOL_MIN.get(pa, 0)]]])])
                ok = fx[:, 1] > 1e-15
                ax.loglog(fx[ok, 0], fx[ok, 1], "--", color=c, marker="s", ms=3,
                          label=f"{pa.split('/')[0]}, пост. шаг")
                ok = np.isfinite(ad[:, 1]) & (ad[:, 1] > 1e-15)
                ax.loglog(ad[ok, 0], ad[ok, 1], "-", color=c, marker="o", ms=3, label=f"{pa}, адапт.")
                g = gain_at(fx, ad, [1e-4, 1e-6, 1e-9])
                gains.append(f"| {p.name} | {bn} | {pa} | " +
                             " | ".join("—" if not np.isfinite(x) else f"{x:.2f}×" for x in g) + " |")
            ax.set_title(bn); ax.set_xlabel("вычислений f")
            ax.set_ylim(1e-15, 1e1)
        axs[0].set_ylabel("ошибка в t = T"); axs[0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Адаптивный и постоянный шаг: {p.descr}", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig_adapt_cost_{p.name}.png"), dpi=130)
        plt.close(fig)
        print(f"стоимость {p.name} готово", flush=True)

    # --- 3. история шага на Кеплере e = 0.9
    p = hard[0]
    fig, axs = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    for pa in PAIR_ORDER:
        r = integrate_adaptive(p, BASE_METHODS["KD"], pa, rtol=1e-8)
        axs[0].semilogy(r["t"][1:], r["h"], "--" if pa == "P6C/P4C" else "-", color=PAIR_COLOR[pa],
                        lw=2.4 if pa == "P6C/P4C" else 1.2,
                        label=f"{pa}: {r['n_accepted']} шагов, {r['n_rejected']} отброшено")
    path = integrate(p, BASE_METHODS["KD"], SCHEMES["P6C"], 4000, return_path=True)["path"]
    tt = np.linspace(0, p.T, len(path))
    axs[1].plot(tt, np.hypot(path[:, 0], path[:, 1]), color="#2a78d6")
    axs[0].set_ylabel("шаг h"); axs[0].legend(fontsize=7, frameon=False)
    axs[1].set_ylabel("расстояние r(t)"); axs[1].set_xlabel("t")
    fig.suptitle("История шага, KD, tol = 1e-8: " + p.descr +
                 "\n(у P6C/P4C и P6SC/P4C оценщик один и тот же, P4C, поэтому шаги совпадают)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_adapt_steps_kepler.png"), dpi=130); plt.close(fig)

    with open(os.path.join(OUT, "adaptive_summary.md"), "w") as fh:
        fh.write("## Допуск → ошибка (8 базовых задач)\n\n" + "\n".join(summ) +
                 "\n\n## Выигрыш адаптивного шага по числу вычислений f\n\n"
                 "Отношение nf(постоянный шаг) / nf(адаптивный) при одинаковой ошибке в t = T; "
                 "схема высокого порядка та же, у адаптивного решателя в nf входят "
                 "оценщик и отброшенные шаги. «—»: уровень не достигнут одним из решателей.\n\n" +
                 "\n".join(gains) + "\n")
    print("\n".join(summ)); print("\n".join(gains))
    print(f"время: {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
