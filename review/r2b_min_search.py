"""Ревью п.6-7: проверка утверждений о МИНИМАЛЬНОМ числе стадий (coeff_search_log.txt, conj_search.py).

Автор искал 40 (24 при m≥7) стартами Левенберга-Марквардта и при неудаче писал «нет решений».
Здесь - независимый многостартовый поиск по точным условиям (слова, rv_words), NSTART стартов,
двумя распределениями начальных точек, для числа стадий на 1-2 меньше заявленного минимума.
Найденные решения уточняются в mpmath и перепроверяются.
Запуск: python r2b_min_search.py [NSTART] -> out/r2b_min_search.txt, data/r2b_found.json
"""
import os, sys, json, time
from multiprocessing import Pool
import numpy as np
import mpmath as mp
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rv_words as W

HERE = os.path.dirname(os.path.abspath(__file__))
NSTART = int(sys.argv[1]) if len(sys.argv) > 1 else 1500

# (модель, порядок, m, структура, проекция Re, заявленный минимум автора)
CASES = [
    ("sym", 6, 4, "conj", True, 5),
    ("gragg2", 6, 4, "gen", True, 5),
    ("gragg2", 6, 5, "pal", True, 7),
    ("gragg2", 6, 6, "pal", True, 7),
    ("gragg2", 6, 7, "pal", False, 9),
    ("gragg2", 6, 8, "pal", False, 9),
    ("general", 4, 3, "gen", False, 4),
    ("general", 4, 4, "pal", False, 5),
    ("general", 6, 6, "gen", True, 7),
    ("general", 6, 7, "pal", True, 9),
    ("general", 6, 8, "pal", True, 9),
    ("gragg2", 6, 5, "conj", True, None),   # симметрично-сопряжённые для Грэгга (автор не искал)
    ("general", 6, 5, "conj", True, None),
    ("general", 6, 6, "conj", True, None),
]


def expand(x, m, struct):
    q = len(x) // 2
    z = x[:q] + 1j * x[q:]
    if struct == "gen":
        return z
    if struct == "pal":
        return np.concatenate([z, z[:m // 2][::-1]])
    if struct == "conj":
        z = z.copy()
        if m % 2:
            z[-1] = z[-1].real
        return np.concatenate([z, np.conj(z[:m // 2][::-1])])


def run(case):
    model, p, m, struct, proj, claimed = case
    ws = W.WordSet(W.MODELS[model], p)
    q = m if struct == "gen" else (m + 1) // 2
    rng = np.random.default_rng(12345 + m * 7 + p)
    sols = []
    t0 = time.time()
    for k in range(NSTART):
        if k % 2 == 0:
            x0 = np.concatenate([rng.uniform(-0.2, 0.6, q), rng.uniform(-0.6, 0.6, q)]) / max(1, m / 3)
        else:
            x0 = np.concatenate([rng.normal(1.0 / m, 0.5, q), rng.normal(0, 0.5, q)])
        f = lambda x: ws.residual(expand(x, m, struct), proj)
        try:
            r = least_squares(f, x0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
        except Exception:
            continue
        if np.max(np.abs(r.fun)) < 1e-11:
            g = expand(r.x, m, struct)
            if not any(np.allclose(np.sort_complex(g), np.sort_complex(s), atol=1e-7) for s in sols):
                sols.append(g)
    out = dict(case=case, n=len(sols), time=time.time() - t0)
    if sols:
        sols.sort(key=lambda g: (np.min(g.real) <= 0, np.max(np.abs(g))))
        g = sols[0]
        mp.mp.dps = 40
        # контроль точными условиями
        ws7 = W.WordSet(W.MODELS[model], p)
        d = ws7.defects([mp.mpc(z) for z in g], proj, use_mp=True)
        out.update(best=[[float(z.real), float(z.imag)] for z in g], defect=float(max(d)),
                   min_re=float(np.min(g.real)), any_all_pos=bool(any(np.min(s.real) > 0 for s in sols)))
    return out


if __name__ == "__main__":
    with Pool(4) as pool:
        res = pool.map(run, CASES, chunksize=1)
    json.dump(res, open(os.path.join(HERE, "data", "r2b_found.json"), "w"), indent=1)
    lines = [f"# Поиск схем с меньшим числом стадий ({NSTART} стартов LM на случай, точные условия)\n",
             "| модель опоры | p | m | структура | Re | мин. у автора | решений | все Re γ>0 есть? | лучшее решение γ (max|γ| min) | дефект (mp) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        model, p, m, struct, proj, claimed = r["case"]
        if r["n"]:
            gs = ", ".join(f"{a:+.6f}{b:+.6f}i" for a, b in r["best"])
            lines.append(f"| {model} | {p} | {m} | {struct} | {'да' if proj else 'нет'} | {claimed} | {r['n']} | "
                         f"{'да' if r['any_all_pos'] else 'нет'} | {gs} | {r['defect']:.1e} |")
        else:
            lines.append(f"| {model} | {p} | {m} | {struct} | {'да' if proj else 'нет'} | {claimed} | 0 | — | — | — |")
    txt = "\n".join(lines) + "\n"
    print(txt)
    open(os.path.join(HERE, "out", "r2b_min_search.txt"), "w").write(txt)
