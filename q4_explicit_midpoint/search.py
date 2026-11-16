"""Многостартовый поиск комплексных коэффициентов композиции для явной средней точки
по точным условиям на деревьях (em_trees.py). Для каждой постановки m перебирается
от малого к большему; найденные решения сохраняются в data/search.json.
Запуск: python search.py [NSTART]
"""
import os, sys, json, time
from multiprocessing import Pool
import numpy as np
from scipy.optimize import least_squares
import em_trees as E

HERE = os.path.dirname(os.path.abspath(__file__))
NSTART = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def expand(x, m, struct):
    """struct: gen — общий вид; pal — γ_{m+1-i} = γ_i; conj — γ_{m+1-i} = conj γ_i."""
    q = len(x) // 2
    z = x[:q] + 1j * x[q:]
    if struct == "gen":
        return z
    if struct == "pal":
        return np.concatenate([z, z[:m // 2][::-1]])
    z = z.copy()
    if m % 2:
        z[-1] = z[-1].real
    return np.concatenate([z, np.conj(z[:m // 2][::-1])])


def nparams(m, struct):
    return 2 * (m if struct == "gen" else (m + 1) // 2)


def run(case):
    p, m, struct, proj = case
    q = nparams(m, struct) // 2
    rng = np.random.default_rng(1000 * p + 10 * m + len(struct) + proj)
    f = lambda x: E.residual(expand(x, m, struct), p, proj)
    sols, t0 = [], time.time()
    for k in range(NSTART):
        if k % 2 == 0:
            x0 = np.concatenate([rng.uniform(-0.2, 0.6, q), rng.uniform(-0.6, 0.6, q)]) / max(1, m / 3)
        else:
            x0 = np.concatenate([rng.normal(1.0 / m, 0.4, q), rng.normal(0, 0.4, q)])
        try:
            meth = "lm" if len(f(x0)) >= len(x0) else "trf"   # LM не берёт недоопределённые системы
            r = least_squares(f, x0, method=meth, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=200 * len(x0))
        except Exception:
            continue
        if np.max(np.abs(r.fun)) < 1e-12:
            g = expand(r.x, m, struct)
            if not any(np.allclose(np.sort_complex(g), np.sort_complex(s), atol=1e-7) for s in sols):
                sols.append(g)
    sols.sort(key=lambda g: (np.min(g.real) <= 0, np.max(np.abs(g))))
    return dict(order=p, stages=m, struct=struct, project_real=bool(proj), n_solutions=len(sols),
                time=round(time.time() - t0, 1), sols=[[[z.real, z.imag] for z in g] for g in sols[:30]])


CASES = [(4, m, s, pr) for s in ("gen", "pal", "conj") for pr in (1, 0) for m in range(2, 7)] + \
        [(6, m, s, pr) for s in ("gen", "pal", "conj") for pr in (1, 0) for m in range(4, 10)]
# второй проход: без проекции Re для p = 6 нужно больше стадий
CASES2 = [(6, m, s, 0) for s in ("gen", "pal", "conj") for m in range(10, 14)]

if __name__ == "__main__":
    if len(sys.argv) > 2:
        CASES = CASES2
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with Pool(4) as pool:
        res = []
        for r in pool.imap_unordered(run, CASES):
            print(r["order"], r["stages"], r["struct"], "Re" if r["project_real"] else "C",
                  "->", r["n_solutions"], f"({r['time']} c)", flush=True)
            res.append(r)
    res.sort(key=lambda r: (r["order"], r["struct"], -r["project_real"], r["stages"]))
    json.dump(res, open(os.path.join(HERE, "data", "search.json" if len(sys.argv) <= 2 else "search2.json"), "w"), indent=1)
