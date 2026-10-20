"""Эксперимент 4 (вопрос 4): поиск комплексных коэффициентов композиции
для несимметричной опоры 2-го порядка.

Для каждой постановки перебирается число стадий m, пока многостартовый
Левенберг–Марквардт не найдёт решение системы условий порядка.
Запуск: python experiments/coeff_search.py   (результат: results/coeff_search.json, .md)
"""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from csolver.order_conditions import search, K_MIDPOINT, K_GENERAL

CASES = [  # (опора, K, порядок, палиндром, проекция Re)
    ("midpoint", "K_MIDPOINT", 6, True, True), ("midpoint", "K_MIDPOINT", 6, True, False),
    ("midpoint", "K_MIDPOINT", 6, False, True),
    ("general", "K_GENERAL", 4, True, True), ("general", "K_GENERAL", 4, False, True),
    ("general", "K_GENERAL", 4, True, False), ("general", "K_GENERAL", 4, False, False),
    ("general", "K_GENERAL", 6, True, True), ("general", "K_GENERAL", 6, False, True),
]
KS = {"K_MIDPOINT": K_MIDPOINT, "K_GENERAL": K_GENERAL}


def run(case):
    base, kname, p, pal, proj = case
    t = time.time()
    for m in range(2, 12 if pal else 9):
        sols = search(KS[kname], p, m, pal, proj, nstart=40 if m < 7 else 24)
        if sols:
            # предпочтение: все Re γ > 0, затем наименьший max|γ|
            sols.sort(key=lambda g: (np.min(g.real) <= 0, np.max(np.abs(g))))
            return dict(base=base, K=kname, order=p, palindromic=pal, project_real=proj, stages=m,
                        n_solutions=len(sols), time=time.time() - t,
                        best=[[g.real, g.imag] for g in sols[0]],
                        all=[[[g.real, g.imag] for g in s] for s in sols[:10]])
        print(f"  {base} p={p} pal={pal} proj={proj}: m={m} нет решений", flush=True)
    return dict(base=base, K=kname, order=p, palindromic=pal, project_real=proj, stages=None)


if __name__ == "__main__":
    with Pool(4) as pool:
        res = pool.map(run, CASES, chunksize=1)
    json.dump(res, open(os.path.join(ROOT, "results", "coeff_search.json"), "w"), indent=1)
    for r in res:
        print(r["base"], r["order"], r["palindromic"], r["project_real"], "->", r["stages"])
