"""Эксперимент 3б: симметрично-сопряжённые схемы (γ_{m+1-i} = conj γ_i) для симметричной опоры.

Набор коэффициентов замкнут относительно сопряжения, поэтому при Re γ_i > 0 композиция
A-устойчивого симметричного метода остаётся A-устойчивой (см. RESEARCH.md, вопрос 3).
Ищем минимальное число стадий для порядков 4 и 6 (с проекцией Re).
"""
import sys, os, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from csolver.order_conditions import search, K_SYMMETRIC, achieved_order

out = []
for p in (4, 6):
    for m in range(2, 9):
        sols = search(K_SYMMETRIC, p, m, "conj", True, nstart=40)
        sols = [g for g in sols if np.min(g.real) > 0]
        if sols:
            sols.sort(key=lambda g: np.max(np.abs(g)))
            g = sols[0]
            print(p, m, len(sols), np.round(g, 6), achieved_order(g, K_SYMMETRIC), flush=True)
            out.append(dict(order=p, stages=m, best=[[z.real, z.imag] for z in g]))
            break
        print(p, m, "нет", flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "conj_search.json"), "w"), indent=1)
