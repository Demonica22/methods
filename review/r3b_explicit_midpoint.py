"""Ревью п.7 (дополнение): если под «средней точкой» в вопросе 4 понимать обычную явную среднюю точку
y1 = y0 + h f(y0 + h/2 f(y0)) (несимметричная, дефект симметрии h^4, см. r1), то работают только схемы,
построенные для общей несимметричной опоры (general_*), а схемы для Gragg-2 (midpoint_*) - нет.
Локальная ошибка в mpmath, как в r3. Запуск: python r3b_explicit_midpoint.py -> out/r3b_explicit_midpoint.txt
"""
import os, sys
from multiprocessing import Pool
import mpmath as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rv_common as R
import r3_local_order as L

L.bases = lambda: {"expl_midpoint": R.ExplicitMidpoint()}
SCH = ["P4C", "P6C", "P6SC*", "midpoint_p6_gen_Re*", "general_p4_gen_C*", "general_p6_pal_Re*", "general_p6_gen_Re*"]

if __name__ == "__main__":
    jobs = [(pn, "expl_midpoint", sn) for pn in L.probs() for sn in SCH]
    with Pool(4) as pool:
        res = pool.map(L.job, jobs, chunksize=1)
    pn_list = list(L.probs())
    keyed = {(a, c): s for a, b, c, e, s in res}
    lines = ["# Явная средняя точка как опора: порядок по локальной ошибке (наклон−1, две последние пары)\n",
             "| схема | " + " | ".join(pn_list) + " |", "|---|" + "---|" * len(pn_list)]
    for sn in SCH:
        lines.append(f"| {sn} | " + " | ".join(f"{keyed[(pn, sn)][-2]:.2f}/{keyed[(pn, sn)][-1]:.2f}" for pn in pn_list) + " |")
    txt = "\n".join(lines) + "\n"
    print(txt)
    open(os.path.join(L.HERE, "out", "r3b_explicit_midpoint.txt"), "w").write(txt)
