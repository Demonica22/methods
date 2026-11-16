"""Порядок по локальной ошибке в высокой точности (mpmath, 40 знаков): один шаг композиции
длины h против эталона mp.odefun (Тейлор). Наклон локальной ошибки = p + 1.
Глобальный тест (verify.py) на Кеплере быстро упирается в округление, поэтому нужен этот.
Результат: local_order.md.  Запуск: python local_order.py
"""
import os, json
from multiprocessing import Pool
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
DPS = 40


def vdp(t, y):
    return [y[1], (1 - y[0] ** 2) * y[1] - y[0]]


def kepler(t, y):
    r3 = (y[0] ** 2 + y[1] ** 2) ** mp.mpf(1.5)
    return [y[2], y[3], -y[0] / r3, -y[1] / r3]


def lotka(t, y):
    return [y[0] * (mp.mpf(2) / 3 - mp.mpf(4) / 3 * y[1]), y[1] * (y[0] - 1)]


def duffing(t, y):  # неавтономная
    return [y[1], -mp.mpf("0.2") * y[1] - y[0] - y[0] ** 3 + mp.mpf("0.3") * mp.cos(t)]


PROBS = {"vanderpol": (vdp, ["2", "0"]), "kepler": (kepler, ["0.5", "0", "0", "1.7320508075688772935"]),
         "lotka": (lotka, ["1.5", "0.7"]), "duffing": (duffing, ["1", "0"])}
HS = [mp.mpf(2) ** -k for k in range(4, 11)]


def em_step(f, t, y, h):
    k1 = f(t, y)
    yh = [a + h / 2 * b for a, b in zip(y, k1)]
    return [a + h * b for a, b in zip(y, f(t + h / 2, yh))]


def comp_step(f, y, h, gs, proj):
    t = mp.mpf(0)
    y = [mp.mpc(v) for v in y]
    for g in gs:
        y = em_step(f, t, y, g * h)
        t = t + g * h
    return [v.real for v in y] if proj else y


def job(a):
    pn, b = a
    mp.mp.dps = DPS
    f, y0 = PROBS[pn]
    y0 = [mp.mpf(v) for v in y0]
    sol = mp.odefun(lambda t, y: f(t, y), 0, y0)
    gs = [mp.mpc(mp.mpf(x), mp.mpf(z)) for x, z in b["gammas"]]
    es = []
    for h in HS:
        y1 = comp_step(f, y0, h, gs, b["project_real"])
        es.append(max(abs(u - v) for u, v in zip(y1, sol(h))))
    sl = [float(mp.log(es[i] / es[i + 1], 2)) - 1 for i in range(len(es) - 1)]
    return pn, b, sl


if __name__ == "__main__":
    best = json.load(open(os.path.join(HERE, "data", "best.json")))
    jobs = [(pn, b) for b in best for pn in PROBS]
    with Pool(4) as pool:
        res = pool.map(job, jobs)
    L = ["# Порядок по локальной ошибке (mpmath, 40 знаков)\n",
         "Наклон log2(e(h)/e(h/2)) − 1 для двух последних пар шагов, h = 1/16 … 1/1024.\n",
         "| схема | стадий | " + " | ".join(PROBS) + " |", "|---|---|" + "---|" * len(PROBS)]
    for b in best:
        row = {pn: sl for pn, bb, sl in res if bb is b or bb == b}
        L.append(f"| p{b['order']} {b['struct']} {'Re' if b['project_real'] else 'C'} | {b['stages']} | "
                 + " | ".join(f"{row[pn][-2]:.2f}/{row[pn][-1]:.2f}" for pn in PROBS) + " |")
    txt = "\n".join(L) + "\n"
    print(txt)
    open(os.path.join(HERE, "local_order.md"), "w").write(txt)
