"""Ревью п.3 и п.7: порядок по ЛОКАЛЬНОЙ ошибке в высокой точности (mpmath, dps=50).

err(h) = || P(Ψ_h(y0)) − φ_h(y0) ||, P = Re (схемы «с проекцией») или тождество (схемы «без проекции»),
φ_h - точный поток (ряд Тейлора mpmath.odefun). Наклон log2 err(h)/err(h/2) -> p+1.
Округление не влияет (ошибки до 1e-30 видны), эталон точный, задачи нелинейные и неавтономные
(в т.ч. generic3, которой не было у автора).
Запуск: python r3_local_order.py        -> out/r3_local_order.txt (опубликованные цифры)
        python r3_local_order.py found  -> out/r3_local_order_found.txt (уточнённые найденные схемы, нужен r2)
"""
import os, sys, json
from multiprocessing import Pool
import mpmath as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rv_common as R

HERE = os.path.dirname(os.path.abspath(__file__))
DPS = 50
KS = range(3, 9)  # h = 0.4 * 2^-k


def hs_for(P):
    sc = mp.mpf(1) / 8 if P.name == "lorenz" else mp.mpf(1)
    return [sc * mp.mpf("0.4") * mp.mpf(2) ** (-k) for k in KS]


def probs():
    return {p.name: p for p in [R.vdp(), R.lotka(), R.duffing(), R.kepler(), R.lorenz(), R.generic3()]}


def bases():
    return {"trapezoid": R.Trapezoid(), "KD": R.KD(0.5), "gragg2": R.Gragg(2), "heun": R.Gragg(1)}


def schemes_all():
    mp.mp.dps = DPS
    S = {k: (v, True) for k, v in R.schemes().items()}
    pol = json.load(open(os.path.join(HERE, "data", "polished.json")))
    S["P6SC*"] = ([mp.mpc(a, b) for a, b in pol["P6SC"]["g"]], True)  # уточнённая
    for k, v in R.found_schemes().items():
        S[k] = (v["g"], v["proj"])
        if k in pol and k != "general_p4_pal_Re":
            S[k + "*"] = ([mp.mpc(a, b) for a, b in pol[k]["g"]], v["proj"])  # уточнённые (точное решение рядом)
    # точное решение для general_p4_pal_Re существует только с Re γ2 = 0 (см. r2): γ = (1/2 - i/(2√3), i/√3, 1/2 - i/(2√3))
    g1 = mp.mpf(1) / 2 - 1j / (2 * mp.sqrt(3))
    S["general_p4_pal_Re_exact"] = ([g1, 1j / mp.sqrt(3), g1], True)
    return S


def job(args):
    pname, bname, sname = args
    mp.mp.dps = DPS
    P, b = probs()[pname], bases()[bname]
    g, proj = schemes_all()[sname]
    hs = hs_for(P)
    F = lambda t, y: [mp.re(v) for v in P.f(R.MP, t, R.MP.vec(y))]
    sol = mp.odefun(F, mp.mpf(P.t0), [mp.mpf(v) for v in P.y0])
    y0 = R.MP.vec(P.y0)
    errs = []
    for h in hs:
        y = R.compose_step(R.MP, b, P, mp.mpf(P.t0), y0, h, g)
        if proj:
            y = R.MP.re(y)
        ex = R.MP.vec(sol(mp.mpf(P.t0) + h))
        errs.append(R.MP.norm(y - ex))
    sl = [float(mp.log(errs[i] / errs[i + 1], 2)) - 1 for i in range(len(errs) - 1)]
    return (pname, bname, sname, [float(e) for e in errs], sl)


if __name__ == "__main__":
    mp.mp.dps = DPS
    S = schemes_all()
    ONLY_FOUND = len(sys.argv) > 1 and sys.argv[1] == "found"
    std = [] if ONLY_FOUND else ["P2", "P3C", "P4C", "P4R", "P6C", "P6R", "P6SC", "P6SC*"]
    if ONLY_FOUND:
        S = {k: v for k, v in S.items() if k.endswith("*") or k.endswith("_exact")}
    jobs = []
    for pn in probs():
        for bn in bases():
            for sn in std:
                jobs.append((pn, bn, sn))
        for sn in S:
            if sn.startswith("midpoint"):
                for bn in ("gragg2", "heun", "KD"):
                    jobs.append((pn, bn, sn))
            elif sn.startswith("general"):
                for bn in ("heun", "gragg2", "trapezoid"):
                    jobs.append((pn, bn, sn))
    with Pool(4) as pool:
        res = pool.map(job, jobs, chunksize=1)
    tag = "_found" if ONLY_FOUND else ""
    json.dump(res, open(os.path.join(HERE, "data", f"r3_local{tag}.json"), "w"))
    lines = ["# Порядок по локальной ошибке (mpmath dps=50). В ячейке: наклон−1 по двум последним парам шагов",
             f"h = 0.4·2^-k, k = {KS.start}..{KS.stop - 1} (Лоренц: h/8). Ожидание = номинальный порядок.\n"]
    pn_list = list(probs())
    keyed = {(a, b, c): (e, s) for a, b, c, e, s in res}
    snames = std + [s for s in S if s not in std]
    for bn in bases():
        rows = [s for s in snames if (pn_list[0], bn, s) in keyed]
        lines.append(f"\n## опора {bn}\n")
        lines.append("| схема | " + " | ".join(pn_list) + " | min ошибка |")
        lines.append("|---|" + "---|" * (len(pn_list) + 1))
        for s in rows:
            cells = []
            mins = []
            for pn in pn_list:
                e, sl = keyed[(pn, bn, s)]
                cells.append(f"{sl[-2]:.2f}/{sl[-1]:.2f}")
                mins.append(e[-1])
            lines.append(f"| {s} | " + " | ".join(cells) + f" | {min(mins):.1e} |")
    txt = "\n".join(lines) + "\n"
    print(txt)
    open(os.path.join(HERE, "out", f"r3_local_order{tag}.txt"), "w").write(txt)
