"""Ревью п.8: классы задач, на которых композиции с комплексными коэффициентами не работают.

A. y' = -|y| (неаналитическая f) - воспроизведение.
B. np.interp с комплексным аргументом - проверка утверждения.
C. Слабо затухающий осциллятор ω=40, трапеции - воспроизведение + сверка с предсказанием по μ(z) (r5).
D. KD: нормальная форма vs форма Фробениуса - воспроизведение.
E. (новое) Комплекснозначная голоморфная ОДУ y' = i y^2: проекцию Re применить нельзя,
   поэтому схемы, чей порядок держится только на проекции (P3C, P6SC, найденные *_Re), теряют порядок.
Запуск: python r6_failure.py -> out/r6_failure.txt
"""
import os, sys, json
import numpy as np
import mpmath as mp
from scipy.linalg import expm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rv_common as R

np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
lines = []
pr = lambda s="": (print(s, flush=True), lines.append(s))
mp.mp.dps = 30
S = {k: [complex(x) for x in v] for k, v in R.schemes().items()}
pol = json.load(open(os.path.join(HERE, "data", "polished.json")))
S["P6SC"] = [complex(float(a), float(b)) for a, b in pol["P6SC"]["g"]]
for k in ["midpoint_p6_gen_Re", "general_p6_pal_Re"]:
    S[k] = [complex(float(a), float(b)) for a, b in pol[k]["g"]]
BASES = {"trapezoid": R.Trapezoid(), "KD": R.KD(0.5), "gragg2": R.Gragg(2)}
NS = [10, 20, 40, 80, 160, 320, 640]


def slopes(errs):
    e = np.asarray(errs)
    return np.log2(e[:-1] / e[1:])


def last_clean(errs, lo=1e-12):
    s = slopes(errs)
    e = np.asarray(errs)
    s = s[(e[1:] > lo)]
    return f"{s[-1]:.2f}" if len(s) else "—"


# ---------------------------------------------------------------- A
pr("## A. y' = −|y| против y' = −y, y(0)=1, T=2 (порядок по последней паре N с ошибкой > 1e-12)\n")
pa = R.Prob("abs", lambda B, t, y: B.vec([-abs(y[0])]), lambda B, t, y: B.mat([[-np.sign(y[0].real)]]), [1.0], 2.0)
pl = R.Prob("lin", lambda B, t, y: B.vec([-y[0]]), lambda B, t, y: B.mat([[-1.0]]), [1.0], 2.0)
ex = np.exp(-2.0)
pr("| опора | схема | −y | −\\|y\\| |")
pr("|---|---|---|---|")
for bn, b in BASES.items():
    for sn in ["P2", "P4R", "P4C", "P6C", "P6SC"]:
        el = [abs(R.integrate(R.NP, b, pl, S[sn], N)[0].real - ex) for N in NS]
        ea = [abs(R.integrate(R.NP, b, pa, S[sn], N)[0].real - ex) for N in NS]
        pr(f"| {bn} | {sn} | {last_clean(el)} | {last_clean(ea)} |")

# ---------------------------------------------------------------- B
pr("\n## B. np.interp с комплексным аргументом\n")
for arg in [1 + 1j, np.complex128(1 + 1j)]:
    try:
        v = np.interp(arg, [0.0, 2.0], [0.0, 2.0])
        pr(f"- np.interp({type(arg).__name__}(1+1j), [0,2], [0,2]) = {v!r}")
    except Exception as ex_:
        pr(f"- np.interp({type(arg).__name__}(1+1j), ...) -> {type(ex_).__name__}: {ex_}")
pr(f"- numpy {np.__version__}")

# ---------------------------------------------------------------- C
pr("\n## C. x'' + 2ζωx' + ω²x = 0, ω = 40, ζ = 0.001, T = 20, трапеции: амплитуда в t=T (точно 0.449)\n")
om, ze = 40.0, 0.001
A = np.array([[0, 1], [-om ** 2, -2 * ze * om]])
pc = R.Prob("osc40", lambda B, t, y: B.vec([y[1], A[1, 0] * y[0] + A[1, 1] * y[1]]),
            lambda B, t, y: B.mat(A.tolist()), [1.0, 0.0], 20.0)
lam = -ze * om + 1j * om * np.sqrt(1 - ze ** 2)


def mu(z, g):
    Rz = np.prod([(1 + gi * z / 2) / (1 - gi * z / 2) for gi in g])
    Rc = np.prod([(1 + gi * np.conj(z) / 2) / (1 - gi * np.conj(z) / 2) for gi in g])
    return (Rz + np.conj(Rc)) / 2


pr("| hω | схема | амплитуда (численно) | предсказание |μ(hλ)|^N |")
pr("|---|---|---|---|")
for x in [5, 10, 15, 30]:
    N = int(round(20 * om / x))
    h = 20 / N
    for sn in ["P2", "P4C", "P6C", "P6SC"]:
        y = R.integrate(R.NP, R.Trapezoid(), pc, S[sn], N)
        amp = np.hypot(y[0].real, y[1].real / om)
        pred = abs(mu(h * lam, S[sn])) ** N
        pr(f"| {x} | {sn} | {amp:.3g} | {pred:.3g} |")

# ---------------------------------------------------------------- D
pr("\n## D. KD: λ = −5 ± 1.5i, нормальная форма (k=1) и форма Фробениуса (k=0), T = 20, max|z(T)| (точно ≈ 4e-44)\n")
sig, w = -5.0, 1.5
A1 = np.array([[sig, w], [-w, sig]])
Sm = np.array([[1.0, 0.0], [sig, w]])
A0 = Sm @ A1 @ np.linalg.inv(Sm)
mk = lambda M, y0: R.Prob("lin2", lambda B, t, y, M=M: B.vec([M[0, 0] * y[0] + M[0, 1] * y[1], M[1, 0] * y[0] + M[1, 1] * y[1]]),
                          lambda B, t, y, M=M: B.mat(M.tolist()), y0, 20.0)
P1, P0 = mk(A1, [1.0, 0.0]), mk(A0, list(Sm @ np.array([1.0, 0.0])))
Si = np.linalg.inv(Sm)
pr("| h | KD k=1 | KD k=0 | KD+P4C k=1 | KD+P4C k=0 | KD+P6C k=0 | трапеции k=0 |")
pr("|---|---|---|---|---|---|---|")
for h in [0.1, 0.25, 0.4, 0.6, 1.0]:
    N = int(round(20 / h))
    c = []
    for sn, P, back in [("P2", P1, np.eye(2)), ("P2", P0, Si), ("P4C", P1, np.eye(2)), ("P4C", P0, Si), ("P6C", P0, Si)]:
        y = R.integrate(R.NP, R.KD(), P, S[sn], N).real
        c.append(np.max(np.abs(back @ y)))
    y = R.integrate(R.NP, R.Trapezoid(), P0, S["P2"], N).real
    c.append(np.max(np.abs(Si @ y)))
    pr(f"| {h} | " + " | ".join("разнос" if not np.isfinite(v) or v > 1 else f"{v:.1e}" for v in c) + " |")

# ---------------------------------------------------------------- E
pr("\n## E. (новое) Комплекснозначная голоморфная ОДУ y' = i·y², y(0)=1, T=1 (точно y = 1/(1 − i t)); проекция Re невозможна\n")
pe = R.Prob("cplx", lambda B, t, y: B.vec([1j * y[0] ** 2]), lambda B, t, y: B.mat([[2j * y[0]]]), [1.0], 1.0)
exact = 1 / (1 - 1j)
pr("| опора | схема | наблюдаемый порядок (без проекции) | ожидание с Re-проекцией |")
pr("|---|---|---|---|")
for bn, b, schs in [("trapezoid", R.Trapezoid(), ["P4C", "P3C", "P6C", "P6SC", "general_p6_pal_Re"]),
                    ("gragg2", R.Gragg(2), ["P4C", "midpoint_p6_gen_Re"])]:
    for sn in schs:
        e = [abs(R.integrate(R.NP, b, pe, S[sn], N, project_real=False)[0] - exact) for N in NS]
        exp_ = {"P4C": 4, "P3C": 4, "P6C": 6 if bn == "trapezoid" else 5, "P6SC": 6, "general_p6_pal_Re": 6,
                "midpoint_p6_gen_Re": 6}[sn]
        pr(f"| {bn} | {sn} | {last_clean(e)} (наклоны {' '.join(f'{v:.2f}' for v in slopes(e))}) | {exp_} |")
open(os.path.join(HERE, "out", "r6_failure.txt"), "w").write("\n".join(lines) + "\n")
