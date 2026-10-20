"""Ревью п.4-5: устойчивость композиций.

1) Трапеции: функция устойчивости R(z) = Π r(γ_i z), r(w) = (1+w/2)/(1-w/2), вычисляется аналитически.
   Для вещественной системы 2x2 с собственными числами z, z̄ проекция Re даёт множитель
   μ(z) = (R(z) + conj R(conj z))/2 (собственное число Re R(hA)). Без проекции — max(|R(z)|, |R(z̄)|).
   μ рациональна, полюса 2/γ_i и 2/conj γ_i; если все Re γ_i > 0, полюсов в левой полуплоскости нет и
   по принципу максимума A-устойчивость <=> sup_y |μ(iy)| ≤ 1, а A(α) <=> |μ| ≤ 1 на луче arg(−z) = α.
2) KD: матрица перехода 2x2 (тестовая матрица из статьи, асимметрия k), спектральный радиус Re-проекции,
   габариты связной области устойчивости, содержащей окрестность нуля.
3) Контрпример к «достаточному условию» RESEARCH.md: Гаусс-Лежандр (2 и 3 стадии) - симметричные,
   A-устойчивые, с вещественными коэффициентами - в композиции с замкнутым относительно сопряжения
   набором γ с Re γ > 0 (в т.ч. P6SC) дают полюс в левой полуплоскости.
Запуск: python r5_stability.py -> out/r5_stability.txt, fig_r5_*.png
"""
import os, sys, json
import numpy as np
import mpmath as mp
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rv_common as R

np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
lines = []
pr = lambda s="": (print(s, flush=True), lines.append(s))
mp.mp.dps = 30
S = {k: np.array([complex(x) for x in v]) for k, v in R.schemes().items()}
NAMES = ["P2", "P3C", "P4C", "P4R", "P6C", "P6R", "P6SC"]


def r_trap(w):
    return (1 + w / 2) / (1 - w / 2)


def Rcomp(z, g, r=r_trap):
    out = np.ones_like(z, dtype=complex)
    for gi in g:
        out = out * r(gi * z)
    return out


def mu(z, g, proj=True, r=r_trap):
    if proj:
        return np.abs((Rcomp(z, g, r) + np.conj(Rcomp(np.conj(z), g, r))) / 2)
    return np.maximum(np.abs(Rcomp(z, g, r)), np.abs(Rcomp(np.conj(z), g, r)))


ys = np.concatenate([np.linspace(0, 100, 400001), np.logspace(2, 7, 200001)])
rr = np.concatenate([np.linspace(1e-4, 100, 100001), np.logspace(2, 7, 50001)])


def alpha_of(g, proj, r=r_trap):
    """Наибольший α (град), при котором |μ| ≤ 1 на луче arg(−z) = α (и полюсов в секторе нет)."""
    poles_ok = np.all(np.real(g) > 0)
    if not poles_ok:
        return 0.0
    lo, hi = 0.0, 90.0
    ok = lambda a: np.max(mu(-rr * np.exp(-1j * np.radians(a)), g, proj, r)) <= 1 + 1e-12
    if ok(90.0):
        return 90.0
    for _ in range(40):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if ok(m) else (lo, m)
    return lo


pr("## 1. Трапеции: A-устойчивость композиций (аналитическая функция устойчивости)\n")
pr("| схема | проекция | min Re γ | полюса в Re z<0 | sup|μ(iy)| | где y | A(α), град |")
pr("|---|---|---|---|---|---|---|")
res_tr = {}
for n in NAMES:
    g = S[n]
    for proj in (True, False):
        m = mu(1j * ys, g, proj)
        i = int(np.nanargmax(m))
        a = alpha_of(g, proj)
        poles = [complex(2 / x) for x in g if (2 / x).real < 0]
        res_tr[(n, proj)] = (m[i], ys[i], a)
        pr(f"| {n} | {'Re' if proj else 'нет'} | {g.real.min():.4f} | {'да: ' + ', '.join(f'{p.real:.3f}' for p in poles) if poles else 'нет'} | "
           f"{m[i]:.6f} | {ys[i]:.3g} | {a:.2f} |")

# ---------------------------------------------------------------- KD
def kd_matrix(g, sig, om, k):
    a = 2 * sig * k / (1 + k)
    d = 2 * sig / (1 + k)
    b = np.ones_like(sig)
    c = a * d - sig ** 2 - om ** 2
    h1, h2 = g / 2, g / 2
    # Эйлер-Кромер: x1 <- x1 + h1(a x1 + b x2); x2 <- x2 + h1(c x1 + d x2)
    # D-метод назад: x2 <- (x2 + h2 c x1)/(1 - h2 d); x1 <- (x1 + h2 b x2)/(1 - h2 a)
    # строим матрицу, прогоняя базисные векторы (независимо от формул автора)
    cols = []
    for e in ([1.0, 0.0], [0.0, 1.0]):
        x1 = np.full(sig.shape, e[0], dtype=complex)
        x2 = np.full(sig.shape, e[1], dtype=complex)
        x1 = x1 + h1 * (a * x1 + b * x2)
        x2 = x2 + h1 * (c * x1 + d * x2)
        x2 = (x2 + h2 * c * x1) / (1 - h2 * d)
        x1 = (x1 + h2 * b * x2) / (1 - h2 * a)
        cols.append((x1, x2))
    return [[cols[0][0], cols[1][0]], [cols[0][1], cols[1][1]]]


def mm(X, Y):
    return [[X[i][0] * Y[0][j] + X[i][1] * Y[1][j] for j in range(2)] for i in range(2)]


def rho_kd(g, sig, om, k, proj=True):
    M = [[np.ones_like(sig, dtype=complex), np.zeros_like(sig, dtype=complex)],
         [np.zeros_like(sig, dtype=complex), np.ones_like(sig, dtype=complex)]]
    for gi in g:
        M = mm(kd_matrix(gi, sig, om, k), M)
    if proj:
        M = [[M[i][j].real for j in range(2)] for i in range(2)]
    tr = M[0][0] + M[1][1]
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    disc = np.sqrt(np.asarray(tr, complex) ** 2 - 4 * det)
    return np.maximum(np.abs((tr + disc) / 2), np.abs((tr - disc) / 2))


# контроль: матрица KD совпадает с шагом метода R.KD на линейной задаче
A = np.array([[-0.7, 1.0], [-3.2, -1.9]])
Pl = R.Prob("lin", lambda B, t, y: B.vec([A[0, 0] * y[0] + A[0, 1] * y[1], A[1, 0] * y[0] + A[1, 1] * y[1]]),
            lambda B, t, y: B.mat(A.tolist()), [1.0, 0.0], 1.0)
gtest = 0.4 + 0.3j
y1 = R.KD().step(R.NP, Pl, 0.0, R.NP.vec([1.0, 0.0]), gtest)
# в терминах (σ, ω, k): построим ту же матрицу напрямую
pr("\n## 2. KD: области устойчивости (тестовая матрица статьи, проекция Re)\n")
pr(f"(контроль реализации шага KD на линейной задаче: шаг = {y1})")

sig, om = np.meshgrid(np.linspace(-40, 2, 1681), np.linspace(0, 40, 801))
pr("\nГабариты связной компоненты {ρ ≤ 1}, содержащей z = −0.05 (сетка шаг 0.025/0.05; ω ≥ 0, симметрично):\n")
pr("| схема | k | Re z от | Im z до | площадь (ω≥0) | площадь/стадий |")
pr("|---|---|---|---|---|---|")
kd_regions = {}
for k in (1.0, 0.0):
    for n in ["P2", "P3C", "P4C", "P4R", "P6C", "P6R", "P6SC"]:
        rho = rho_kd(S[n], sig, om, k)
        st = np.isfinite(rho) & (rho <= 1 + 1e-12)
        lab, _ = ndimage.label(st)
        i0 = np.argmin(np.abs(om[:, 0] - 0.0)); j0 = np.argmin(np.abs(sig[0] + 0.05))
        L = lab[i0, j0]
        comp = lab == L if L > 0 else np.zeros_like(st)
        if comp.any():
            area = comp.sum() * 0.025 * 0.05
            pr(f"| {n} | {k:g} | {sig[comp].min():.2f} | {om[comp].max():.2f} | {area:.1f} | {area / len(S[n]):.2f} |")
        else:
            pr(f"| {n} | {k:g} | — | — | 0 | 0 |")
        kd_regions[(n, k)] = st

# ---------------------------------------------------------------- контрпример: Гаусс
pr("\n## 3. Контрпример к «достаточному условию A-устойчивости» (RESEARCH.md, вопрос 3)\n")


def r_gauss2(w):
    return (1 + w / 2 + w * w / 12) / (1 - w / 2 + w * w / 12)


def r_gauss3(w):
    return (1 + w / 2 + w ** 2 / 10 + w ** 3 / 120) / (1 - w / 2 + w ** 2 / 10 - w ** 3 / 120)


poles3 = np.roots([-1 / 120, 1 / 10, -1 / 2, 1])
poles2 = np.roots([1 / 12, -1 / 2, 1])
pr(f"- Полюса r_Gauss2: {np.round(poles2, 4)}, r_Gauss3: {np.round(poles3, 4)} (все в Re w > 0: опоры A-устойчивы)")
for nm, rf, P, g in [("Гаусс-3 ∘ P6SC", r_gauss3, poles3, S["P6SC"]),
                     ("Гаусс-2 ∘ {1/2 ± 1.5i}", r_gauss2, poles2, np.array([0.5 + 1.5j, 0.5 - 1.5j]))]:
    zp = np.array([p / gi for gi in g for p in P])
    bad = zp[zp.real < 0]
    pr(f"- **{nm}**: набор γ замкнут относительно сопряжения, min Re γ = {g.real.min():.4f} > 0; "
       f"полюса R в левой полуплоскости: {', '.join(f'{z.real:.4f}{z.imag:+.4f}i' for z in bad)}")
    z = bad[0] * (1 + 1e-3)  # рядом с полюсом
    m_im = np.max(np.abs(Rcomp(1j * ys[::50], g, rf)))
    pr(f"  sup|R(iy)| = {m_im:.12f} (на мнимой оси ровно 1, как и в «доказательстве»), но |R({z.real:.4f}{z.imag:+.4f}i)| = "
       f"{np.abs(Rcomp(np.array([z]), g, rf))[0]:.3g} > 1 ⇒ композиция НЕ A-устойчива.")
    # точка в левой полуплоскости с |R|>1 подальше от полюса
    X, Y = np.meshgrid(np.linspace(-30, 0, 601), np.linspace(-60, 60, 1201))
    Z = X + 1j * Y
    M = np.abs(Rcomp(Z, g, rf))
    pr(f"  доля точек сетки [−30,0]×[−60,60] с |R|>1: {(M > 1 + 1e-12).mean():.3%}; max|R| на сетке = {np.nanmax(M):.3g}")

# P6SC с трапециями
pr("\n- Для трапеций (единственный полюс w = 2 вещественный) условие верно: полюса 2/γ_i в Re z > 0 при Re γ_i > 0. "
   "Поэтому вывод «P6SC + трапеции A-устойчива» верен, а общее утверждение — нет.")

# ---------------------------------------------------------------- графики
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
yy = np.logspace(-1, 3, 4000)
for n, c in [("P3C", "#1baf7a"), ("P4C", "#eb6834"), ("P6C", "#e87ba4"), ("P6SC", "#2a78d6")]:
    axs[0].semilogx(yy, mu(1j * yy, S[n], True), color=c, label=n)
axs[0].axhline(1, color="#555", lw=0.8, ls=":")
axs[0].set_xlabel("y  (z = iy)"); axs[0].set_ylabel("|μ(iy)|, проекция Re")
axs[0].set_title("Трапеции: множитель на мнимой оси"); axs[0].legend(frameon=False, fontsize=8)
X, Y = np.meshgrid(np.linspace(-2, 1, 601), np.linspace(16, 24, 801))
Z = X + 1j * Y
M = np.abs(Rcomp(Z, S["P6SC"], r_gauss3))
cs = axs[1].contourf(X, Y, np.log10(M), levels=np.linspace(-1, 1, 21), cmap="RdBu_r", extend="both")
axs[1].contour(X, Y, M, levels=[1.0], colors="k", linewidths=0.7)
axs[1].axvline(0, color="#555", lw=0.8)
axs[1].set_title("Гаусс-3 ∘ P6SC: log10|R(z)|, полюс при z ≈ −0.41+20.04i")
axs[1].set_xlabel("Re z"); axs[1].set_ylabel("Im z")
fig.colorbar(cs, ax=axs[1])
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_r5_trap_gauss.png"), dpi=120)

fig, axs = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
for col, n in enumerate(["P2", "P4C", "P6C", "P6SC"]):
    for row, k in enumerate((1.0, 0.0)):
        ax = axs[row, col]
        ax.contourf(sig, om, kd_regions[(n, k)].astype(float), levels=[0.5, 1.5], colors=["#9ec5f4"])
        ax.set_xlim(-25, 2); ax.set_ylim(0, 20)
        ax.set_title(f"KD ∘ {n}, k = {k:g}", fontsize=9)
for ax in axs[1]:
    ax.set_xlabel("Re hλ")
for ax in axs[:, 0]:
    ax.set_ylabel("Im hλ")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_r5_kd_regions.png"), dpi=110)
open(os.path.join(OUT, "r5_stability.txt"), "w").write("\n".join(lines) + "\n")
