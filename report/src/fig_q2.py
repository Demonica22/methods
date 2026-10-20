"""Вопрос 2: классы задач, на которых схемы не работают, и рабочие обходы."""
import numpy as np
from scipy.linalg import expm
import common as c
from common import plt
from csolver.problems import Problem

NS = np.unique(np.round(np.geomspace(4, 1500, 22)).astype(int))


def conv_abs(prob, base, sch, pr=True):
    return c.conv(prob, c.BASES[base], c.scheme(sch), NS, project_real=pr)


# ---------- A. неаналитическая правая часть ----------
def fig_abs():
    ex = lambda: [np.exp(-2.0)]
    p_abs = Problem("abs", lambda t, y: -np.abs(y), lambda t, y: np.array([[-np.sign(y[0].real)]], complex),
                    [1.0], 2.0, exact=ex)
    p_lin = Problem("lin", lambda t, y: -y, lambda t, y: np.array([[-1.0]], complex), [1.0], 2.0, exact=ex)
    p_smooth = Problem("smooth", lambda t, y: -np.sqrt(y * y), lambda t, y: np.array([[-y[0] / np.sqrt(y[0] ** 2)]]),
                       [1.0], 2.0, exact=ex)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    # ошибка вдоль траектории
    ax = axs[0]
    N = 20; tn = np.linspace(0, 2, N + 1)
    for P, lab, ls in [(p_lin, "−y", "-"), (p_abs, "−|y|", "--")]:
        for n in ["P4C", "P4R"]:
            r = c.integrate(P, c.BASES["trapezoid"], c.scheme(n), N, return_path=True)
            ax.semilogy(tn[1:], np.abs(r["path"][1:, 0] - np.exp(-tn[1:])), c.MRK[n] + ls, color=c.COL[n],
                        ms=4, lw=1.3, label=f"{n}, f = {lab}")
    ax.set_xlabel("t"); ax.set_ylabel("|y_n − e^(−t_n)|"); ax.legend(fontsize=7.5, ncol=2)
    ax.set_title("Решение e⁻ᵗ одно, ошибка разная (N = 20)", fontsize=10)
    for ax, P, ttl in [(axs[1], p_lin, "y' = −y (аналитическая f): работает"),
                       (axs[2], p_abs, "y' = −|y| (неаналитическая f): порядок 2")]:
        for n in ["P2", "P4C", "P4R", "P6C", "P6R", "P6SC"]:
            d = conv_abs(P, "trapezoid", n)
            if len(d):
                ax.loglog(d[:, 0], d[:, 1], c.MRK[n] + "-", color=c.COL[n], ms=4, lw=1.4,
                          label=n + ("" if np.isnan(c.slope(d[:, 0], d[:, 1], 1e-13)) else f" (наклон {c.slope(d[:, 0], d[:, 1], 1e-13):.1f})"))
        ax.set_title(ttl, fontsize=10); ax.set_xlabel("шаг h"); ax.legend(fontsize=7, loc="lower right")
        ax.axhspan(1e-17, 3e-13, color="#e1e0d9", alpha=0.6, lw=0); ax.set_ylim(1e-16, 1)
    axs[1].set_ylabel("ошибка в T = 2")
    fig.suptitle("Класс A: |y| не имеет аналитического продолжения, на комплексной стадии вычисляется не аналитическое продолжение f (опора: трапеции)",
                 fontsize=10.5)
    fig.tight_layout()
    c.save(fig, "fig_q2_abs.png")


# ---------- E. комплекснозначная ОДУ ----------
def fig_complex():
    ex = lambda: [1 / (1 - 1j)]
    pc = Problem("cplx", lambda t, y: 1j * y * y, lambda t, y: np.array([[2j * y[0]]]), [1.0], 1.0)
    pc.reference = lambda: np.array([1 / (1 - 1j)])
    # та же задача в вещественной форме: y = u + i v
    pr = Problem("real", lambda t, y: np.array([-2 * y[0] * y[1], y[0] ** 2 - y[1] ** 2]),
                 lambda t, y: np.array([[-2 * y[1], -2 * y[0]], [2 * y[0], -2 * y[1]]]), [1.0, 0.0], 1.0,
                 exact=lambda: [0.5, 0.5])

    def err_c(sch, N):
        r = c.integrate(pc, c.BASES["trapezoid"], c.scheme(sch), int(N), project_real=False)
        return abs(r["y_complex"][0] - 1 / (1 - 1j))

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axs[0]
    tt = np.linspace(0, 1, 200)
    yex = 1 / (1 - 1j * tt)
    ax.plot(tt, yex.real, color="#898781", label="Re y точно")
    ax.plot(tt, yex.imag, "--", color="#898781", label="Im y точно")
    r = c.integrate(pc, c.BASES["trapezoid"], c.scheme("P6SC"), 8, project_real=False, return_path=True)
    # path хранит только Re, поэтому пройдём шагами вручную
    y = np.array([1.0 + 0j]); ys = [y[0]]; h = 1 / 8; t = 0.0
    for _ in range(8):
        y = c.scheme("P6SC").step(c.BASES["trapezoid"], pc, t, y, h); t += h; ys.append(y[0])
    ys = np.array(ys); tn = np.linspace(0, 1, 9)
    ax.plot(tn, ys.real, "P", color=c.COL["P6SC"], ms=7, label="P6SC, N = 8: Re")
    ax.plot(tn, ys.imag, "P", color=c.COL["P6C"], ms=7, label="P6SC, N = 8: Im")
    ax.set_xlabel("t"); ax.set_title("y' = i·y², y(0) = 1, точно y = 1/(1 − it)", fontsize=10)
    ax.legend(fontsize=7.5)
    ax = axs[1]
    for n in ["P3C", "P4C", "P6C", "P6SC"]:
        hs, es = [], []
        for N in NS:
            e = err_c(n, N)
            if e < 3e-14:
                break
            hs.append(1 / N); es.append(e)
        hs, es = np.array(hs), np.array(es)
        ax.loglog(hs, es, c.MRK[n] + "-", color=c.COL[n], ms=4, lw=1.4,
                  label=f"{n} (наклон {c.slope(hs, es, 1e-13):.1f})")
    ax.set_title("Комплексная задача, Re брать нельзя: P3C → 3, P6SC → 3", fontsize=10)
    ax.set_xlabel("шаг h"); ax.set_ylabel("|ошибка| в T = 1"); ax.legend(fontsize=7.5, loc="lower right")
    ax.set_ylim(1e-16, 1)
    ax = axs[2]
    for n in ["P3C", "P4C", "P6C", "P6SC"]:
        d = c.conv(pr, c.BASES["trapezoid"], c.scheme(n), NS)
        ax.loglog(d[:, 0], d[:, 1], c.MRK[n] + "-", color=c.COL[n], ms=4, lw=1.4,
                  label=f"{n} (наклон {c.slope(d[:, 0], d[:, 1], 1e-13):.1f})")
    ax.set_title("Обход: та же задача как система (u, v) ∈ ℝ²", fontsize=10)
    ax.set_xlabel("шаг h"); ax.legend(fontsize=7.5, loc="lower right"); ax.set_ylim(1e-16, 1)
    for a in axs[1:]:
        a.axhspan(1e-17, 3e-13, color="#e1e0d9", alpha=0.6, lw=0)
    fig.suptitle("Класс E (найден ревью): комплекснозначные ОДУ ломают схемы, которые опираются на проекцию Re (опора: трапеции)",
                 fontsize=10.5)
    fig.tight_layout()
    c.save(fig, "fig_q2_complex.png")


# ---------- C. слабо затухающий осциллятор, крупный шаг ----------
def osc(om=40.0, z=0.001, T=20.0):
    A = np.array([[0, 1], [-om ** 2, -2 * z * om]])
    return Problem("osc40", lambda t, y: A @ y, lambda t, y: A.astype(complex), [1.0, 0.0], T,
                   exact=lambda: expm(A * T) @ np.array([1.0, 0.0]))


def fig_osc():
    om, T = 40.0, 20.0
    p = osc()
    hw = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    for n in ["P2", "P3C", "P4C", "P4R", "P6C", "P6SC"]:
        amps = []
        for q in hw:
            N = int(round(T * om / q))
            with np.errstate(all="ignore"):
                y = c.integrate(p, c.BASES["trapezoid"], c.scheme(n), N)["y"]
            a = np.hypot(y[0], y[1] / om)
            amps.append(a if np.isfinite(a) else np.nan)
        ax.semilogy(hw, amps, c.MRK[n] + "-", color=c.COL[n], ms=5, label=n)
    ax.axhline(np.exp(-0.001 * om * T), color="#0b0b0b", ls=":", lw=1.2, label="точно 0.449")
    ax.set_xscale("log"); ax.set_ylim(0.3, 1e4)
    ax.set_xlabel("hω (шаг в долях периода быстрой моды ×2π)"); ax.set_ylabel("амплитуда в t = T")
    ax.set_title("Амплитуда после T = 20 при разных шагах", fontsize=10); ax.legend(fontsize=7.5, ncol=2)
    ax = axs[1]
    tt = np.linspace(0, T, 4000)
    ax.plot(tt, np.exp(-0.001 * om * tt), color="#898781", lw=1, label="огибающая точного решения")
    for n, q in [("P4C", 10), ("P6SC", 10), ("P6C", 15)]:
        N = int(round(T * om / q))
        with np.errstate(all="ignore"):
            r = c.integrate(p, c.BASES["trapezoid"], c.scheme(n), N, return_path=True)
        amp = np.hypot(r["path"][:, 0], r["path"][:, 1] / om)
        ax.semilogy(np.linspace(0, T, N + 1), amp, c.MRK[n] + "-", color=c.COL[n], ms=3, lw=1.2,
                    label=f"трапеции + {n}, hω = {q}")
    ax.set_ylim(0.3, 1e3)
    ax.set_xlabel("t"); ax.set_title("Огибающая |x, x′/ω| по шагам", fontsize=10); ax.legend(fontsize=7.5)
    fig.suptitle("Класс C: x″ + 2ζωx′ + ω²x = 0, ω = 40, ζ = 0.001. P4C и P6C раскачивают неразрешённую быструю моду",
                 fontsize=10.5)
    fig.tight_layout()
    c.save(fig, "fig_q2_osc.png")


# ---------- D. KD зависит от формы записи ----------
def lin_sys(A, T=20.0):
    A = np.asarray(A, float)
    return Problem("lin", lambda t, y: A @ y, lambda t, y: A.astype(complex), [1.0, 0.0], T,
                   exact=lambda: expm(A * T) @ np.array([1.0, 0.0]))


def fig_kd_form():
    s, w = -5.0, 1.5
    pn = lin_sys([[s, w], [-w, s]])              # нормальная форма, k = 1
    pf = lin_sys([[0, 1], [-(s * s + w * w), 2 * s]])   # форма Фробениуса, k = 0
    h = 0.6; N = int(round(20 / h)); tn = np.linspace(0, 20, N + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.9))
    ax = axs[0]
    for P, b, n, lab, k in [(pn, "KD", "P2", "KD, нормальная форма (k = 1)", "P4R"),
                            (pf, "KD", "P2", "KD, форма Фробениуса (k = 0)", "P6C"),
                            (pf, "trapezoid", "P2", "трапеции, форма Фробениуса", "P2"),
                            (pf, "KD", "P6C", "KD + P6C, форма Фробениуса", "P4C")]:
        with np.errstate(all="ignore"):
            r = c.integrate(P, c.BASES[b], c.scheme(n), N, return_path=True)
        nr = np.maximum(np.max(np.abs(r["path"]), axis=1), 1e-300)
        ax.semilogy(tn, nr, c.MRK[k] + "-", color=c.COL[k], ms=3, lw=1.3, label=lab)
    ax.semilogy(tn, np.exp(s * tn), ":", color="#0b0b0b", lw=1.2, label="точно ∝ e⁻⁵ᵗ")
    ax.set_ylim(1e-50, 1e30); ax.set_xlabel("t"); ax.set_ylabel("max |yᵢ(t)|")
    ax.set_title(f"λ = −5 ± 1.5i, шаг h = {h}", fontsize=10); ax.legend(fontsize=7.5)
    # область устойчивости KD при k = 0 и k = 1 + точка hλ
    ax = axs[1]
    from csolver.stability import amplification
    X, Y = np.meshgrid(np.linspace(-16.5, 1, 500), np.linspace(-13.5, 13.5, 420))
    for k, col, lab in [(1.0, c.COL["P4R"], "KD, k = 1"), (0.0, c.COL["P6C"], "KD, k = 0")]:
        R = amplification("KD", [1.0], X, Y, k=k)
        ax.contourf(X, Y, (R <= 1.0).astype(float), levels=[0.5, 1.5], colors=[col], alpha=0.25)
        ax.contour(X, Y, R, levels=[1.0], colors=[col], linewidths=1.5)
        ax.plot([], [], color=col, label=lab)
    R = amplification("KD", c.scheme("P6C").gammas, X, Y, k=0.0)
    ax.contour(X, Y, R, levels=[1.0], colors=[c.COL["P4C"]], linewidths=1.5, linestyles="--")
    ax.plot([], [], "--", color=c.COL["P4C"], label="KD + P6C, k = 0")
    for hh, m in [(0.25, "o"), (0.6, "s"), (1.0, "^"), (2.5, "D")]:
        ax.plot([s * hh, s * hh], [w * hh, -w * hh], m, color="#0b0b0b", ms=5)
        ax.annotate(f"h = {hh}", (s * hh, -w * hh), textcoords="offset points", xytext=(-14, -13), fontsize=7.5)
    ax.set_xlabel("Re hλ"); ax.set_ylabel("Im hλ"); ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("Граница ρ = 1 и положение hλ = h(−5 ± 1.5i)", fontsize=10)
    fig.suptitle("Класс D: KD зависит от формы записи системы (одна и та же динамика, разные координаты)", fontsize=10.5)
    fig.tight_layout()
    c.save(fig, "fig_q2_kd_form.png")


# ---------- B. правая часть из таблицы ----------
def fig_table():
    from scipy.interpolate import CubicSpline
    tk = np.linspace(-0.5, 3.5, 4001)         # таблица g(t) = cos t с шагом 1e-3, гладкий сплайн
    spl = CubicSpline(tk, np.cos(tk))
    ex = lambda: [0.5 * (np.cos(3.0) + np.sin(3.0)) + 0.5 * np.exp(-3.0)]   # y' = -y + cos t, y(0) = 1
    mk = lambda g: Problem("tab", lambda t, y: -y + g(t), lambda t, y: np.array([[-1.0 + 0j]]), [1.0], 3.0, exact=ex)
    p_an = mk(lambda t: np.cos(t))
    p_tab = mk(lambda t: spl(np.real(t)))   # табличная функция принимает только вещественный t (как np.interp с complex128)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axs[0]
    tt = np.linspace(0, 3, 300)
    ax.plot(tt, 0.5 * (np.cos(tt) + np.sin(tt)) + 0.5 * np.exp(-tt), color="#898781", label="точное решение")
    for P, lab, k in [(p_an, "cos t аналитически", "P4C"), (p_tab, "cos t из таблицы", "P6C")]:
        r = c.integrate(P, c.BASES["trapezoid"], c.scheme("P4C"), 6, return_path=True)
        ax.plot(np.linspace(0, 3, 7), r["path"][:, 0], c.MRK[k], color=c.COL[k], ms=7, label=f"P4C, N = 6: {lab}")
    ax.set_xlabel("t"); ax.set_ylabel("y"); ax.legend(fontsize=7.5)
    ax.set_title("y′ = −y + g(t), g = cos t", fontsize=10)
    for ax, P, ttl in [(axs[1], p_an, "g(t) аналитическая: работает"), (axs[2], p_tab, "g(t): сплайн по таблице, только вещественный t")]:
        for n in ["P2", "P4C", "P4R", "P6C", "P6R"]:
            d = conv_abs(P, "trapezoid", n)
            sl = c.slope(d[:, 0], d[:, 1], 1e-11)
            ax.loglog(d[:, 0], d[:, 1], c.MRK[n] + "-", color=c.COL[n], ms=4, lw=1.4,
                      label=n + ("" if np.isnan(sl) else f" (наклон {sl:.1f})"))
        ax.set_title(ttl, fontsize=10); ax.set_xlabel("шаг h"); ax.legend(fontsize=7, loc="lower right")
        ax.axhspan(1e-17, 1e-11, color="#e1e0d9", alpha=0.6, lw=0); ax.set_ylim(1e-15, 1)
    axs[1].set_ylabel("ошибка в T = 3")
    fig.suptitle("Класс B: табличная правая часть принимает только вещественный t. Комплексные схемы теряют порядок, вещественные P4R/P6R работают",
                 fontsize=10.5)
    fig.tight_layout()
    c.save(fig, "fig_q2_table.png")


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    fig_abs(); fig_complex(); fig_osc(); fig_kd_form(); fig_table()
