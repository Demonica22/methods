"""Уточнение найденных коэффициентов до высокой точности (mpmath, Гаусс–Ньютон).

Для недоопределённых систем (семейства решений) шаг — псевдорешение минимальной нормы,
поэтому точка сдвигается к ближайшему точному решению. Структура (gen/pal/conj)
сохраняется: уточняются только свободные параметры.
"""
import mpmath as mp
import em_trees as E


def expand_mp(x, m, struct):
    q = len(x) // 2
    z = [mp.mpc(x[i], x[q + i]) for i in range(q)]
    if struct == "gen":
        return z
    if struct == "pal":
        return z + z[:m // 2][::-1]
    if m % 2:
        z[-1] = mp.mpc(z[-1].real, 0)
    return z + [mp.conj(w) for w in z[:m // 2][::-1]]


def residual_mp(x, m, struct, p, proj):
    w = E.weights_mp(expand_mp(x, m, struct), p)
    t = [mp.mpf(1) / E.factorial(tr) for n in range(1, p + 1) for tr in E.TREES[n]]
    d = [a - b for a, b in zip(w, t)]
    return [v.real for v in d] + ([] if proj else [v.imag for v in d])


def params_from(g, m, struct):
    q = m if struct == "gen" else (m + 1) // 2
    return [mp.mpf(z.real) for z in g[:q]] + [mp.mpf(z.imag) for z in g[:q]]


def polish(g, struct, p, proj, dps=60, iters=40):
    mp.mp.dps = dps
    m = len(g)
    x = params_from(g, m, struct)
    if struct == "conj" and m % 2:
        x[len(x) - 1] = mp.mpf(0)          # мнимая часть среднего коэффициента
    free = [i for i in range(len(x)) if not (struct == "conj" and m % 2 and i == len(x) - 1)]
    eps = mp.mpf(10) ** (-(3 * dps) // 5)
    for _ in range(iters):
        r = mp.matrix(residual_mp(x, m, struct, p, proj))
        if mp.norm(r, mp.inf) < mp.mpf(10) ** (-dps + 8):
            break
        J = mp.matrix(len(r), len(free))
        for jj, i in enumerate(free):
            xp = list(x); xp[i] += eps
            rp = residual_mp(xp, m, struct, p, proj)
            for k in range(len(r)):
                J[k, jj] = (rp[k] - r[k]) / eps
        # псевдорешение через SVD (ранг определяем по порогу)
        U, S, V = mp.svd_r(J)
        tol = max(S) * mp.mpf(10) ** (-(2 * dps) // 5)
        dx = mp.matrix(len(free), 1)
        for k in range(len(S)):
            if S[k] > tol:
                c = sum(U[i, k] * r[i] for i in range(len(r))) / S[k]
                for jj in range(len(free)):
                    dx[jj] += c * V[k, jj]
        for jj, i in enumerate(free):
            x[i] -= dx[jj]
    g = expand_mp(x, m, struct)
    res = mp.norm(mp.matrix(residual_mp(x, m, struct, p, proj)), mp.inf)
    return g, res
