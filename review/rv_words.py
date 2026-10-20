"""Точные условия порядка композиций через свободную ассоциативную алгебру (слова).

Опора: Φ_{γh} = exp(Σ_{k∈{1}∪K} (γh)^k F_k), F_k - свободные (некоммутирующие) символы,
K = {3,5,7} для симметричной опоры, {3,5,6,7} для Gragg-2 (F_4 = 0, см. r1), {3,4,5,6,7} для общей.
Коэффициент слова w в exp(X) равен γ^{deg w}/|w|!. Для композиции Ψ = E_m ... E_1 коэффициент
слова - сумма по разбиениям слова на m последовательных (возможно пустых) кусков.
Порядок p: коэффициенты Ψ совпадают с exp(hF_1) для всех слов степени ≤ p
(с проекцией Re - совпадают вещественные части; F_k вещественны, поэтому для нелинейной задачи
Re(Ψ_h(y)) = Σ Re(c_w) D_w Id (y) и условие то же).
Никаких случайных матриц: это точная проверка (в т.ч. в mpmath).
"""
import itertools, math
import numpy as np
import mpmath as mp

MODELS = {"sym": (3, 5, 7), "gragg2": (3, 5, 6, 7), "general": (3, 4, 5, 6, 7)}


def words(K, p):
    letters = (1,) + tuple(k for k in K if k <= p)
    out = []

    def rec(w, d):
        if w:
            out.append(tuple(w))
        for k in letters:
            if d + k <= p:
                rec(w + [k], d + k)
    rec([], 0)
    return out


class WordSet:
    def __init__(self, K, p):
        self.K, self.p = K, p
        self.W = words(K, p)
        L = max(len(w) for w in self.W)
        self.L = L
        nW = len(self.W)
        self.deg = np.array([sum(w) for w in self.W])
        self.len = np.array([len(w) for w in self.W])
        # D[w, j, j'] = deg(w[j:j']), valid if j<=j'<=len(w)
        D = np.zeros((nW, L + 1, L + 1), dtype=int)
        V = np.zeros((nW, L + 1, L + 1), dtype=bool)
        F = np.zeros((nW, L + 1, L + 1))
        for a, w in enumerate(self.W):
            for j in range(len(w) + 1):
                for jj in range(j, len(w) + 1):
                    D[a, j, jj] = sum(w[j:jj])
                    V[a, j, jj] = True
                    F[a, j, jj] = 1.0 / math.factorial(jj - j)
        self.D, self.V, self.F = D, V, F
        self.target = np.array([1.0 / math.factorial(len(w)) if all(k == 1 for k in w) else 0.0 for w in self.W])

    def target_mp(self, a):
        w = self.W[a]
        return mp.mpf(1) / mp.factorial(len(w)) if all(k == 1 for k in w) else mp.mpf(0)

    def coeffs(self, gammas):
        """numpy complex: коэффициенты всех слов композиции (порядок стадий γ_1 первой)."""
        nW, L = len(self.W), self.L
        dp = np.zeros((nW, L + 1), dtype=complex)
        dp[:, 0] = 1.0
        # самый левый кусок слова принадлежит последней стадии γ_m
        for g in gammas[::-1]:
            M = np.where(self.V, self.F * np.power(complex(g), self.D), 0.0)
            dp = np.einsum("wj,wjk->wk", dp, M)
        return dp[np.arange(nW), self.len]

    def coeffs_mp(self, gammas):
        out = []
        for a, w in enumerate(self.W):
            L = len(w)
            dp = [mp.mpc(1)] + [mp.mpc(0)] * L
            for g in gammas[::-1]:
                new = [mp.mpc(0)] * (L + 1)
                for j in range(L + 1):
                    if dp[j] == 0:
                        continue
                    for jj in range(j, L + 1):
                        new[jj] += dp[j] * mp.power(g, int(self.D[a, j, jj])) / math.factorial(jj - j)
                dp = new
            out.append(dp[L])
        return out

    def defects(self, gammas, proj, use_mp=False):
        """max |дефект| по степеням 1..p."""
        c = self.coeffs_mp(gammas) if use_mp else self.coeffs(gammas)
        out = [0.0] * (self.p + 1)
        for a, w in enumerate(self.W):
            d = c[a] - (self.target_mp(a) if use_mp else self.target[a])
            v = abs(mp.re(d) if use_mp else d.real) if proj else abs(d)
            out[self.deg[a]] = max(out[self.deg[a]], v)
        return out[1:]

    def residual(self, gammas, proj):
        d = self.coeffs(gammas) - self.target
        return d.real if proj else np.concatenate([d.real, d.imag])

    def residual_mp(self, gammas, proj):
        c = self.coeffs_mp(gammas)
        r = []
        for a in range(len(self.W)):
            d = c[a] - self.target_mp(a)
            r.append(mp.re(d))
            if not proj:
                r.append(mp.im(d))
        return r


def order_of(gammas, model, proj, pmax=7, tol=1e-10, use_mp=False):
    ws = WordSet(MODELS[model], pmax)
    d = ws.defects(gammas, proj, use_mp)
    for j, v in enumerate(d):
        if v > tol:
            return j, d
    return pmax, d
