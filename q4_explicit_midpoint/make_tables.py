"""Таблицы коэффициентов (50 знаков) из data/best.json -> coeffs.md, coeffs_50digits.json."""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
best = json.load(open(os.path.join(HERE, "data", "best.json")))
best.sort(key=lambda b: (b["order"], not b["project_real"], ["gen", "conj", "pal"].index(b["struct"])))
SR = {"gen": "без симметрии", "pal": "палиндром (γ_{m+1−i} = γ_i)", "conj": "симметрично-сопряжённая (γ_{m+1−i} = conj γ_i)"}
L = ["# Коэффициенты композиции для явной средней точки (50 знаков)\n",
     "Шаг: Ψ_h = Φ_{γ_m h} ∘ … ∘ Φ_{γ_1 h}, Φ_h(y) = y + h f(t + h/2, y + h/2 f(t, y)), Σγ = 1. "
     "«С Re» — после каждого полного шага композиции берётся Re y (вещественная задача). "
     "«Без Re» — решение остаётся комплексным, условия выполнены и для мнимых частей.\n",
     "Дефект — max |Φ(τ) − 1/τ!| по деревьям порядка ≤ p для коэффициентов, округлённых до 50 знаков (mpmath). "
     "Главный член ошибки — норма дефектов на деревьях порядка p+1 (меньше — точнее при том же шаге).\n"]
for b in best:
    p, s, pr, m = b["order"], b["struct"], b["project_real"], b["stages"]
    L.append(f"\n## Порядок {p}, стадий: {m}, {SR[s]}, {'с Re' if pr else 'без Re'}\n")
    note = ""
    if (p, s, pr) == (4, "gen", True):
        note = " Уточнение пришло в ту же точку, что и симметрично-сопряжённая 3-стадийная схема: при p = 4 с Re это одно и то же решение (с точностью до перестановки)."
    if (p, s, pr) == (4, "pal", True):
        note = " Точный вид: γ₁ = γ₃ = 1/2 + i/(2√3), γ₂ = −i/√3 (Re γ₂ = 0, решение особое)."
    repos = "нет (Re γ₂ = 0)" if (p, s, pr) == (4, "pal", True) else ("да" if b["all_re_pos"] else "нет")
    L.append(f"- дефект условий: {b['defect']}; главный член ошибки: {b['lead_err']:.2e}; "
             f"max|γ| = {b['max_abs']:.4f}; все Re γ > 0: {repos}; различных решений в поиске: {b['n_solutions']}.{note}\n")
    L.append("| k | Re γ_k | Im γ_k |\n|---|---|---|")
    for k, (a, c) in enumerate(b["gammas"], 1):
        L.append(f"| {k} | {a} | {c} |")
open(os.path.join(HERE, "coeffs.md"), "w").write("\n".join(L) + "\n")
json.dump([{k: b[k] for k in ("order", "struct", "project_real", "stages", "gammas")} for b in best],
          open(os.path.join(HERE, "coeffs_50digits.json"), "w"), indent=1)
