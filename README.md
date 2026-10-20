# Композиционные схемы с комплексными коэффициентами для ОДУ

Решатель ОДУ на основе композиционных схем с комплексными коэффициентами 2, 4 и 6-го порядка
с разными опорными методами второго порядка:

- метод трапеций (неявный, симметричный);
- метод KD (CD, композиционный диагональный, полуявный, симметричный при s = 0.5);
- модифицированная явная средняя точка Грэгга со сглаживающим шагом (почти симметричный).

Вопросы исследования:

1. Для каких опорных методов достигается заявленный порядок точности?
2. Существует ли класс нежёстких задач, которые такими методами не решаются?
3. Является ли композиция A-устойчивого метода A-устойчивой?
4. Существуют ли коэффициенты композиции, при которых нужный порядок достигается для несимметричного
   или почти симметричного опорного метода (средней точки)?

Краткие ответы — в [`composition_solver/RESEARCH.md`](composition_solver/RESEARCH.md),
полный текст — в отчёте [`report/report.pdf`](report/report.pdf) и презентации
[`report/presentation.pptx`](report/presentation.pptx).

## Структура репозитория

```
composition_solver/          решатель и основные эксперименты
├── csolver/                 пакет решателя
│   ├── base_methods.py      опорные методы (трапеции, KD, средняя точка Грэгга)
│   ├── compositions.py      композиционные схемы P2, P3C, P4C/P4R, P6C/P6R, P6SC и найденные
│   ├── solver.py            интегрирование с постоянным шагом
│   ├── problems.py          8 тестовых нежёстких задач
│   ├── order_conditions.py  условия порядка и поиск коэффициентов
│   └── stability.py         функции устойчивости
├── experiments/             скрипты экспериментов (по вопросам 1–4)
├── results/                 результаты: csv, md-таблицы, графики
├── README.md                описание решателя и результатов
└── RESEARCH.md              ответы на вопросы исследования

review/                      независимая проверка результатов (свой код, без csolver)
├── r1…r6_*.py               проверки: опоры, условия порядка, локальный/глобальный порядок,
│                            устойчивость, классы задач, на которых схемы не работают
├── rv_common.py, rv_words.py  общая независимая реализация (numpy + mpmath)
├── out/                     текстовые выводы проверок
├── data/                    данные (в т. ч. уточнённые коэффициенты polished.json)
└── REVIEW.md                итог ревью

kd_article_repro/            воспроизведение 2D-рисунков 1–9 статьи о методе KD
├── kd_repro.py              построение рисунков → figs/
├── compare.py               сравнение «статья | воспроизведение» → compare/
├── figs/, compare/, article_crops/
└── README.md                что сравнивалось и расхождения

report/                      отчёт и презентация
├── report.pdf               итоговый отчёт
├── presentation.pptx        презентация
├── figs/, data/             графики и данные отчёта
└── src/                     скрипты графиков (fig_q1…fig_q4.py), report.html, сборка PDF/PPTX

notes/kd_method_summary.md   конспект статьи о методе KD
```

## Установка

Python 3.10+:

```bash
pip install -r requirements.txt
```

Для сборки PDF и презентации дополнительно нужен Node.js: `npm install playwright pptxgenjs`.

## Как запускать

Все скрипты сами находят пути относительно своего расположения, запускать можно из любой папки.

**Решатель из Python:**

```python
import sys; sys.path.insert(0, "composition_solver")
from csolver import integrate, BASE_METHODS, SCHEMES, problems

p = problems.van_der_pol()
r = integrate(p, BASE_METHODS["KD"], SCHEMES["P6C"], 100)   # 100 шагов на [0, 10]
print(r["y"], r["nf"])
```

**Эксперименты** (результаты пишутся в `composition_solver/results/`):

```bash
cd composition_solver
python experiments/order_convergence.py   # вопрос 1: порядок сходимости (~3.5 мин)
python experiments/diagnostics.py         # симметричность опор, проекция Re
python experiments/failure_classes.py     # вопрос 2: задачи, на которых схемы не работают
python experiments/stability_regions.py   # вопрос 3: области устойчивости
python experiments/conj_search.py         # вопрос 3: симметрично-сопряжённая схема 6-го порядка
python experiments/coeff_search.py        # вопрос 4: поиск коэффициентов (~12 мин)
python experiments/verify_found.py        # вопрос 4: проверка найденных схем
```

**Ревью** (результаты в `review/out/`, `review/data/`):

```bash
cd review
python r1_base_methods.py
python r2_order_conditions.py      # пишет data/polished.json, нужен для r3 found и отчёта
python r3_local_order.py && python r3_local_order.py found
python r4_global_order.py
python r5_stability.py
python r6_failure.py
```

**Отчёт** (графики в `report/figs/`):

```bash
cd report/src
python fig_q1.py; python fig_q2.py; python fig_q3.py; python fig_q4.py
node build_pdf.js     # report/report.pdf
node build_pptx.js    # report/presentation.pptx
```

**Статья о KD:**

```bash
python kd_article_repro/kd_repro.py      # рисунки → kd_article_repro/figs/
```

`compare.py` нужен только для пересборки сравнений и требует страницы статьи в PNG
(`pdftoppm -r 130 -png article.pdf kd_article_repro/pdf/r`, путь можно задать переменной `KD_PAGES`).

## Статья о методе KD

M. Galchenko, V. Sheptunova, A. Karimov, E. Kopets, S. S. Muni, D. Butusov.
*New Approaches to Stability Analysis for Semi-Implicit Numerical Integration Methods.*
Mathematics 2026, 14, 231. [doi:10.3390/math14020231](https://doi.org/10.3390/math14020231).

Сам PDF статьи в репозиторий не включён; в `kd_article_repro/article_crops/` лежат только
вырезки рисунков для сравнения со ссылкой на источник.
