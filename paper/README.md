# Статья: A-устойчивость композиционных методов с комплексными коэффициентами

Текст конференционной статьи (на русском, структура как у Galchenko et al., Mathematics 2026, 14, 231) по вопросу 3 исследования.

- `paper.pdf` — собранная статья.
- `src/paper.html` — исходник (HTML + KaTeX), `src/build_pdf.js` — сборка PDF.
- `src/calc.py` — все числа статьи (`data/numbers.json`) и рисунки (`figs/`); использует `composition_solver/csolver` без изменений.
- `REVIEW_NOTES.md` — замечания независимого ревью и что по ним исправлено.

Сборка из корня репозитория:

```
pip install -r requirements.txt
python3 paper/src/calc.py                      # ~2.5 мин
cd paper/src && npm install && node build_pdf.js   # нужен playwright с Chromium
```

Автор и организация в шапке — заглушки, их нужно заполнить.
