## Допуск → ошибка (8 базовых задач)

| опора | пара | медиана наклона err(tol) | медиана err/tol | max err/tol | доля отброшенных шагов |
|---|---|---|---|---|---|
| trapezoid | P4C/P2 | 1.33 | 1.0e-03 | 7.3e-01 | 2.4% |
| trapezoid | P6C/P4C | 1.16 | 5.7e-04 | 2.8e-02 | 6.0% |
| trapezoid | P6SC/P4C | 1.22 | 6.6e-03 | 7.0e-01 | 6.1% |
| KD | P4C/P2 | 1.33 | 5.9e-04 | 2.6e-01 | 2.0% |
| KD | P6C/P4C | 1.14 | 4.2e-04 | 6.5e-01 | 6.3% |
| KD | P6SC/P4C | 1.22 | 4.9e-03 | 1.4e+00 | 6.3% |
| midpoint | P4C/P2 | 1.33 | 1.7e-03 | 3.7e-01 | 2.2% |
| midpoint | P6C/P4C | 1.02 | 2.8e-03 | 4.8e-01 | 6.0% |
| midpoint | P6SC/P4C | 1.00 | 5.4e-02 | 6.7e-01 | 6.1% |

## Выигрыш адаптивного шага по числу вычислений f

Отношение nf(постоянный шаг) / nf(адаптивный) при одинаковой ошибке в t = T; схема высокого порядка та же, у адаптивного решателя в nf входят оценщик и отброшенные шаги. «—»: уровень не достигнут одним из решателей.

| задача | опора | пара | выигрыш по f при ε = 1e-4 | ε = 1e-6 | ε = 1e-9 |
|---|---|---|---|---|---|
| kepler_e09 | trapezoid | P4C/P2 | 10.54× | — | — |
| kepler_e09 | trapezoid | P6C/P4C | 10.12× | 8.50× | — |
| kepler_e09 | trapezoid | P6SC/P4C | 6.37× | 7.12× | — |
| kepler_e09 | KD | P4C/P2 | 8.23× | — | — |
| kepler_e09 | KD | P6C/P4C | 8.19× | 8.77× | 11.79× |
| kepler_e09 | KD | P6SC/P4C | 6.62× | 8.54× | — |
| kepler_e09 | midpoint | P4C/P2 | 14.64× | — | — |
| kepler_e09 | midpoint | P6C/P4C | 9.86× | 8.11× | 9.66× |
| kepler_e09 | midpoint | P6SC/P4C | 8.61× | 9.35× | — |
| vanderpol_mu5 | trapezoid | P4C/P2 | — | 1.48× | 1.62× |
| vanderpol_mu5 | trapezoid | P6C/P4C | 1.49× | 2.01× | 1.32× |
| vanderpol_mu5 | trapezoid | P6SC/P4C | — | 1.40× | 1.52× |
| vanderpol_mu5 | KD | P4C/P2 | — | 2.36× | 2.80× |
| vanderpol_mu5 | KD | P6C/P4C | — | 1.36× | 1.32× |
| vanderpol_mu5 | KD | P6SC/P4C | — | 1.83× | 1.87× |
| vanderpol_mu5 | midpoint | P4C/P2 | — | 2.10× | 2.86× |
| vanderpol_mu5 | midpoint | P6C/P4C | — | 1.29× | 1.83× |
| vanderpol_mu5 | midpoint | P6SC/P4C | — | 1.69× | 1.99× |
| kepler | trapezoid | P4C/P2 | 1.21× | 1.41× | 1.15× |
| kepler | trapezoid | P6C/P4C | — | 1.52× | 1.74× |
| kepler | trapezoid | P6SC/P4C | 1.05× | 1.06× | 1.26× |
| kepler | KD | P4C/P2 | 1.04× | 1.13× | 1.11× |
| kepler | KD | P6C/P4C | — | 1.14× | 1.50× |
| kepler | KD | P6SC/P4C | 1.05× | 1.23× | 1.55× |
| kepler | midpoint | P4C/P2 | 1.21× | 1.54× | 1.51× |
| kepler | midpoint | P6C/P4C | — | 1.14× | 1.12× |
| kepler | midpoint | P6SC/P4C | 1.11× | 1.13× | 1.58× |
| vanderpol | trapezoid | P4C/P2 | — | 0.60× | 0.69× |
| vanderpol | trapezoid | P6C/P4C | — | 0.55× | 0.40× |
| vanderpol | trapezoid | P6SC/P4C | — | 0.45× | 0.39× |
| vanderpol | KD | P4C/P2 | — | 0.93× | 1.28× |
| vanderpol | KD | P6C/P4C | — | 0.59× | 0.54× |
| vanderpol | KD | P6SC/P4C | — | 0.52× | 0.66× |
| vanderpol | midpoint | P4C/P2 | — | 0.80× | 1.25× |
| vanderpol | midpoint | P6C/P4C | — | 0.41× | 0.52× |
| vanderpol | midpoint | P6SC/P4C | 0.45× | 0.65× | 0.82× |
