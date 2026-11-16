// Презентация по итогам проекта (pptxgenjs, структурированная колода).
const pptxgen = require("pptxgenjs");
const path = require("path");
// Тема оформления применяется скриптом apply_theme.js из pptx-скилла Claude; путь к нему можно задать
// через PPTX_APPLY_THEME. Без него презентация собирается со стандартной темой.
let applyTheme = null;
try { applyTheme = require(process.env.PPTX_APPLY_THEME || "./apply_theme.js").applyTheme; } catch (e) { /* без темы */ }
const SIZES = require("./figsizes.json");
const FIG = path.resolve(__dirname, "..", "figs");
const OUT = path.resolve(__dirname, "..", "presentation.pptx");

const THEME = {
  name: "Complex Composition",
  headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "141327", lt1: "FFFFFF", dk2: "2B2950", lt2: "F3F2FB",
    accent1: "6250D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "2A78D6",
    accent5: "E87BA4", accent6: "EDA100", hlink: "2A78D6", folHlink: "6250D6" },
};
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";        // 13.33 x 7.5
pres.title = "Композиционные схемы с комплексными коэффициентами";
pres.author = "Semyon";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, M = 0.55;

pres.defineSlideMaster({ title: "DARK", background: { color: THEME.colors.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.3, w: W - 2 * M, h: 1.6, fontSize: 40, bold: true, color: C.background1, valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.05, w: W - 2 * M, h: 1.6, fontSize: 18, color: "CADCFC".length ? C.background2 : C.background2, valign: "top" }, text: "" } },
  ] });
pres.defineSlideMaster({ title: "CONTENT", background: { color: "FFFFFF" },
  slideNumber: { x: W - 1.0, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: "898781" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.32, w: W - 2 * M, h: 0.85, fontSize: 28, bold: true, color: C.text1, valign: "middle", align: "left" }, text: "" } },
    { text: { text: "Композиционные схемы с комплексными коэффициентами", options: { x: M, y: 7.05, w: 7, h: 0.3, fontSize: 10, color: "898781" } } },
  ] });

function slide(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
// картинка, вписанная в прямоугольник (x, y, w, h), по центру прямоугольника
function img(s, name, x, y, w, h, align = "center") {
  const [pw, ph] = SIZES[name]; const r = pw / ph;
  let iw = w, ih = w / r;
  if (ih > h) { ih = h; iw = h * r; }
  const ix = align === "left" ? x : x + (w - iw) / 2;
  s.addImage({ path: path.join(FIG, name), x: ix, y: y + (h - ih) / 2, w: iw, h: ih, objectName: name,
    altText: name.replace(".png", "") });
  return { x: ix, y: y + (h - ih) / 2, w: iw, h: ih };
}
function caption(s, text, x, y, w, h = 0.5) {
  s.addText(text, { x, y, w, h, fontSize: 12, color: C.text2, isTextBox: true, margin: 0, valign: "top" });
}
function bullets(s, items, x, y, w, h, fs = 15) {
  s.addText(items.map((t, i) => {
    const runs = Array.isArray(t) ? t : [{ text: t }];
    return runs.map((r, j) => ({ text: r.text, options: { bold: !!r.b, color: r.c || C.text1, breakLine: j === runs.length - 1 && i < items.length - 1, paraSpaceAfter: 10 } }));
  }).flat(), { x, y, w, h, fontSize: fs, valign: "top", isTextBox: true, margin: 0 });
}
function card(s, x, y, w, h, head, body, color, fs = 15) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: THEME.colors.lt2 }, line: { color: THEME.colors.lt2 }, objectName: "card " + head });
  s.addShape(pres.shapes.OVAL, { x: x + 0.22, y: y + 0.22, w: 0.42, h: 0.42, fill: { color }, line: { color } });
  s.addText(head, { x: x + 0.78, y: y + 0.16, w: w - 0.95, h: 0.55, fontSize: 16, bold: true, color: C.text1, fontFace: THEME.headFontFace, isTextBox: true, margin: 0, valign: "middle" });
  s.addText(body, { x: x + 0.25, y: y + 0.8, w: w - 0.5, h: h - 0.95, fontSize: fs, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
}
const HDR = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: THEME.colors.dk2 }, fontSize: 12 } });
function table(s, rows, x, y, w, colW, fs = 12) {
  const data = rows.map((r, i) => r.map((c) => {
    if (i === 0) return HDR(String(c));
    const bad = typeof c === "string" && c.startsWith("!");
    const t = bad ? c.slice(1) : String(c);
    return { text: t, options: { fontSize: fs, color: bad ? "B3261E" : THEME.colors.dk1, bold: bad } };
  }));
  s.addTable(data, { x, y, w, colW, border: { type: "solid", pt: 0.5, color: "E1E0D9" }, fill: { color: "FFFFFF" }, margin: 0.06, fontFace: "Calibri", rowH: 0.34 });
}

// ---------------- 1. Титул
pres.addSection({ title: "Введение" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Введение" });
s.addText("Композиционные схемы с комплексными коэффициентами", { placeholder: "title" });
s.addText("Решатель ОДУ 2, 4 и 6-го порядка на опорных методах: трапеции, KD (s = 0.5), средняя точка Грэгга со сглаживанием. Итоги по вопросам исследования 1–4 с учётом ревью · 20.10.2026", { placeholder: "body" });

// ---------------- 2. Ответы
s = slide("Четыре ответа", "Введение");
card(s, M, 1.35, 5.95, 2.6, "1. Порядок 2/4/6", "Трапеции и KD(0.5) дают ровно 2/4/6 со всеми схемами. Средняя точка Грэгга даёт 2/4/5: порядок 6 стандартные схемы не дают. Хойн и явная средняя точка: только 3.", THEME.colors.accent1);
card(s, M + 6.28, 1.35, 5.95, 2.6, "2. Где схемы не работают", "Неаналитическая f (|y|, if), f только от вещественного t (таблицы), комплекснозначные ОДУ, KD в форме Фробениуса, раскачка неразрешённой быстрой моды.", THEME.colors.accent2);
card(s, M, 4.2, 5.95, 2.6, "3. A-устойчивость", "В общем случае не сохраняется: P4C даёт A(89.5°), P6C A(87.4°). Сохраняется для трапеций при наборе γ, замкнутом относительно сопряжения, с Re γ > 0 (P3C, P6SC).", THEME.colors.accent3);
card(s, M + 6.28, 4.2, 5.95, 2.6, "4. Коэффициенты для несимм. опоры", "Существуют. Грэгг: порядок 6 даёт схема из 5 стадий. Любая опора 2-го порядка: порядок 4 из 3 стадий, порядок 6 из 7 стадий. Проверено в 50 знаках.", THEME.colors.accent4);

// ---------------- 3. Решатель
s = slide("Решатель: m комплексных подшагов", "Введение");
bullets(s, [
  [{ text: "Ψₕ = Φ(γₘh) ∘ … ∘ Φ(γ₁h)", b: true }, { text: ", Σγ = 1; время стадии тоже комплексное" }],
  "Для вещественной задачи после шага берётся Re(y)",
  [{ text: "Трапеции, KD(0.5): ", b: true }, { text: "симметричны, дефект ≈ 1e-17" }],
  [{ text: "Грэгг, 2 подшага: ", b: true }, { text: "почти симметричен, дефект ∝ h⁶" }],
  [{ text: "Хойн: ", b: true }, { text: "дефект ∝ h⁴; KD(s ≠ 0.5): порядок 1" }],
], M, 1.45, 4.9, 4.6, 16);
img(s, "fig_symdefect.png", 5.65, 1.4, 7.15, 4.9);
caption(s, "Дефект симметрии Φ₋ₕ∘Φₕ − id определяет, какой порядок дадут композиции.", 5.65, 6.4, 7.15);

// ---------------- 4. Схемы
s = slide("Схемы: тройной прыжок и сопряжённые", "Введение");
img(s, "fig_coeffs.png", M, 1.3, W - 2 * M, 4.4);
table(s, [["P3C", "P4C", "P4R", "P6C", "P6R", "P6SC"],
          ["2 ст., 1/2 ± i/(2√3)", "3 ст., комплексный", "3 ст., γ₂ = −1.70", "9 ст., комплексный", "9 ст., вещественный", "5 ст., γ₆₋ᵢ = conj γᵢ"]],
  M, 5.95, W - 2 * M, Array(6).fill((W - 2 * M) / 6), 12);

// ---------------- 4a. Системы уравнений
function sysTable(s, rows, colW, fs) {
  const data = rows.map((r, i) => r.map((c) => i === 0 ? HDR(c) : { text: c, options: { fontSize: fs, color: THEME.colors.dk1 } }));
  s.addTable(data, { x: M, y: 1.25, w: W - 2 * M, colW, border: { type: "solid", pt: 0.5, color: "E1E0D9" }, fill: { color: "FFFFFF" }, margin: 0.05, fontFace: "Calibri", rowH: 0.52, valign: "middle" });
}
s = slide("Системы уравнений: основной набор", "Введение");
sysTable(s, [["задача", "система", "параметры и y(0)", "интервал, эталон"],
  ["Осциллятор", "y₁′ = y₂,  y₂′ = −y₁", "y(0) = (1, 0)", "[0, 10], точное"],
  ["Линейная, форма Фробениуса", "y₁′ = y₂,  y₂′ = −(σ² + ω²)y₁ + 2σy₂", "σ = −0.2, ω = 2;  y(0) = (1, 0)", "[0, 10], точное"],
  ["Логистическое", "y′ = y(1 − y)", "y(0) = 0.1", "[0, 5], точное"],
  ["Ван дер Поль", "y₁′ = y₂,  y₂′ = μ(1 − y₁²)y₂ − y₁", "μ = 1;  y(0) = (2, 0)", "[0, 10], DOP853"],
  ["Лотка–Вольтерра", "y₁′ = ay₁ − by₁y₂,  y₂′ = −cy₂ + dy₁y₂", "a = 1, b = 0.5, c = 0.75, d = 0.25;  y(0) = (2, 1)", "[0, 10], DOP853"],
  ["Кеплер", "qᵢ′ = pᵢ,  pᵢ′ = −qᵢ / r³,  r = √(q₁² + q₂²)", "e = 0.5;  y(0) = (0.5, 0, 0, √3)", "[0, 2π], y(T) = y(0)"],
  ["Лоренц", "x′ = s(y − x),  y′ = x(r − z) − y,  z′ = xy − bz", "s = 10, r = 28, b = 8/3;  (1, 1, 1)", "[0, 1], DOP853"],
  ["Вынужденный Дуффинг", "x″ + 0.2x′ + x + x³ = 0.5 cos 1.2t", "y(0) = (1, 0)", "[0, 10], DOP853"],
  ["generic3 (ревью)", "y₁′ = y₂² − sin y₃ + t,  y₂′ = exp(0.3y₁) − y₃y₁,  y₃′ = cos(ty₂) + y₁y₂", "t₀ = 0.2;  y = (0.3, −0.5, 0.7)", "T = 2, mpmath"]],
  [2.3, 5.2, 3.2, 1.53], 14);

s = slide("Системы уравнений: задачи вопросов 2 и 3", "Введение");
sysTable(s, [["класс", "запись", "параметры, y(0), интервал", "что проверяет"],
  ["2 A", "y′ = −|y|  (сравнение: y′ = −y)", "y(0) = 1, [0, 2], точно e⁻ᵗ", "неаналитическая f"],
  ["2 B", "y′ = −y + g(t), g = cos t по сплайн-таблице от Re t", "y(0) = 1, [0, 3]", "f только от вещественного t"],
  ["2 C", "x″ + 2ζωx′ + ω²x = 0", "ω = 40, ζ = 0.001; (1, 0), [0, 20]", "раскачка быстрой моды"],
  ["2 D", "y′ = [[σ, ω], [−ω, σ]]y  и  y′ = [[0, 1], [−(σ²+ω²), 2σ]]y", "σ = −5, ω = 1.5; (1, 0), [0, 20]", "KD и форма записи"],
  ["2 E", "y′ = i·y²;  вещ. форма: u′ = −2uv, v′ = u² − v²", "y(0) = 1, [0, 1], точно 1/(1 − it)", "комплекснозначная ОДУ"],
  ["3", "X′ = AX: a = 2σk/(1+k), d = 2σ/(1+k), b = 1, c = ad − σ² − ω²", "hλ = σ + iω, k = 1 и k = 0", "области устойчивости (для трапеций = y′ = λy)"],
  ["3", "r(z) трапеций и Гаусса-3, R(z) = Π r(γᵢz)", "Re z ∈ [−3, 1], Im z ∈ [0, 30]", "контрпример"],
  ["3", "y′ = A(y − g) + g′, A = [[σ, ω], [−ω, σ]], g = (sin t, cos t)", "σ = −0.5, ω = 26; (0.5, 1), [0, 30]", "зачем A-устойчивость"]],
  [0.8, 5.6, 3.6, 2.23], 14);

// ---------------- 5. Галерея
s = slide("Работающие примеры: 30–40 шагов", "Введение");
const g = img(s, "fig_gallery.png", M, 1.2, 8.9, 5.75, "left");
bullets(s, ["Точки: шаги композиции; линия: эталон DOP853",
  "Ван дер Поль, KD + P6C: ошибка 1e-8",
  "Лоренц, трапеции + P6C: 4e-10",
  "Лотка–Вольтерра, KD + P6SC: 1e-7",
  "Неавтономный Дуффинг на явном Грэгге работает с комплексным временем стадии"], g.x + g.w + 0.3, 1.5, W - M - (g.x + g.w + 0.3), 5.2, 14);

// ---------------- 6. Q1 таблица
pres.addSection({ title: "Вопрос 1" });
s = slide("Вопрос 1. Порядок задаёт симметрия опоры", "Вопрос 1");
table(s, [["опора", "P2", "P3C", "P4C", "P4R", "P6C", "P6R", "P6SC"],
          ["трапеции", "2", "4", "4", "4", "6", "6", "6"],
          ["KD (s = 0.5)", "2", "4", "4", "4", "6", "6", "6"],
          ["Грэгг + сглаживание", "2", "4", "4", "4", "!5", "!5", "!5"],
          ["Хойн / явная ср. точка", "2", "!3", "!3", "!3", "!3", "!3", "!3"]],
  M, 1.4, 6.6, [2.2, 0.62, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63], 13);
bullets(s, [
  "Грэгг: в поле есть член h⁵F₆, Σγ⁶ ≠ 0 → порядок 5",
  "P3C даёт 4 вместо 3: мнимый член h⁴ убирает Re",
  "P6SC без проекции на каждом шаге (Re в конце) тоже 6",
], 7.45, 1.45, 5.35, 2.4, 14);
img(s, "fig_local_hp.png", M, 3.9, W - 2 * M, 3.0);

// ---------------- 7. Q1 сходимость
s = slide("Вопрос 1. Сходимость: Ван дер Поль, Дуффинг", "Вопрос 1");
img(s, "fig_conv_van_der_pol.png", M, 1.2, W - 2 * M, 2.85);
img(s, "fig_conv_forced_duffing.png", M, 4.05, W - 2 * M, 2.85);

// ---------------- 8. Q1 стоимость
s = slide("Вопрос 1. Стоимость до ошибки 1e-8", "Вопрос 1");
img(s, "fig_cost_van_der_pol.png", M, 1.2, W - 2 * M, 3.4);
const stats = [["1 210", "трапеции + P6SC"], ["2 440", "KD + P6C"], ["1 180", "Грэгг + P6C"], ["19 800", "трапеции + P4R"]];
stats.forEach(([v, l], i) => {
  const x = M + i * 3.1;
  s.addText(v, { x, y: 4.85, w: 2.9, h: 1.0, fontSize: 44, bold: true, color: i === 3 ? THEME.colors.accent2 : THEME.colors.accent1, fontFace: THEME.headFontFace, isTextBox: true, margin: 0 });
  s.addText(l, { x, y: 5.85, w: 2.9, h: 0.4, fontSize: 14, color: C.text2, isTextBox: true, margin: 0 });
});
caption(s, "Ван дер Поль, T = 10. Порядки в легенде взяты из высокоточной проверки: МНК по точкам двойной точности для P6C даёт артефакты.", M, 6.4, W - 2 * M);

// ---------------- 9. Q2 обзор
pres.addSection({ title: "Вопрос 2" });
s = slide("Вопрос 2. Пять классов нежёстких сбоев", "Вопрос 2");
const cls = [["A", "Неаналитическая f", "|y|, if, max: комплексные схемы падают до порядка 2", THEME.colors.accent2],
  ["B", "f только от вещ. t", "таблица, сплайн, внешний код: порядок 2 без ошибок", THEME.colors.accent6],
  ["C", "Неразрешённая быстрая мода", "P4C/P6C раскачивают её при hω ≥ 5", THEME.colors.accent5],
  ["D", "KD и форма записи", "форма Фробениуса: область ≈ [−2,0]×[−2,2]", THEME.colors.accent4],
  ["E", "Комплекснозначные ОДУ", "Re брать нельзя: P6SC → 3, P3C → 3", THEME.colors.accent1]];
cls.forEach(([k, h, b, col], i) => {
  const y = 1.3 + i * 1.0;
  s.addShape(pres.shapes.OVAL, { x: M, y: y + 0.08, w: 0.7, h: 0.7, fill: { color: col }, line: { color: col } });
  s.addText(k, { x: M, y: y + 0.08, w: 0.7, h: 0.7, fontSize: 22, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
  s.addText(h, { x: M + 0.95, y, w: 4.4, h: 0.86, fontSize: 17, bold: true, color: C.text1, valign: "middle", isTextBox: true, margin: 0 });
  s.addText(b, { x: M + 5.4, y, w: 6.8, h: 0.86, fontSize: 15, color: C.text2, valign: "middle", isTextBox: true, margin: 0 });
});
caption(s, "Рабочие обходы: аналитическая запись f, вещественные P4R/P6R, вещественная форма комплексной задачи, A-устойчивые P3C/P6SC, композиции для KD.", M, 6.4, W - 2 * M);

s = slide("Классы A и B: неаналитическая f", "Вопрос 2");
img(s, "fig_q2_abs.png", M, 1.2, W - 2 * M, 2.85);
img(s, "fig_q2_table.png", M, 4.05, W - 2 * M, 2.85);

s = slide("Классы C и D: быстрая мода, форма записи", "Вопрос 2");
img(s, "fig_q2_osc.png", M, 1.2, W - 2 * M, 2.85);
img(s, "fig_q2_kd_form.png", M, 4.05, W - 2 * M, 2.85);

s = slide("Класс E: комплексная задача y′ = i·y²", "Вопрос 2");
img(s, "fig_q2_complex.png", M, 1.3, W - 2 * M, 3.8);
bullets(s, ["Без проекции Re P6SC и P3C дают порядок 3; P4C и P6C сохраняют 4 и 6",
  "Работает: та же задача как система (u, v) ∈ ℝ², тогда P6SC снова даёт 6"], M, 5.4, W - 2 * M, 1.3, 15);

// ---------------- Q3
pres.addSection({ title: "Вопрос 3" });
s = slide("Вопрос 3. P4C и P6C не A-устойчивы", "Вопрос 3");
table(s, [["схема", "A-устойчива", "A(α)", "max ρ на iℝ"],
          ["P2, P3C, P6SC", "да", "90°", "1"],
          ["P4C", "!нет", "89.54°", "1.018 (y ≈ 10.1)"],
          ["P6C", "!нет", "87.36°", "1.404 (y ≈ 26.3)"],
          ["P4R, P6R", "!нет", "0°", "полюс hλ ≈ −1.17"]], M, 1.4, 5.6, [1.6, 1.3, 1.0, 1.7], 13);
img(s, "fig_q3_trap_imag.png", 6.45, 1.3, 6.35, 3.5);
bullets(s, [[{ text: "Достаточное условие (трапеции): ", b: true }, { text: "набор γ замкнут относительно сопряжения и все Re γ > 0. Тогда полюсы справа, а на мнимой оси сомножители гасятся парами: |R(iy)| = 1" }],
  [{ text: "Не обобщается ", b: true }, { text: "на опоры с комплексными полюсами (Гаусс-3 ∘ P6SC неустойчива)" }]], M, 5.0, W - 2 * M, 1.9, 17);

s = slide("Вопрос 3. Почему P4C и P6C теряют A-устойчивость", "Вопрос 3");
img(s, "fig_q3_factors.png", M, 1.15, W - 2 * M, 3.9);
bullets(s, [[{ text: "Стадия сдвигает точку с оси: ", b: true }, { text: "Re(γ·iy) = −y·Im γ ≠ 0. При Im γ > 0 точка уходит влево и |r| < 1, при Im γ < 0 вправо и |r| > 1" }],
  [{ text: "У P4C нет сопряжённых стадий: ", b: true }, { text: "при y = 10.1 множители 0.711 · 1.631 · 0.711 = 0.826, при y = −10.1 произведение 1.21. Проекция Re смешивает λ и λ̄, но горб 1.018 остаётся, отсюда A(89.54°). У P6C горб 1.404, A(87.36°)" }],
  [{ text: "У P3C и P6SC ", b: true }, { text: "каждой γ соответствует γ̄, |r(γ̄iy)| = 1/|r(γiy)|, множители сокращаются: ρ ≡ 1" }]],
  M, 5.15, W - 2 * M, 1.85, 13);

s = slide("Вопрос 3. Карта ρ у мнимой оси и «усы»", "Вопрос 3");
img(s, "fig_q3_trap_regions.png", M, 1.15, 7.3, 5.8, "left");
card(s, 8.0, 1.2, 4.85, 2.75, "Шкала: log₁₀ ρ", "ρ ничем не ограничен: это множитель роста ошибки за шаг. Пример KD, k = 0: ρ(−3) = 2, ρ(−10) = 9, ρ(5i) ≈ 23. Логарифм делает шкалу симметричной вокруг ρ = 1: ±1.5 соответствует ρ от 0.03 до 30, здесь ±0.25 (0.56…1.78).", THEME.colors.accent4, 15);
card(s, 8.0, 4.1, 4.85, 2.8, "«Усы»", "Чёрное: граница ρ = 1, серое: мнимая ось. У P4C и P6C граница отходит от оси при |Im| ≈ 5 и ≈ 10 и уходит влево (Re ≈ −0.2 и −1.8 при |Im| = 40). Жёлтое: Re hλ < 0, но ρ > 1, решение затухает, а схема растёт.", "F2C200", 15);

s = slide("Вопрос 3. Работающий пример A-устойчивости", "Вопрос 3");
img(s, "fig_q3_demo.png", M, 1.2, W - 2 * M, 4.0);
bullets(s, ["y′ = A(y − g(t)) + g′(t), собственные числа −0.5 ± 26i, медленное решение g = (sin t, cos t)",
  "h = 1: hλ ≈ −0.5 + 26i попадает на горб P6C (ρ = 1.4), и схема разваливается; P6SC держит решение",
  "h = 0.5: работают все схемы"], M, 5.4, W - 2 * M, 1.5, 15);

s = slide("Вопрос 3. Контрпример: Гаусс-3 ∘ P6SC", "Вопрос 3");
img(s, "fig_q3_gauss.png", M, 1.2, W - 2 * M, 4.6);
caption(s, "Полюс R(z) в −0.41 + 20.04i, |R| ≈ 29 рядом с ним, хотя |R(iy)| ≡ 1. Для опор с комплексными полюсами нужно |arg γ| + |arg w_p| < 90°.", M, 6.0, W - 2 * M, 0.8);

s = slide("KD и Грэгг: области расширяются", "Вопрос 3");
img(s, "fig_q3_kd_regions.png", M, 1.2, 6.0, 4.9);
img(s, "fig_q3_mid_regions.png", 6.8, 1.2, 6.0, 4.9);
caption(s, "Граница по вещественной оси. KD (k = 0): P2 −2, P4C −5.2, P6SC −7.9, P6C −15, P4R −0.56. Грэгг: P2 −3.1, P4C −8, P6C −22, P6R −1.4. Схемы с отрицательным шагом (P4R, P6R) область дробят.", M, 6.2, W - 2 * M, 0.75);

// ---------------- Q4
pres.addSection({ title: "Вопрос 4" });
s = slide("Вопрос 4. Найденные схемы", "Вопрос 4");
table(s, [["схема", "опора", "порядок", "стадий", "проекция Re", "проверка"],
          ["midpoint_p6_gen_Re", "Грэгг", "6", "5", "нужна", "5.99–6.01"],
          ["midpoint_p6_pal_Re", "Грэгг", "6", "7", "нужна", "5.99–6.01"],
          ["midpoint_p6_pal_C", "Грэгг", "6", "9", "не нужна", "5.99–6.00"],
          ["general_p4_gen_Re", "любая", "4", "3", "нужна", "3.99–4.00"],
          ["general_p4_pal_Re (Re γ₂ = 0)", "любая", "4", "3", "нужна", "3.99–4.01"],
          ["general_p6_gen_Re", "любая", "6", "7", "нужна", "5.99–6.00"],
          ["general_p6_pal_Re", "любая", "6", "9", "нужна", "5.99–6.00"]],
  M, 1.4, 7.6, [2.75, 0.9, 0.95, 0.85, 1.15, 1.0], 12);
bullets(s, ["Опора как exp(hF₁ + Σ hᵏFₖ): для Грэгга K = {3,5,6,7}, для общей K = {3,…,7}",
  "Многостартовый Левенберг–Марквардт по условиям порядка",
  "Ревью: точные условия через слова свободной алгебры, уточнение в 50 знаках",
  "Минимальность стадий не доказана, 1500 стартов меньше не нашли"], 8.45, 1.45, 4.35, 5.3, 14);

s = slide("Вопрос 4. Порядок 6 на Грэгге и Хойне", "Вопрос 4");
img(s, "fig_q4_local_hp.png", M, 1.2, W - 2 * M, 3.1);
img(s, "fig_q4_conv_mid_forced_duffing.png", M, 4.35, 6.0, 2.6);
img(s, "fig_q4_conv_heun_van_der_pol.png", 6.8, 4.35, 6.0, 2.6);

s = slide("Вопрос 4. Пример и стоимость", "Вопрос 4");
img(s, "fig_q4_solution.png", M, 1.2, W - 2 * M, 3.9);
bullets(s, ["Дуффинг на Грэгге, midpoint_p6_gen_Re, h = 0.4: ошибка 7e-7",
  "Хойн, ошибка 1e-8 на Ван дер Поле: 1 100 вызовов f против 9 700 у P6C",
  "На Грэгге при 1e-6…1e-10 P6C почти не уступает: малая константа"], M, 5.3, W - 2 * M, 1.6, 15);

s = slide("Нейросеть и исправления после ревью", "Вопрос 4");
card(s, M, 1.35, 5.95, 5.4, "Нейросеть не понадобилась", "Условия порядка полиномиальны, многостартовый поиск находит решения, а проверка точная (невязка 1e-50 после уточнения).\n\nРазумные роли для нейросети: генератор начальных приближений при больших m; суррогат для оптимизации семейства решений по области устойчивости или константе ошибки.", THEME.colors.accent1, 17);
card(s, M + 6.28, 1.35, 5.95, 5.4, "Что исправлено", "Условие A-устойчивости верно только при вещественных полюсах опоры.\n\nP6SC без проекции даёт 6, а 3 только на комплексных ОДУ.\n\nУ general_p4_pal_Re Re γ₂ = 0, а не > 0.\n\nПорядки подтверждены в высокой точности; медиана прятала артефакты.\n\nКлассу C нужна формулировка «колебательно-жёсткий режим».", THEME.colors.accent2, 17);

// ---------------- Повтор статьи о KD
pres.addSection({ title: "Статья о KD" });
s = slide("Статья о KD: области устойчивости повторены", "Статья о KD");
img(s, "kd/fig02_compare.png", M, 1.15, 7.9, 3.9, "left");
bullets(s, ["Galchenko et al., Mathematics 2026: рис. 1–9 повторены, совпадают по форме, границам и оттенкам",
  "CD, s = 0.5: при k = 0 область ≈ [−2, 0]×[−2, 2], при k = 1 полоса |Im| ≤ 2",
  "Совпадает с нашей функцией устойчивости KD (вопросы 2–3)"], 8.75, 1.3, 4.05, 3.9, 14);
img(s, "kd/fig01_compare.png", M, 5.25, 7.9, 1.6, "left");
caption(s, "Слева: статья | наш расчёт. Сверху рис. 2 (CD при разных k), снизу рис. 1 (явный Эйлер, контроль).", 8.75, 5.4, 4.05, 1.2);

s = slide("Статья о KD: области при разных s", "Статья о KD");
img(s, "kd/fig03_compare.png", M, 1.2, 6.05, 3.6);
img(s, "kd/fig05_compare.png", 6.75, 1.2, 6.05, 3.6);
bullets(s, ["k = 1 (слева): при s < 0.5 метод пересглаживает, есть области в правой полуплоскости; при s > 0.5 область сжимается",
  "k = 0 (справа): все области ограничены; наибольшая при s = 0 (до Re ≈ −5), при s = 0.5 около [−2, 0]×[−2, 2]"], M, 5.05, W - 2 * M, 1.8, 15);

s = slide("Статья о KD: движение корней", "Статья о KD");
img(s, "kd/fig06_compare.png", M, 1.15, 6.05, 2.0);
img(s, "kd/fig08_compare.png", 6.75, 1.15, 6.05, 2.0);
img(s, "kd/fig09_compare.png", M, 3.3, W - 2 * M, 2.4);
bullets(s, ["Собств. числа F = (R − I)/h при h ∈ [1e-3, 1e-1], цвет: s от 0 до 1; при s = 0.5 корни почти не сдвигаются",
  "Сверху k = 1 и k = 0 (рис. 6 и 8), снизу крупный масштаб (рис. 9)"], M, 5.85, W - 2 * M, 1.1, 14);

s = slide("Статья о KD: три расхождения кода с рисунками", "Статья о KD");
table(s, [["№", "код приложения A", "рисунки статьи", "что сделано"],
  ["1", "при ω = 0 рисуется только lams(2): одна ветвь", "дуги симметричны относительно оси", "рисуются оба собственных числа"],
  ["2", "тестовые точки рис. 9 не перечислены", "видны по красным кругам", "сняты с рисунка; точки на оси со сдвигом ±1e-3 по Im"],
  ["3", "фон: Re ∈ [−4.1, 0.1], Im ∈ [−1.75, 1.65]", "фон покрывает |Im| ≥ 2", "фон считается на окне рисунка"]],
  M, 1.45, W - 2 * M, [0.6, 4.0, 3.4, 4.23], 15);
caption(s, "Скрипт и сравнения: kd_article_repro/ (kd_repro.py, compare/). Трёхмерные рис. 10–14 не строились.", M, 4.6, W - 2 * M);

// ---------------- Итог
pres.addSection({ title: "Итог" });
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Итог" });
s.addText("Итог", { placeholder: "title" });
s.addText("Трапеции и KD дают 2/4/6, Грэгг 2/4/5. Порядок 6 на Грэгге возможен со специальными коэффициентами (5 стадий). A-устойчивость сохраняют P3C и P6SC на трапециях. Нежёсткие сбои связаны с неаналитичностью f и с комплексными задачами. Код и отчёт: github.com/Demonica22/methods (report/report.pdf, графики: report/src).", { placeholder: "body" });

(async () => {
  await pres.writeFile({ fileName: OUT });
  if (applyTheme) await applyTheme(OUT, THEME);
  console.log("written", OUT);
})();
