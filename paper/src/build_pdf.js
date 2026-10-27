// Статья (HTML + KaTeX) -> PDF через Chromium (playwright). Запуск: npm install && node build_pdf.js
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto('file://' + path.resolve(__dirname, 'paper.html'), { waitUntil: 'networkidle' });
  if (errors.length) { console.error(errors.join('\n')); process.exit(1); }
  await page.pdf({
    path: path.resolve(__dirname, '..', 'paper.pdf'), format: 'A4', printBackground: true,
    displayHeaderFooter: true, headerTemplate: '<span></span>',
    footerTemplate: '<div style="width:100%;font-size:8px;color:#666;text-align:center;font-family:Liberation Serif">' +
      '<span class="pageNumber"></span> из <span class="totalPages"></span></div>',
    margin: { top: '18mm', bottom: '18mm', left: '18mm', right: '18mm' },
  });
  await browser.close();
})();
