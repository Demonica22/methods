// HTML-отчёт -> PDF через Chromium (playwright).
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + path.resolve(__dirname, 'report.html'), { waitUntil: 'networkidle' });
  await page.pdf({
    path: path.resolve(__dirname, '..', 'report.pdf'), format: 'A4', printBackground: true,
    displayHeaderFooter: true, headerTemplate: '<span></span>',
    footerTemplate: '<div style="width:100%;font-size:8px;color:#898781;text-align:center;font-family:DejaVu Sans">' +
      'Композиционные схемы с комплексными коэффициентами · <span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    margin: { top: '16mm', bottom: '16mm', left: '16mm', right: '16mm' },
  });
  await browser.close();
})();
