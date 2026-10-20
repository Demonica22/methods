import sys, os
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# страницы статьи, отрендеренные в PNG при 130 dpi (pdftoppm -r 130 -png article.pdf pdf/r)
P = os.environ.get("KD_PAGES", "pdf/r-%02d.png")
D = os.path.dirname(os.path.abspath(__file__))
crops = {  # page, box at 130 dpi
 "fig01": (7, (292, 625, 910, 815), ["fig01_euler_k.png"]),
 "fig02": (8, (292, 146, 894, 680), ["fig02_cd_s05_k.png"]),
 "fig03": (8, (292, 780, 894, 1310), ["fig03_cd_k1_s.png"]),
 "fig04": (9, (292, 146, 894, 680), ["fig04_cd_k0.5_s.png"]),
 "fig05": (9, (292, 780, 894, 1310), ["fig05_cd_k0_s.png"]),
 "fig06": (10, (295, 955, 1000, 1350), ["fig06_roots_k1.png"]),
 "fig07": (11, (295, 365, 1000, 760), ["fig07_roots_k05.png"]),
 "fig08": (11, (295, 915, 1000, 1310), ["fig08_roots_k0.png"]),
 "fig09": (12, (295, 395, 1000, 1200), ["fig09a_roots_k1_zoom.png", "fig09b_roots_k0_zoom.png"]),
}
os.makedirs(f"{D}/article_crops", exist_ok=True); os.makedirs(f"{D}/compare", exist_ok=True)
for name, (pg, box, ours) in crops.items():
    im = Image.open(P % pg).crop(box)
    im.save(f"{D}/article_crops/{name}_article.png")
    imgs = [im] + [Image.open(f"{D}/figs/{o}") for o in ours]
    n = len(imgs)
    fig, axs = plt.subplots(1, n, figsize=(7 * n, 6.5), gridspec_kw={"width_ratios": [i.width / i.height for i in imgs]})
    for ax, img, t in zip(axs, imgs, ["Статья"] + ["Воспроизведение"] * (n - 1)):
        ax.imshow(img); ax.axis("off"); ax.set_title(t, fontsize=14)
    fig.tight_layout(); fig.savefig(f"{D}/compare/{name}_compare.png", dpi=110); plt.close(fig)
print("ok")
