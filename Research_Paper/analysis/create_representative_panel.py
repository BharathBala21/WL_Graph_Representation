from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "illustrative_images"
OUT = ROOT / "figures" / "fig1_representative_leaf_images.png"
items = [
    ("01_NON-STRESSED_ash_gourd_healthy.JPG", "(a) Ash gourd - healthy"),
    ("07_NON-STRESSED_tomato_healthy.JPG", "(b) Tomato - healthy"),
    ("08_MODERATE-STRESSED_ash_gourd_N.JPG", "(c) Ash gourd - nitrogen deficiency"),
    ("09_MODERATE-STRESSED_bitter_gourd_K.JPG", "(d) Bitter gourd - potassium deficiency"),
    ("15_STRESSED_ash_gourd_PM.JPG", "(e) Ash gourd - powdery mildew"),
    ("20_STRESSED_tomato_LM.JPG", "(f) Tomato - leaf miner"),
]
W, H, label_h, margin, gap, image_side = 540, 345, 44, 30, 18, 345
canvas = Image.new("RGB", (3*W + 2*gap + 2*margin, 2*(H+label_h) + gap + 2*margin), "white")
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 22)
except OSError:
    font = ImageFont.load_default()
draw = ImageDraw.Draw(canvas)
for i, (name, label) in enumerate(items):
    im = Image.open(SRC/name).convert("RGB")
    im = ImageOps.fit(im, (image_side, H), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    x = margin + (i % 3) * (W + gap)
    y = margin + (i // 3) * (H + label_h + gap)
    ix = x + (W-image_side)//2
    canvas.paste(im, (ix, y))
    draw.rectangle((ix, y, ix+image_side-1, y+H-1), outline=(185,185,185), width=2)
    draw.text((x+7, y+H+7), label, fill=(20,20,20), font=font)
canvas.save(OUT, dpi=(300,300), optimize=True)
print(f"Wrote {OUT}")
