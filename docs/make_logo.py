"""Generate the plantbench logo in docs/_static: a stable-focus spiral and a JetBrains Mono wordmark.

    pip install fonttools cairosvg     # cairosvg needs the cairo library (brew install cairo)
    python docs/make_logo.py

The two weights of JetBrains Mono (SIL Open Font License) are fetched from Google Fonts
at run time, and the wordmark is written as outlines so the SVGs need no font.
"""

import math
import re
import tempfile
import urllib.request
from pathlib import Path

import cairosvg
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

OUT = Path(__file__).parent / "_static"
FONTS_CSS = "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@{}"

ORANGE = "#e77500"
INK = {"light": "#1d1f21", "dark": "#e6e4df"}

# Spiral: r = exp(-b theta) over 1.75 turns, radius ratio 5, fitted into a 50-unit box
# centered on (32, 32) of a 64-unit square, the outer arm starting at the lower left.
TURNS, RATIO, ROT, BOX, N = 1.75, 5.0, 1.2 + math.pi, 50.0, 400
WIDTH, DOT = 5.0, 6.0         # stroke width; fixed-point radius

SIZE = 45.0       # cap height 0.73 em, centered on the mark
BASELINE = 48.5
X0 = 72.0
TRACK = -0.02     # em


def spiral():
    T = 2 * math.pi * TURNS
    b = math.log(RATIO) / T
    pts = []
    for i in range(N + 1):
        th = T * i / N
        r, a = math.exp(-b * th), th + ROT
        pts.append((r * math.cos(a), -r * math.sin(a)))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    s = BOX / max(max(xs) - min(xs), max(ys) - min(ys))
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return [(32 + s * (x - cx), 32 + s * (y - cy)) for x, y in pts]


def mark(ink):
    pts = spiral()
    ex, ey = pts[-1]
    d = "M" + "L".join(f"{x:.2f} {y:.2f}" for x, y in pts)
    return (f'<path fill="none" stroke="{ink}" stroke-width="{WIDTH:g}" '
            f'stroke-linecap="round" stroke-linejoin="round" d="{d}"/>\n'
            f'<circle fill="{ORANGE}" cx="{ex:.2f}" cy="{ey:.2f}" r="{DOT:g}"/>')


def fetch_font(weight, into):
    """Download the TTF of one JetBrains Mono weight and return its path."""
    # Without a browser user agent the CSS API serves TrueType URLs.
    with urllib.request.urlopen(FONTS_CSS.format(weight)) as r:
        url = re.search(r"url\((https://[^)]+\.ttf)\)", r.read().decode()).group(1)
    path = into / f"jetbrainsmono-{weight}.ttf"
    urllib.request.urlretrieve(url, path)
    return path


def word_paths(fonts):
    """Return the SVG path data for 'plant' (400) + 'bench' (800) and the end x."""
    parts, x = [], X0
    for text, font in (("plant", fonts[400]), ("bench", fonts[800])):
        f = TTFont(font)
        gs, cmap, hmtx = f.getGlyphSet(), f.getBestCmap(), f["hmtx"]
        scale = SIZE / f["head"].unitsPerEm
        for ch in text:
            name = cmap[ord(ch)]
            pen = SVGPathPen(gs)
            gs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, x, BASELINE)))
            parts.append(pen.getCommands())
            x += hmtx[name][0] * scale + TRACK * SIZE
    return " ".join(parts), x - TRACK * SIZE


def svg(width, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:g} 64" '
            f'width="{width:g}" height="64">\n{body}\n</svg>\n')


def main():
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        fonts = {w: fetch_font(w, Path(tmp)) for w in (400, 800)}
        d, end = word_paths(fonts)
    width = round(end + 4)
    for theme, suffix in (("light", ""), ("dark", "-dark")):
        m = mark(INK[theme])
        (OUT / f"logo-mark{suffix}.svg").write_text(svg(64, m))
        word = f'<path fill="{INK[theme]}" d="{d}"/>'
        (OUT / f"logo{suffix}.svg").write_text(svg(width, m + "\n" + word))
    cairosvg.svg2pdf(url=str(OUT / "logo.svg"), write_to=str(OUT / "logo.pdf"))


if __name__ == "__main__":
    main()
