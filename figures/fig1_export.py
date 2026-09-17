#!/usr/bin/env python
"""
Figure 1 (pipeline schematic) — export the hand-drawn SVG for submission.

The master is results/figures/main/fig1_pipeline.svg (edited by hand; no
plotting script).  This script renders it with the Chrome that ships with
Playwright, which keeps text as text (selectable, editable) in the PDF:

  results/figures/main/fig1_pipeline.pdf              vector, 183 mm wide
  results/figures/main/fig1_pipeline.png              300 dpi raster, 183 mm wide
  results/figures/submission/figure1_combined.{svg,pdf,png}
  results/figures/submission/figure1_{A,B,C}.{svg,pdf,png}

The three panels are separate top-level <g> groups in the SVG (panel B and C
carry translate(0, 270) / translate(0, 540)), so each panel file is the same
SVG restricted to one group with the viewBox cropped to that band.  Nothing in
the master file is altered.

Usage (from the repository root; needs `pip install playwright` and Chrome):
    python figures/fig1_export.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "results/figures/main/fig1_pipeline.svg"
MAIN_DIR = SVG.parent
SUBMISSION_DIR = ROOT / "results/figures/submission"

WIDTH_MM = 183.0            # full-width figure
DPI = 300                   # journal minimum for raster copies
CSS_PX_PER_MM = 96 / 25.4

# panel bands of the 1250 x 820 viewBox (the two separator lines sit at y = 270 and 540)
PANELS = {"A": (0, 270), "B": (270, 540), "C": (540, 820)}


def parse_master(text: str):
    """Split the SVG into header (up to and including the white background rect),
    the three panel groups, and the closing tag."""
    m_bg = re.search(r'<rect width="1250" height="820" fill="#ffffff"/>', text)
    header = text[:m_bg.end()]
    # top-level groups: panel A has no transform, B and C are translated
    starts = [m.start() for m in re.finditer(r'\n  <g (?:transform="translate\(0, \d+\)" )?fill="#000000">', text)]
    assert len(starts) == 3, f"expected 3 panel groups, found {len(starts)}"
    ends = [text.index("\n  </g>", s) + len("\n  </g>") for s in starts]
    groups = {letter: text[s:e] for letter, s, e in zip("ABC", starts, ends)}
    return header, groups


def panel_svg(header: str, group: str, y0: int, y1: int) -> str:
    """The master's header with the viewBox cropped to one band + that panel's group."""
    h = y1 - y0
    hdr = header.replace('viewBox="0 0 1250 820"', f'viewBox="0 {y0} 1250 {h}"', 1)
    hdr = hdr.replace('<rect width="1250" height="820" fill="#ffffff"/>',
                      f'<rect y="{y0}" width="1250" height="{h}" fill="#ffffff"/>', 1)
    return hdr + group + "\n</svg>\n"


def html_for(svg_text: str, w_mm: float, h_mm: float) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>"
            f"@page{{size:{w_mm}mm {h_mm}mm;margin:0}} html,body{{margin:0;padding:0;background:#fff}}"
            f"svg{{display:block;width:{w_mm}mm;height:{h_mm}mm}}</style></head><body>"
            f"{svg_text}</body></html>")


def render(page_factory, svg_text: str, pdf_path: Path, png_path: Path, w_mm: float) -> None:
    vb = re.search(r'viewBox="(-?\d+) (-?\d+) (\d+) (\d+)"', svg_text)
    vw, vh = float(vb.group(3)), float(vb.group(4))
    h_mm = w_mm * vh / vw
    html = html_for(svg_text, w_mm, h_mm)
    # vector PDF (Chrome keeps SVG text as text)
    page = page_factory(1.0, w_mm, h_mm)
    page.set_content(html)
    page.pdf(path=str(pdf_path), width=f"{w_mm}mm", height=f"{h_mm:.3f}mm", print_background=True,
             margin={"top": "0", "right": "0", "bottom": "0", "left": "0"}, page_ranges="1")
    page.context.close()
    # raster PNG at DPI
    page = page_factory(DPI / 96, w_mm, h_mm)
    page.set_content(html)
    page.screenshot(path=str(png_path), full_page=False, omit_background=False)
    page.context.close()
    from PIL import Image
    im = Image.open(png_path)
    im.save(png_path, dpi=(DPI, DPI))
    print(f"  saved: {pdf_path}  ({w_mm:.0f} x {h_mm:.1f} mm)")
    print(f"  saved: {png_path}  ({im.size[0]} x {im.size[1]} px at {DPI} dpi)")


def main() -> int:
    from playwright.sync_api import sync_playwright
    text = SVG.read_text(encoding="utf-8")
    header, groups = parse_master(text)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome")

        def page_factory(scale: float, w_mm: float, h_mm: float):
            ctx = browser.new_context(viewport={"width": round(w_mm * CSS_PX_PER_MM),
                                                "height": round(h_mm * CSS_PX_PER_MM)},
                                      device_scale_factor=scale)
            return ctx.new_page()

        # combined figure: alongside the master + the submission copy
        render(page_factory, text, MAIN_DIR / "fig1_pipeline.pdf", MAIN_DIR / "fig1_pipeline.png", WIDTH_MM)
        for fmt in ("svg", "pdf", "png"):
            src = MAIN_DIR / f"fig1_pipeline.{fmt}"
            dst = SUBMISSION_DIR / f"figure1_combined.{fmt}"
            dst.write_bytes(src.read_bytes())
            print(f"  saved: {dst}")
        # individual panels
        for letter, (y0, y1) in PANELS.items():
            svg_text = panel_svg(header, groups[letter], y0, y1)
            svg_path = SUBMISSION_DIR / f"figure1_{letter}.svg"
            svg_path.write_text(svg_text, encoding="utf-8")
            print(f"  saved: {svg_path}")
            render(page_factory, svg_text, SUBMISSION_DIR / f"figure1_{letter}.pdf",
                   SUBMISSION_DIR / f"figure1_{letter}.png", WIDTH_MM)
        browser.close()
    print("Figure 1 export done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
