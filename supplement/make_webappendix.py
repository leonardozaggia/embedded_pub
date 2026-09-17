#!/usr/bin/env python
"""
Build the Webappendix (supplementary material) for The Lancet Child &
Adolescent Health as a single Word document, ready for PDF export.

Lancet "Guidelines for supplementary material" applied here:
  * one document, table of contents, numbered pages
  * main heading 12 pt Times New Roman bold; text 10 pt single-spaced;
    section headings 10 pt bold
  * tables 8 pt Times New Roman, table headings 10 pt bold, in-table
    headings 8 pt bold, legends 10 pt
  * figure headings 10 pt bold, legends 10 pt; images >= 300 dpi
  * midline decimals (23·4), no leading zero on p, p to two significant
    figures unless p < 0·0001
  * references in Vancouver style, numbered separately from the main text

Inputs
  results/tables/table_s*.csv               the 13 supplementary tables
  results/figures/supplementary/figS*.png   the 8 supplementary figures
  supplement/supplement_meta.json           captions, column labels, notes,
      footnotes and appendix references (the manuscript wording), as the
      keys TABLE_SECTIONS, FIGURE_SECTIONS, ABBREVIATIONS and REFERENCES

Output
  results/supplement/webappendix.docx       then run supplement/export_pdf.ps1
  (Word: updates the table of contents and page numbers, saves the .docx and
  exports results/supplement/webappendix.pdf)

    python supplement/make_webappendix.py
    powershell -ExecutionPolicy Bypass -File supplement/export_pdf.ps1
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Mm, Pt, RGBColor
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures" / "supplementary"
META = ROOT / "supplement" / "supplement_meta.json"
OUT_DIR = ROOT / "results" / "supplement"
OUT_DOCX = OUT_DIR / "webappendix.docx"

TITLE = ("From Composite to Components: Evaluating the Impact of Domain Differentiation "
         "on Bayley-III Cognitive Diagnostic Precision")
AUTHORS = "Leonardo Zaggia, Adele Diederich, Anne Hilgendorff, Axel Heep, Andrea Hildebrandt"
FONT = "Times New Roman"
TEXT_WIDTH_MM = 170.0          # A4, 2 cm margins
MAX_FIG_HEIGHT_MM = 225.0
MIN_DPI = 300

# Column names used in supplement_meta.json -> column names of the CSVs in
# results/tables/ (per table), so the column-label map applies to the CSVs.
PUB_ALIAS = {
    "s2": {"theoretical_domain_3F": "translated_domain"},
    "s5": {"subfolder": "outcome_code", "label": "outcome_label"},
}
EXTRA_LABELS = {          # columns with no label in supplement_meta.json
    "s2": {"translated_domain": "Domain (translated)", "empirical_factor": "Empirical factor"},
    "s4b": {"sig": "Sig."}, "s4a": {"sig": "Sig."}, "s5": {"sig": "Sig."},
}
EXTRA_HIDE = {"s5": ["outcome_code", "outcome"]}
DOMAIN_ABBREV = {"attention": "ATT", "working_memory": "WM", "goal_directed_problem_solving": "GDPS",
                 "flexibility_shift": "FS", "higher_order_processing": "HOP"}
VALUE_MAPS = {"s1": {"theoretical_domain": DOMAIN_ABBREV}}   # snake_case labels -> abbreviations
# Short figure titles for the headings / table of contents (the full caption is the legend).
FIG_TITLES = {
    "fig-s1": "Centroid similarity profiles, unpaired items in the dHCP administration range",
    "fig-s2": "Centroid similarity profiles, all 52 unpaired items",
    "fig-s3": "Three-way Bayley-4 alluvial diagram",
    "fig-s4": "Bayley-III alluvial diagram",
    "fig-s5": "Embedding cosine similarity vs tetrachoric correlation, 23-item CFA model",
    "fig-s6": "Three-way correspondence of the Bayley-4 domain structures",
    "fig-s7": "Within- vs between-cluster cosine similarity by factor",
    "fig-s8": "Sensitivity of the cascade pairing to the title-similarity threshold",
}
P_COLS = {"p_value", "chi_p", "difftest_p"}
DASH = ".."                    # Lancet blank-cell marker (not applicable / no value)
MINUS = "−"
MIDDOT = "·"


# ---------------------------------------------------------------------------
# number formatting (Lancet house style)
# ---------------------------------------------------------------------------
def midline(s: str) -> str:
    """23.4 -> 23·4, -0.12 -> −0·12; leaves non-numeric text alone except decimals inside numbers."""
    s = re.sub(r"(?<=\d)\.(?=\d)", MIDDOT, s)
    s = re.sub(r"(^|(?<=[\s(=]))-(?=\d)", MINUS, s)
    return s


def fmt_p(v) -> str:
    """Two significant figures (never more decimals than the stored value carries),
    with the leading zero (0·026, as in Lancet articles), '<0·0001' below that."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
        return DASH
    x = float(v)
    if x < 0.0001:
        return f"<0{MIDDOT}0001"
    if x >= 1:
        return f"1{MIDDOT}0"
    decimals = -math.floor(math.log10(x)) + 1
    src = str(v).strip()
    if re.fullmatch(r"0\.\d+", src):                 # 0.001 stays ·001, not ·0010
        decimals = min(decimals, len(src) - 2)
    s = f"{x:.{decimals}f}"
    return "0" + MIDDOT + s[2:]      # 0.026 -> 0·026


def house(text: str) -> str:
    """Wording fixes applied to captions, notes and footnotes: em dashes out
    (spaced ones become a colon in sub-headings, otherwise an en dash), the
    blank marker explained as '..', British spelling, 'vs' without a stop."""
    text = text.replace(" \u2014 marks a blank", " .. marks a blank")
    text = text.replace("\u2014 marks a blank", ".. marks a blank")
    text = text.replace(" \u2014 ", " \u2013 ").replace("\u2014", "\u2013")
    text = re.sub(r"\bvs\.", "vs", text)
    text = text.replace("standardization", "standardisation").replace("Standardization", "Standardisation")
    return text


def subheading(text: str) -> str:
    """'S3a — 3-factor model' -> 'S3a: 3-factor model'."""
    return house(re.sub(r"^(\S+)\s+\u2014\s+", r"\1: ", text))


def fmt_cell(col: str, v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "" if col == "sig" else DASH
    s = str(v).strip()
    if s == "" or s.lower() == "nan" or s in ("\u2014", "-", ".."):
        return "" if col == "sig" else DASH
    if col in P_COLS:
        return fmt_p(v)
    if re.fullmatch(r"-?\d+\.0", s) and col in ("df", "difftest_df", "n_valid", "n_items"):
        s = s[:-2]
    return midline(s)


# ---------------------------------------------------------------------------
# docx helpers
# ---------------------------------------------------------------------------
def set_font(run, size: float, bold: bool | None = None, italic: bool | None = None):
    run.font.name = FONT
    run.font.size = Pt(size)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts"); rpr.append(rfonts)
    for k in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(k), FONT)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic


def style_font(style, size: float, bold: bool = False):
    style.font.name = FONT
    style.font.size = Pt(size)
    rpr0 = style.element.get_or_add_rPr()
    szcs = rpr0.find(qn("w:szCs"))
    if szcs is None:
        szcs = OxmlElement("w:szCs"); rpr0.append(szcs)
    szcs.set(qn("w:val"), str(int(size * 2)))
    style.font.bold = bold
    style.font.italic = False
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts"); rpr.append(rfonts)
    for k in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(k), FONT)
    for k in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rfonts.get(qn(k)) is not None:
            del rfonts.attrib[qn(k)]


def para(doc, text: str = "", size: float = 10, bold: bool = False, style: str | None = None,
         space_after: float = 4, space_before: float = 0, align=None, keep_next: bool = False):
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after); pf.space_before = Pt(space_before)
    pf.line_spacing = 1.0
    if align is not None:
        pf.alignment = align
    if keep_next:
        pf.keep_with_next = True
    if text:
        r = p.add_run(text); set_font(r, size, bold)
    return p


def add_field(paragraph, instr: str, placeholder: str = "", size: float = 10):
    """Complex field (TOC, PAGE) that Word fills in on update."""
    def fld(kind):
        r = paragraph.add_run(); set_font(r, size)
        fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), kind); r._element.append(fc)
    fld("begin")
    r = paragraph.add_run(); set_font(r, size)
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = f" {instr} "
    r._element.append(it)
    fld("separate")
    r = paragraph.add_run(placeholder); set_font(r, size)
    fld("end")


def table_borders(table, size: int = 4):
    """Lancet-like rules: top/bottom of the table and under the header, no vertical rules."""
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, val in (("top", "single"), ("bottom", "single"), ("insideH", "single"),
                      ("left", "nil"), ("right", "nil"), ("insideV", "nil")):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        if val == "single":
            el.set(qn("w:sz"), str(size)); el.set(qn("w:space"), "0"); el.set(qn("w:color"), "808080")
        borders.append(el)
    tblPr.append(borders)


def cell_text(cell, text: str, size: float = 8, bold: bool = False, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0); p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    if align is not None:
        p.paragraph_format.alignment = align
    r = p.add_run(text); set_font(r, size, bold)
    # tight cell margins
    tcPr = cell._tc.get_or_add_tcPr()
    mar = OxmlElement("w:tcMar")
    for side in ("top", "bottom"):
        el = OxmlElement(f"w:{side}"); el.set(qn("w:w"), "15"); el.set(qn("w:type"), "dxa"); mar.append(el)
    for side in ("left", "right"):
        el = OxmlElement(f"w:{side}"); el.set(qn("w:w"), "50"); el.set(qn("w:type"), "dxa"); mar.append(el)
    tcPr.append(mar)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader"); el.set(qn("w:val"), "true"); trPr.append(el)


def no_split(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:cantSplit"); el.set(qn("w:val"), "true"); trPr.append(el)


def is_numeric_col(values: list[str]) -> bool:
    return all(re.fullmatch(rf"[<>]?\s*[{MINUS}-]?\d+({MIDDOT}\d+)?%?|{re.escape(DASH)}|<0{MIDDOT}0001", v) for v in values)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def load_table(key: str) -> tuple[pd.DataFrame, str]:
    files = sorted(TABLES.glob(f"table_{key}_*.csv"))
    if len(files) != 1:
        raise SystemExit(f"expected one results/tables/table_{key}_*.csv, found {files}")
    return pd.read_csv(files[0], dtype=str, keep_default_na=False), files[0].name


def resolve_part(part: dict) -> dict:
    """Apply the supplement_meta -> results/tables column aliases to a part definition."""
    key = part["key"]
    alias = PUB_ALIAS.get(key, {})
    cols = {alias.get(k, k): v for k, v in part.get("cols", {}).items()}
    cols.update(EXTRA_LABELS.get(key, {}))
    hide = [alias.get(h, h) for h in part.get("hide", [])] + EXTRA_HIDE.get(key, [])
    return {"key": key, "sub": part.get("sub"), "cols": cols, "hide": hide,
            "groupBy": alias.get(part.get("groupBy"), part.get("groupBy")),
            "groupExtra": alias.get(part.get("groupExtra"), part.get("groupExtra"))}


def build_matrix(part: dict) -> tuple[list[str], list[list[str]], list[bool], list[int]]:
    """Header labels, body rows (group rows = single-element lists), numeric flags, group-row indices."""
    df, _ = load_table(part["key"])
    skip = set(part["hide"]) | {c for c in (part["groupBy"], part["groupExtra"]) if c}
    show = [c for c in df.columns if c not in skip]
    if part["key"] == "s3c":                                   # transpose: statistics as rows, models as columns
        labels = part["cols"]
        stats = [c for c in show if c != "model"]
        header = ["Statistic"] + [midline(m) for m in df["model"]]
        body = [[labels.get(c, c)] + [fmt_cell(c, v) for v in df[c]] for c in stats]
        numeric = [False] + [True] * (len(header) - 1)
        return header, body, numeric, []
    header = [part["cols"].get(c, c.replace("_", " ").capitalize()) for c in show]
    body, groups, last = [], [], None
    for _, r in df.iterrows():
        if part["groupBy"] and r[part["groupBy"]] != last:
            last = r[part["groupBy"]]
            extra = f" · {r[part['groupExtra']]}" if part["groupExtra"] and r[part["groupExtra"]] else ""
            groups.append(len(body)); body.append([midline(str(last)) + extra])
        vmap = VALUE_MAPS.get(part["key"], {})
        body.append([fmt_cell(c, vmap.get(c, {}).get(r[c], r[c])) for c in show])
    numeric = [is_numeric_col([row[i] for row in body if len(row) > 1]) for i in range(len(show))]
    return header, body, numeric, groups


def add_table(doc, part: dict) -> int:
    header, body, numeric, groups = build_matrix(part)
    ncol = len(header)
    t = doc.add_table(rows=1, cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    table_borders(t)
    hdr = t.rows[0]; repeat_header(hdr); no_split(hdr)
    for i, h in enumerate(header):
        cell_text(hdr.cells[i], h, 8, True, WD_ALIGN_PARAGRAPH.RIGHT if numeric[i] else None)
    for ri, row in enumerate(body):
        r = t.add_row(); no_split(r)
        if len(row) == 1:                                     # group heading row
            merged = r.cells[0].merge(r.cells[ncol - 1])
            cell_text(merged, row[0], 8, True)
            continue
        for i, v in enumerate(row):
            cell_text(r.cells[i], v, 8, False, WD_ALIGN_PARAGRAPH.RIGHT if numeric[i] else None)
    para(doc, "", 4, space_after=2)
    return len(body) - len(groups)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figure_file(base: str) -> Path:
    n = re.match(r"figure_s(\d+)_", base).group(1)
    files = sorted(FIGURES.glob(f"figS{n}_*.png"))
    if len(files) != 1:
        raise SystemExit(f"expected one results/figures/supplementary/figS{n}_*.png, found {files}")
    return files[0]


def figure_width_mm(path: Path) -> tuple[float, float]:
    w_px, h_px = Image.open(path).size
    w = min(TEXT_WIDTH_MM, w_px / MIN_DPI * 25.4)            # never below 300 dpi
    if h_px / w_px * w > MAX_FIG_HEIGHT_MM:                  # tall figures: fit the page
        w = MAX_FIG_HEIGHT_MM * w_px / h_px
    return w, w_px / (w / 25.4)


def first_sentence(text: str) -> str:
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    return (m.group(1) if m else text).rstrip(".")


# ---------------------------------------------------------------------------
def main() -> int:
    meta = json.loads(META.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = Document()

    # page setup: A4, 2 cm margins, page numbers in the footer
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.PORTRAIT
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(2))
    foot = sec.footer.paragraphs[0]
    foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(foot, "PAGE", "1", 10)

    # styles: everything Times New Roman; headings 10 pt bold (Heading 1/2 feed the TOC)
    style_font(doc.styles["Normal"], 10)
    # document defaults (docDefaults) also say 10 pt so nothing can fall back to 11 pt
    rpr_default = doc.styles.element.find(qn("w:docDefaults")).find(qn("w:rPrDefault")).find(qn("w:rPr"))
    for tag in ("w:sz", "w:szCs"):
        el = rpr_default.find(qn(tag))
        if el is None:
            el = OxmlElement(tag); rpr_default.append(el)
        el.set(qn("w:val"), "20")
    rf = rpr_default.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr_default.append(rf)
    for k in list(rf.attrib):
        del rf.attrib[k]
    for k in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(k), FONT)
    doc.styles["Normal"].paragraph_format.space_after = Pt(4)
    doc.styles["Normal"].paragraph_format.line_spacing = 1.0
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        st = doc.styles[name]; style_font(st, 10, True)
        st.paragraph_format.space_before = Pt(10 if name == "Heading 1" else 6)
        st.paragraph_format.space_after = Pt(3)
        st.paragraph_format.keep_with_next = True
    # Word rebuilds the table of contents with the TOC styles and the footer /
    # captions inherit from Normal: every one of them is pinned to 10 pt here.
    for name in ("TOC 1", "TOC 2", "TOC 3", "TOC Heading", "Caption", "Footer", "Header",
                 "Table Grid", "List Paragraph"):
        if name in [st.name for st in doc.styles]:
            style_font(doc.styles[name], 10, False)
            doc.styles[name].paragraph_format.space_after = Pt(2)
            doc.styles[name].paragraph_format.line_spacing = 1.0
    for name in ("Title", "Subtitle"):
        if name in [st.name for st in doc.styles]:
            style_font(doc.styles[name], 12, True)

    # ---- cover ----------------------------------------------------------
    para(doc, "Supplementary appendix", 12, True, space_after=8)
    para(doc, TITLE, 10, True, space_after=2)
    para(doc, AUTHORS, 10, space_after=2)
    para(doc, "Supplementary appendix to the manuscript submitted to The Lancet Child & Adolescent Health. "
              "Tables S1–S8 and Figures S1–S8; appendix references are numbered separately "
              "from those of the main text. In the tables, .. marks a value that does not apply "
              "or was not given.", 10, space_after=10)
    para(doc, "Contents", 10, True, space_after=4)
    toc = para(doc, "", 10)
    add_field(toc, 'TOC \\o "1-2" \\h \\z \\u', "[Table of contents — update fields (F9) or run export_pdf.ps1]", 10)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---- abbreviations --------------------------------------------------
    para(doc, "Abbreviations", 10, True, style="Heading 1")
    p = para(doc, "", 10, space_after=8)
    for i, (ab, full) in enumerate(meta["ABBREVIATIONS"]):
        if i:
            r = p.add_run(" · "); set_font(r, 10)
        r = p.add_run(ab); set_font(r, 10, True)
        r = p.add_run(f" {full}"); set_font(r, 10)
    r = p.add_run(". ARI adjusted Rand index; NMI normalised mutual information; CFA confirmatory factor "
                  "analysis; MIMIC multiple indicators multiple causes; WLSMV weighted least squares, "
                  "mean- and variance-adjusted; STDYX/STDY Mplus standardisations (fully / semi-standardised); "
                  "UC unclassified cluster; dHCP developing Human Connectome Project.")
    set_font(r, 10)

    # ---- tables ---------------------------------------------------------
    para(doc, "Supplementary tables", 10, True, style="Heading 1")
    n_rows_total = 0
    for s in meta["TABLE_SECTIONS"]:
        para(doc, f"{s['label']}. {midline(s['title'])}", 10, True, style="Heading 1")
        para(doc, house(midline(s["caption"])), 10, space_after=4)
        if s.get("note"):
            p = para(doc, "", 10, space_after=6)
            if s.get("noteLabel"):
                r = p.add_run(s["noteLabel"] + " "); set_font(r, 10, True)
            r = p.add_run(house(midline(s["note"]))); set_font(r, 10)
        for part in s["parts"]:
            part = resolve_part(part)
            if part["sub"]:
                para(doc, subheading(midline(part["sub"])), 10, True, style="Heading 2")
            n_rows_total += add_table(doc, part)
        if s.get("footnote"):
            para(doc, house(midline(s["footnote"])), 10, space_after=10)

    # ---- figures --------------------------------------------------------
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    para(doc, "Supplementary figures", 10, True, style="Heading 1")
    dpi_report = []
    for s in meta["FIGURE_SECTIONS"]:
        for f in s["figures"]:
            path = figure_file(f["base"])
            w_mm, dpi = figure_width_mm(path)
            dpi_report.append((f["label"], path.name, round(w_mm), round(dpi)))
            para(doc, f"{f['label']}. {FIG_TITLES.get(f['id'], first_sentence(f['caption']))}", 10, True, style="Heading 1")
            p = doc.add_paragraph(); p.paragraph_format.keep_with_next = True
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str(path), width=Mm(w_mm))
            para(doc, house(midline(f["caption"])), 10, space_after=14)

    # ---- appendix references -------------------------------------------
    para(doc, "Appendix references", 10, True, style="Heading 1")
    para(doc, "Numbered separately from the references of the main text; cited by superscript in the "
              "note to Table S4.", 10, space_after=6)
    for i, ref in enumerate(meta["REFERENCES"], 1):
        p = para(doc, "", 10, space_after=3)
        p.paragraph_format.left_indent = Cm(0.8); p.paragraph_format.first_line_indent = Cm(-0.8)
        r = p.add_run(f"{i}\t{ref}"); set_font(r, 10)

    doc.core_properties.title = "Supplementary appendix: " + TITLE
    doc.core_properties.author = AUTHORS
    doc.save(OUT_DOCX)
    print(f"written {OUT_DOCX.relative_to(ROOT)}: {len(meta['TABLE_SECTIONS'])} table sections "
          f"({n_rows_total} data rows), {sum(len(s['figures']) for s in meta['FIGURE_SECTIONS'])} figures, "
          f"{len(meta['REFERENCES'])} appendix references")
    for lab, name, w, dpi in dpi_report:
        print(f"  {lab}: {name} at {w} mm -> {dpi} dpi")
    print("next: powershell -ExecutionPolicy Bypass -File supplement/export_pdf.ps1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
