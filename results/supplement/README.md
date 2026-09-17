# Webappendix (supplementary material) for submission

`webappendix.pdf` is the single supplementary file to upload with the
manuscript to *The Lancet Child & Adolescent Health*; `webappendix.docx` is
the editable source of the same document (Word, with a live table of
contents).

Built by `supplement/make_webappendix.py` (python-docx) and exported by
`supplement/export_pdf.ps1` (Word, updates the table of contents and page
numbers, saves the .docx, writes the PDF):

```bash
python supplement/make_webappendix.py
powershell -ExecutionPolicy Bypass -File supplement/export_pdf.ps1
```

Content and sources

| Part | Source |
|---|---|
| Tables S1-S8 (13 tables) | `results/tables/table_s*.csv` (`tables/make_supplementary_tables.py`) |
| Figures S1-S8 | `results/figures/supplementary/figS*.png` |
| headings, legends, column labels, the Table S4 covariate-instruments note, footnotes, appendix references | `supplement/supplement_meta.json` (the manuscript wording) |
| title and author line | constants at the top of the builder |

Lancet supplementary-material rules applied: one document with a table of
contents and numbered pages; main heading 12 pt Times New Roman bold, text
10 pt single-spaced, section headings 10 pt bold; tables 8 pt with 8 pt bold
in-table headings, table headings 10 pt bold, legends 10 pt; figure headings
10 pt bold, legends 10 pt, every image placed at a width that keeps it at or
above 300 dpi (Figure S1 at 159 mm, the tall Figure S2 at 98 mm, the rest at
the 170 mm text width); midline decimals; p values to two significant figures
without a leading zero (`<0·0001` below that; never more decimals than the
stored value); Vancouver-style appendix references numbered separately from
the main text. Table S3c is transposed (statistics as rows) so that it fits
the page.

Word substitutes `—` for blank cells (no consensus, no expert domain, not
applicable); the `Sig.` column is blank when not significant.
