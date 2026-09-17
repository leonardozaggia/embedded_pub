import re, json, pandas as pd
from pathlib import Path

def clean_scoring_block(block: str):
    """
    Extract scoring entries like '2 points:', '1 point:', '0 points:' and 'Scoring:' lists.
    Returns a list of dicts with 'points' and 'criteria' fields (points may be None).
    """
    scoring_entries = []

    # Main pattern for numeric scoring (e.g., '2 points:')
    for m in re.finditer(
        r"(?im)^\s*(?:(?P<pts>\d+)\s*points?)\s*:\s*(?P<crit>.+?)(?=\n\s*\d+\s*points?\s*:|\Z)",
        block,
        flags=re.S,
    ):
        pts = int(m.group("pts"))
        crit = re.sub(r"\s*\n\s*", " ", m.group("crit")).strip()
        scoring_entries.append({"points": pts, "criteria": crit})

    # Fallback if no numeric lines found, e.g. 'Scoring:' bullets
    if not scoring_entries:
        m = re.search(r"(?is)^\s*Scoring\s*:\s*(.+)$", block, flags=re.M)
        if m:
            text = m.group(1).strip()
            parts = [p.strip(" -•\t") for p in re.split(r"\n+", text) if p.strip()]
            for p in parts:
                scoring_entries.append({"points": None, "criteria": p})

    return scoring_entries


def sanitize_text(text: str) -> str:
    """
    Automatically clean mojibake, encoding artifacts, and restore formatting.
    Ensures that 'Item 34 – Title' structure exists even if dashes were lost.
    """

    # --- 1. Attempt to repair double-encoded mojibake (ÔÇÖ, â€™, ├óÔé¼Ôäó) ---
    if any(ch in text for ch in ("Ô", "â", "├")):
        try:
            text = text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            pass  # safe fallback if the text is already fine

    # --- 2. Normalize common curly punctuation and dashes ---
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",   # en dash
        "—": "-",   # em dash
        "−": "-",   # math minus
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # --- 3. Restore missing dash after 'Item xx' if replaced by double spaces ---
    text = re.sub(r"(?i)(Item\s+\d{1,3})\s{2,}(?=\S)", r"\1 - ", text)

    # --- 4. Clean spacing / newline inconsistencies ---
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # keep paragraph boundaries
    return text.strip()


def parse_items(file_path: str) -> pd.DataFrame:
    """
    Parse BSID-III-like items from a text file into a structured DataFrame.
    Handles both clean and mojibaked inputs.
    """
    raw = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    raw = sanitize_text(raw)

    # --- Split into item blocks (capture item number) ---
    parts = re.split(
        r"(?i)\bItem\s+(\d{1,3})\s+(?:[-–—\"“”'’]|\s{2,}|[^A-Za-z0-9\s]{1,4})\s+",
        raw,
    )

    records = []
    for i in range(1, len(parts), 2):
        num = parts[i].strip()
        rest = parts[i + 1]

        # Title and content
        title, _, block = rest.partition("\n")
        title = title.strip()
        block = block.replace("\r", "\n")
        block = re.sub(r"[ \t]+$", "", block, flags=re.M).strip()

        # --- MATERIALS ---
        materials = ""
        m_mat = re.search(
            r"(?ism)^\s*Material\s*:\s*(?P<txt>.+?)(?=\n\s*\n|\n\s*(?:Scoring\s*:|\d+\s*points?\s*:)|\Z)",
            block,
        )
        if m_mat:
            materials = re.sub(r"\s*\n\s*", " ", m_mat.group("txt")).strip()
            mat_end = m_mat.end()
        else:
            mat_end = 0

        # --- DESCRIPTION ---
        m_score_hdr = re.search(r"(?mi)^(?:Scoring\s*:|\d+\s*points?\s*:)", block)
        score_start = m_score_hdr.start() if m_score_hdr else len(block)

        desc_region_start = mat_end if m_mat else 0
        description = block[desc_region_start:score_start].strip()
        description = re.sub(
            r"(?ism)^\s*Material\s*:\s*.+?(?=\n\s*\n|\n\s*(?:Scoring\s*:|\d+\s*points?\s*:)|\Z)",
            "",
            description,
        ).strip()
        description = re.sub(r"\s*\n\s*", " ", description).strip()

        # --- SCORING ---
        scoring_block = block[score_start:]
        scoring_entries = clean_scoring_block(scoring_block)
        scoring_compact = " ".join(
            [
                f"[{e['points']} pt] {e['criteria']}"
                if e["points"] is not None
                else f"{e['criteria']}"
                for e in scoring_entries
            ]
        )

        # --- Compose item_text ---
        item_text = f"{title}."
        if materials:
            item_text += f" Materials: {materials}."
        if description:
            item_text += f" Description: {description}."
        if scoring_compact:
            item_text += f" Scoring: {scoring_compact}"
        item_text = item_text.strip()

        item_id = f"COG_{int(num):03d}"
        records.append(
            {
                "item_id": item_id,
                "item_title": title,
                "item_materials": materials,
                "item_description": description,
                "scoring_json": json.dumps(scoring_entries, ensure_ascii=False),
                "item_text": item_text,
            }
        )

    return pd.DataFrame.from_records(records)
