"""Render the submission deliverables to PDF.

The submission portal accepts only PDF and ZIP, and its in-archive scanner
rejects almost every source format (.py, .md, .idf, .epw, .jsonl). Its own
fallback instruction is to convert everything to PDF, which is what this does:

    README.md            -> submission/pdf/architecture.md.pdf
    SUBMISSION.md        -> submission/pdf/submission-map.pdf
    dashboard/report.html -> submission/pdf/savings-dashboard.pdf

Markdown is rendered to styled HTML, then a headless Chrome/Edge prints it.
The dashboard is already a self-contained HTML page, so it prints directly --
its own @media print rules collapse the card grid to one column.

    python src/build_pdfs.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg

OUTDIR = cfg.ROOT / "submission" / "pdf"

# Markdown sources -> output PDF name. Keys are repo-relative.
MD_DOCS = {
    "README.md": "architecture.pdf",
    "SUBMISSION.md": "submission-map.pdf",
}

HTML_DOCS = {
    "dashboard/report.html": "savings-dashboard.pdf",
}

BROWSERS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

# Print stylesheet for the rendered markdown. Mirrors the dashboard's light
# palette so the PDFs read as one set of documents.
CSS = """
@page { size: A4 portrait; margin: 16mm 14mm; }
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
       color: #0b0b0b; line-height: 1.55; font-size: 10.5pt; margin: 0; }
h1 { font-size: 20pt; letter-spacing: -0.02em; margin: 0 0 4pt;
     border-bottom: 2px solid #1baf7a; padding-bottom: 5pt; }
h2 { font-size: 14pt; margin: 18pt 0 5pt; border-bottom: 1px solid #ded9cc;
     padding-bottom: 3pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 13pt 0 3pt; color: #9b4a1f; break-after: avoid; }
h4 { font-size: 10.5pt; margin: 10pt 0 2pt; break-after: avoid; }
p, li { margin: 0 0 6pt; }
ul, ol { margin: 0 0 8pt; padding-left: 18pt; }
a { color: #1c5cab; text-decoration: none; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt;
       background: #f1efe7; padding: 1pt 3pt; border-radius: 2pt; }
pre { background: #f1efe7; border: 1px solid #ded9cc; border-radius: 4pt;
      padding: 7pt 9pt; overflow-x: auto; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.5pt; line-height: 1.4; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt;
        break-inside: avoid; }
th, td { border: 1px solid #ded9cc; padding: 4pt 6pt; text-align: left;
         vertical-align: top; }
th { background: #f1efe7; font-weight: 600; font-size: 8.5pt;
     text-transform: uppercase; letter-spacing: 0.03em; color: #52514e; }
blockquote { margin: 8pt 0; padding: 5pt 10pt; border-left: 3px solid #1baf7a;
             background: #f6f9f7; color: #333; }
hr { border: none; border-top: 1px solid #ded9cc; margin: 14pt 0; }
strong { font-weight: 650; }
"""


def find_browser() -> Path | None:
    for path in BROWSERS:
        if path.exists():
            return path
    for name in ("chrome", "msedge"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def render_markdown(md_path: Path) -> Path:
    """Markdown -> standalone styled HTML in a temp file beside the PDFs."""
    import markdown

    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
    )
    html = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{md_path.stem}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )
    tmp = OUTDIR / f"_{md_path.stem}.tmp.html"
    tmp.write_text(html, encoding="utf-8")
    return tmp


def to_pdf(browser: Path, source: Path, out_pdf: Path,
           settle_ms: int = 0) -> bool:
    """Print a local HTML file to PDF with headless Chrome/Edge."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if out_pdf.exists():
        out_pdf.unlink()

    uri = source.resolve().as_uri()
    cmd = [
        str(browser),
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={out_pdf}",
    ]
    # The dashboard draws its charts with JavaScript, so give the renderer time
    # to finish before the page is captured.
    if settle_ms:
        cmd.append(f"--virtual-time-budget={settle_ms}")
    cmd.append(uri)

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not out_pdf.exists():
        print(f"  FAILED {out_pdf.name}")
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        for line in err[-4:]:
            print(f"         {line}")
        return False
    return True


def main() -> None:
    browser = find_browser()
    if not browser:
        raise SystemExit(
            "No Chrome or Edge found. Install one, or print each file manually "
            "with Ctrl+P -> Save as PDF.")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"using {browser.name}\n")

    written: list[Path] = []
    temps: list[Path] = []

    for rel, out_name in MD_DOCS.items():
        src = cfg.ROOT / rel
        if not src.exists():
            print(f"  SKIP {rel} (not found)")
            continue
        tmp = render_markdown(src)
        temps.append(tmp)
        out = OUTDIR / out_name
        if to_pdf(browser, tmp, out):
            written.append(out)
            print(f"  {rel:24} -> {out.name}")

    for rel, out_name in HTML_DOCS.items():
        src = cfg.ROOT / rel
        if not src.exists():
            print(f"  SKIP {rel} (not found)")
            continue
        out = OUTDIR / out_name
        # Charts are JS-rendered; allow time for Plotly to draw.
        if to_pdf(browser, src, out, settle_ms=20000):
            written.append(out)
            print(f"  {rel:24} -> {out.name}")

    for tmp in temps:
        try:
            tmp.unlink()
        except OSError:
            pass

    if not written:
        raise SystemExit("\nNo PDFs produced.")

    print(f"\nwrote {len(written)} PDF(s) to {OUTDIR}")
    for path in written:
        print(f"  {path.name:28} {path.stat().st_size / 1024:8.0f} KB")

    print("\nUpload these, plus:")
    print("  - the 6-slide presentation, exported to PDF")
    print("  - the demo video (max 3:00)")
    print("  - the GitHub repository URL (deliverable 1: source code,")
    print("    and deliverable 2: the .idf building models)")


if __name__ == "__main__":
    main()
