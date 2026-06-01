"""
build_pdf.py
Converts report_2026-05-24.md -> report_2026-05-24.pdf
  - 1-inch margins, 11pt Times New Roman
  - Tables: centered, compact, vertical column lines (via table_vlines.lua)

Usage:
    cd report/
    python3 build_pdf.py
"""

import subprocess
import sys
from pathlib import Path

REPORT_DIR = Path(__file__).parent
MD_FILE    = REPORT_DIR / "Z23971583_Project_Report.md"
PDF_OUT    = REPORT_DIR / "Z23971583_Project_Report.pdf"
LUA_FILTER = REPORT_DIR / "table_vlines.lua"

# ── 1. Generate PDF via pandoc (Lua filter handles tables) ────────────────────
print("Running pandoc -> xelatex...")
result = subprocess.run(
    [
        "pandoc", str(MD_FILE),
        "-o", str(PDF_OUT),
        "--pdf-engine=xelatex",
        f"--lua-filter={LUA_FILTER}",
        "-V", "geometry:margin=1in",
        "-V", "fontsize=11pt",
        "-V", "mainfont=Times New Roman",
    ],
    capture_output=True, text=True, cwd=REPORT_DIR
)

if result.returncode != 0:
    print("ERROR:")
    print(result.stderr)
    sys.exit(1)

if result.stderr.strip():
    # Show warnings but don't fail
    print("Warnings:", result.stderr.strip())

print(f"Done: {PDF_OUT}")
