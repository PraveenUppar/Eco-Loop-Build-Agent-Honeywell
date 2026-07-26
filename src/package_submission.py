"""Build the submission archive and verify every deliverable is present.

The portal accepts PDF or ZIP only. This assembles a single ZIP containing all
five required deliverables, checks each one exists before writing, and fails
loudly if anything is missing rather than shipping a silently incomplete
archive.

    python src/package_submission.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg

ARCHIVE = cfg.ROOT / "submission" / "eco_loop_submission.zip"

# Directories copied wholesale, minus the exclusions below.
TREES = ["src", "models", "dashboard"]

EXCLUDE_DIRS = {"__pycache__", ".venv", "out", "results_2week",
                "results_3week", "submission", ".git"}

# The submission portal scans inside the archive and rejects executable script
# formats ("UnAcceptable file format: Python script"). Source code is delivered
# through the GitHub repository URL instead -- deliverable 1 asks for the repo,
# not for code in the upload -- so no .py ships here. The deliverable checks
# below still verify every source file exists on disk.
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".py"}

# Individual files at the repo root. The architecture document is a section of
# README.md rather than a separate file, so there is nothing extra to add here.
FILES = ["README.md", "SUBMISSION.md", "requirements.txt", ".gitignore"]

# Results: keep the analysis outputs, drop the bulky raw EnergyPlus dumps.
RESULT_GLOBS = [
    "summary.json",
    "agent_optimized_metrics.json",
    "*/metrics.json",
    "*/timeseries.csv",
    "*/run_log.jsonl",
    "*/llm_calls.jsonl",
    "*/violations.txt",
    "endurance/endurance.json",
]

# Deliverable -> the file that proves it. Checked before the archive is written.
DELIVERABLES = {
    "1. Source code": ["src/eplus_runner.py", "src/agent.py",
                       "src/mcp_server.py", "src/run_experiment.py"],
    "2. Building models": ["models/baseline.idf", "models/simulation.idf",
                           "models/agent_optimized.idf"],
    "3. Savings dashboard": ["dashboard/report.html", "results/summary.json"],
    # The four required architecture sections are headings inside README.md;
    # there is no separate ARCHITECTURE.md any more.
    "4. Architecture document": ["README.md"],
    # The demo video and the presentation are both built outside the repo and
    # submitted alongside this archive, so neither has a source file to verify.
    "Submission map": ["SUBMISSION.md"],
}


def included(path: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return False
    return path.suffix not in EXCLUDE_SUFFIX


def collect() -> list[Path]:
    found: list[Path] = []
    for tree in TREES:
        for path in (cfg.ROOT / tree).rglob("*"):
            if path.is_file() and included(path.relative_to(cfg.ROOT)):
                found.append(path)
    for name in FILES:
        path = cfg.ROOT / name
        if path.exists():
            found.append(path)
    for pattern in RESULT_GLOBS:
        found.extend(p for p in cfg.RESULTS.glob(pattern) if p.is_file())
    return sorted(set(found))


def verify() -> list[str]:
    missing = []
    for label, paths in DELIVERABLES.items():
        absent = [p for p in paths if not (cfg.ROOT / p).exists()]
        status = "OK  " if not absent else "MISS"
        print(f"  [{status}] {label}")
        for p in absent:
            print(f"         missing: {p}")
            missing.append(f"{label}: {p}")
    return missing


def main() -> None:
    print("Checking deliverables\n")
    missing = verify()
    if missing:
        print(f"\n{len(missing)} missing item(s). Archive not written.")
        raise SystemExit(1)

    files = collect()
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        ARCHIVE.unlink()

    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(cfg.ROOT))

    size_mb = ARCHIVE.stat().st_size / 1024 / 1024
    print(f"\nwrote {ARCHIVE}")
    print(f"  {len(files)} files, {size_mb:.1f} MB")
    print("\nSource code is NOT in this archive by design -- the portal rejects")
    print(".py files. Deliverable 1 is satisfied by the GitHub repository URL.")
    print("\nStill to add by hand:")
    print("  - GitHub repository URL (deliverable 1: source code)")
    print("  - demo video (max 3:00)")
    print("  - presentation PDF (export the 6-slide deck to PDF)")
    print("  - dashboard PDF (open dashboard/report.html, Ctrl+P, Save as PDF)")
    print("\nIf the portal still rejects this zip, follow its own fallback:")
    print("print README.md and dashboard/report.html to PDF and upload those.")


if __name__ == "__main__":
    main()
