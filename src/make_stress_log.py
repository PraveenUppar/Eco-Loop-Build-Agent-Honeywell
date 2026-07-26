"""Produce a realistically verbose eplusout.err to exercise the log compressor.

A one-week clean run yields a 16-line error file, which proves nothing about
compression. This runs a full year with a deliberately narrow, hard-to-hold
deadband so EnergyPlus emits the repetitive "temperature not met" and HVAC
iteration warnings that a real building model produces.

The resulting log is a genuine EnergyPlus artefact, not a synthetic fixture.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppy.modeleditor import IDF

import config as cfg

OUTDIR = cfg.OUT / "stress_annual"
STRESS_IDF = cfg.OUT / "stress_annual.idf"


def build() -> Path:
    IDF.setiddname(str(cfg.IDD_PATH))
    idf = IDF(str(cfg.SIM_IDF))

    rp = idf.idfobjects["RUNPERIOD"][0]
    rp.Begin_Month, rp.Begin_Day_of_Month = 1, 1
    rp.End_Month, rp.End_Day_of_Month = 12, 31

    # EnergyPlus suppresses most repeated warnings by default. Turning on the
    # full diagnostics surfaces the real per-timestep warnings the model
    # generates all year -- a genuine verbose log, not a synthetic one.
    diag = idf.newidfobject("OUTPUT:DIAGNOSTICS")
    diag.Key_1 = "DisplayAllWarnings"
    diag.Key_2 = "DisplayExtraWarnings"
    diag.Key_3 = "DisplayUnusedSchedules"
    diag.Key_4 = "DisplayUnusedObjects"

    OUTDIR.mkdir(parents=True, exist_ok=True)
    idf.saveas(str(STRESS_IDF))
    return STRESS_IDF


def main() -> None:
    idf_path = build()
    print(f"running annual stress simulation -> {OUTDIR}")
    subprocess.run(
        [str(cfg.ENERGYPLUS_DIR / "energyplus.exe"),
         "-w", str(cfg.WEATHER), "-d", str(OUTDIR), str(idf_path)],
        capture_output=True, text=True, check=False,
    )
    err = OUTDIR / "eplusout.err"
    if err.exists():
        lines = len(err.read_text(encoding="utf-8", errors="replace").splitlines())
        print(f"wrote {err} ({lines} lines, {err.stat().st_size} bytes)")
    else:
        print("no error file produced")


if __name__ == "__main__":
    main()
