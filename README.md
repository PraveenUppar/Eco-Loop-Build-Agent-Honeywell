# Eco-Loop Building Agent

Closed-loop LLM control of a building HVAC zone: an open-source LLM
(via Ollama) reasons over live zone state and calls tools to adjust the
setpoint, aiming to cut energy use while holding thermal comfort inside
a safe band. See [PLAN.md](PLAN.md) for the build plan and
[problem details.txt](problem%20details.txt) for the original brief.

## Status

- **Mock-mode closed loop: built and measured.** Over 48 simulated hours
  against a fixed setback schedule: cost −4.0 %, carbon −2.1 %,
  peak-hour energy −23.3 %, comfort violations 8/80 → 0/80.
- **Real EnergyPlus integration: written, not yet executed.**
  `run_real_energyplus()` is fully implemented (handle resolution, meter
  reads, actuator writes, same supervisor). It has not been run, because
  EnergyPlus is not installed on the development machine — so treat it as
  untested code until you complete the steps below.
- **Honest caveat on the results:** the supervisor override rate is
  95.8 %, meaning the deterministic rules produce nearly all of the
  benefit and the 1.5B model contributes little. See
  [ARCHITECTURE.md](ARCHITECTURE.md) §3 and §7 — the capability probe and
  the no-LLM control arm are both documented there.

## Setup

```bash
pip install -r requirements.txt
ollama pull qwen2.5-coder:1.5b
ollama serve   # in a separate terminal, if not already running
```

## Running (mock mode — no EnergyPlus required)

```bash
# Rule-based fixed-schedule baseline
python run_loop.py --mode mock --hours 48

# LLM-driven closed loop
python run_loop.py --mode mock-ai --hours 48

# Compare the two and generate the dashboard
python dashboard/generate_report.py --baseline logs/mock.csv --ai logs/mock-ai.csv
```

Open `dashboard/report.html` for the savings/comfort summary.

## Running (real EnergyPlus)

**1. Install EnergyPlus** from [energyplus.net/downloads](https://energyplus.net/downloads),
then point `PYTHONPATH` at the install directory so `pyenergyplus` is importable:

```powershell
$env:PYTHONPATH = "C:\EnergyPlusV24-1-0"
```

```bash
export PYTHONPATH=$PYTHONPATH:/usr/local/EnergyPlus-24-1-0
```

Verify with `python -c "import pyenergyplus; print('ok')"`.

**2. Copy an example model.** EnergyPlus ships example files in
`ExampleFiles/` and weather in `WeatherData/`:

```powershell
copy "C:\EnergyPlusV24-1-0\ExampleFiles\RefBldgSmallOfficeNew2004_Chicago.idf" models\baseline.idf
copy "C:\EnergyPlusV24-1-0\WeatherData\USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw" models\weather.epw
```

**3. Find the zone name.** Zone names differ between models and are not
guessable — ask the model itself rather than reading the IDF by hand:

```bash
python run_loop.py --mode discover --idf models/baseline.idf --epw models/weather.epw
```

This prints every zone and writes the full actuator list to
`out/discover/eplusout.edd`.

**4. Run baseline, then AI**, passing the zone name from step 3:

```bash
python run_loop.py --mode baseline --idf models/baseline.idf --epw models/weather.epw --zone "CORE_ZN ZN"
```

```bash
python run_loop.py --mode ai --idf models/baseline.idf --epw models/weather.epw --zone "CORE_ZN ZN"
```

```bash
python dashboard/generate_report.py --baseline logs/baseline.csv --ai logs/ai.csv
```

If the log comes out empty, the zone name didn't match — the run prints a
warning naming the unresolved handles. Re-check step 3.

## Safety

`tools.py` hard-clamps every setpoint the LLM requests to
`[MIN_SETPOINT_C, MAX_SETPOINT_C]` (18–26°C) before it ever reaches the
simulation — the LLM cannot push the zone outside this range regardless
of what it asks for. The narrower `[COMFORT_MIN_C, COMFORT_MAX_C]`
(20–24°C) band is what the dashboard checks for comfort violations.

## Reproducing without Ollama running

If Ollama isn't running (e.g. a judge cloning this repo), only `--mode mock`
works out of the box — `mock-ai`, `baseline`, and `ai` all require
`ollama serve` to be reachable at `localhost:11434` with the model pulled.
