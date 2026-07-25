# Eco-Loop Building Agent

Closed-loop LLM control of a building HVAC zone: an open-source LLM
(via Ollama) reasons over live zone state and calls tools to adjust the
setpoint, aiming to cut energy use while holding thermal comfort inside
a safe band. See [PLAN.md](PLAN.md) for the build plan and
[problem details.txt](problem%20details.txt) for the original brief.

## Status

- Mock-mode closed loop (lightweight thermal simulator standing in for
  EnergyPlus): **built**, see `tools.py` / `llm_agent.py` / `run_loop.py`.
- Real EnergyPlus integration: **not wired yet** — EnergyPlus isn't
  installed on this machine. `run_real_energyplus()` in `run_loop.py` has
  the integration scaffolded with explicit `TODO`s for the zone/actuator
  handles (Hour 7-13 in PLAN.md).

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

## Running (real EnergyPlus — once installed)

1. Install EnergyPlus, then add its install directory to `PYTHONPATH` so
   `pyenergyplus` is importable:
   ```bash
   export PYTHONPATH=$PYTHONPATH:/usr/local/EnergyPlus-24-1-0
   ```
2. Find an example IDF (EnergyPlus ships examples), e.g.:
   ```bash
   find /usr/local/EnergyPlus-* -iname "*SmallOffice*.idf"
   ```
   Copy it and its matching `.epw` weather file into `models/`.
3. Open the IDF and note the `Zone` object name(s) and the
   `Schedule:Compact` object used for the thermostat setpoint — fill
   these into the `TODO` block in `run_real_energyplus()` in
   [run_loop.py](run_loop.py).
4. Run baseline, then AI:
   ```bash
   python run_loop.py --mode baseline --idf models/baseline.idf --epw models/weather.epw
   python run_loop.py --mode ai       --idf models/baseline.idf --epw models/weather.epw
   python dashboard/generate_report.py --baseline logs/baseline.csv --ai logs/ai.csv
   ```

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
