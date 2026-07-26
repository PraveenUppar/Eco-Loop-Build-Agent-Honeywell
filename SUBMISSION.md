# Submission map

Where each required deliverable lives, and where each evaluation criterion is
evidenced.

---

## Required deliverables

### 1. Fully functional source code

A single Python codebase covering the EnergyPlus API wrapper, the LLM agent
orchestration, and the communication bus.

| Concern                     | File                                                                                                                                                                                                     |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **EnergyPlus API wrapper**  | [`src/eplus_runner.py`](src/eplus_runner.py) — in-process `pyenergyplus` runner, sensing + actuation callbacks, handle resolution                                                                        |
| **LLM agent orchestration** | [`src/agent.py`](src/agent.py) — prompting, JSON schema, self-correction, decision cache, fallback                                                                                                       |
| **Communication bus**       | [`src/mcp_server.py`](src/mcp_server.py) (5 MCP tools, stdio) · [`src/mcp_bridge.py`](src/mcp_bridge.py) (sync↔async bridge) · [`src/shared_state.py`](src/shared_state.py) (atomic cross-process state) |
| **Safety layer**            | [`src/policy.py`](src/policy.py) — clamp + occupancy-aware operating envelope                                                                                                                            |
| **Experiment driver**       | [`src/run_experiment.py`](src/run_experiment.py) — all three modes, one command                                                                                                                          |

Run everything with:

```bash
python src/run_experiment.py --mode all
```

### 2. Building models (.idf)

| File                                                       | What it is                                                                                                           |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [`models/baseline.idf`](models/baseline.idf)               | **Base file** — untouched `5ZoneAirCooled` from EnergyPlus ExampleFiles                                              |
| [`models/simulation.idf`](models/simulation.idf)           | Modified at build time: run period trimmed, timestep outputs instrumented                                            |
| [`models/agent_optimized.idf`](models/agent_optimized.idf) | **Modified during runtime evaluation** — the agent's applied setpoints reconstructed into `Schedule:Compact` objects |

`agent_optimized.idf` exists because actuating the API leaves no file behind —
setpoints live only in memory during the run. [`src/export_idf.py`](src/export_idf.py)
reads what the agent actually applied at each of 2,016 timesteps, takes the
median per (day type, hour), and writes a model that runs standalone with **no
LLM, no MCP server and no Python in the loop**.

Verified: it runs on its own and saves **3.79 %** with 0.00 % comfort violations.

### 3. Quantitative savings dashboard

[`dashboard/report.html`](dashboard/report.html) — self-contained, no server, no network.

Rebuild: `python dashboard/build_report.py` · **Export to PDF:** open it and press Ctrl+P → Save as PDF.

**Explicit proof of kWh reduction with comfort maintained:**

| Controller                 | Total kWh   | **Reduction** | HVAC kWh   | HVAC reduction | Peak kW   | **Comfort violations** |
| -------------------------- | ----------- | ------------- | ---------- | -------------- | --------- | ---------------------- |
| Baseline (stock schedules) | 3059.74     | —             | 904.23     | —              | 20.22     | **0.00 %**             |
| Rule-based (no LLM)        | 2990.15     | +2.27 %       | 832.82     | +7.90 %        | 19.65     | **0.00 %**             |
| **AI closed-loop agent**   | **2958.16** | **+3.32 %**   | **800.20** | **+11.50 %**   | **19.46** | **0.00 %**             |

Comfort is defined as the share of **occupied** zone-timesteps outside
20–25 °C, measured across all five conditioned zones. The unconditioned plenum
is excluded.

Raw data export: [`results/summary.json`](results/summary.json),
`results/*/metrics.json`, `results/*/timeseries.csv`.

### 4. System architecture document

[`ARCHITECTURE.md`](ARCHITECTURE.md) — the four required sections are its four headings:

1. **Tool-calling architecture** — MCP diagram, the 5 tools, why supervisory rather than per-timestep control, the cross-process bus
2. **Prompt engineering strategy** — teaching the domain rather than dictating answers, JSON schema enforcement, history compaction, context-aware constraints, self-correction
3. **Prompt latency management** — the arithmetic, supervisory cadence, decision cache, measured figures
4. **Handling lengthy simulation logs** — filter → deduplicate → truncate, with measured before/after token counts

### 5. PoC demonstration video

[`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) — timed shot list, narration, and the exact
commands for the live-loop segment.

**Not yet recorded.** The script's 0:50–1:50 segment is the required part: split
screen with EnergyPlus timesteps streaming on one side and the agent log showing
tool call → LLM reasoning → new setpoints on the other, proving data moves in
both directions.

### Presentation

[`docs/presentation.md`](docs/presentation.md) — content mapped slide-for-slide onto the
provided SIH template (6 slides including title). Needs filling into the `.pptx`
and exporting to PDF.

---

## Evaluation criteria — where the evidence is

### 1. System Integration — 30 %

_"How robustly and reliably does the closed-loop pipeline execute without crashing over an extended simulation time horizon?"_

**Endurance test — a full cooling season, run end to end without intervention.**

Reproduce with `python src/endurance_run.py` · raw output in
[`results/endurance/endurance.json`](results/endurance/endurance.json)

| | |
|---|---|
| Simulated period | **1 June – 31 August** (92 days) |
| Control timesteps | **8,832** |
| Supervisory decisions | **552** |
| Ran to completion | **Yes** — exit 0, both baseline and agent |
| Crashes / hangs | **0** |
| Clamp violations | **0** |
| Comfort violations | **0.00 %** (baseline also 0.00 %) |
| Malformed JSON from the model | **0** |
| Fell back to rule-based | **5 of 552** (0.9 %) |
| Energy saved | **+2.72 %** facility, **+11.08 %** HVAC |
| Wall clock | 12.8 min, of which EnergyPlus itself is ~13 s |

Two things this run establishes that the three-week experiment could not:

- **Nothing degrades with horizon length.** Clamp violations, comfort breaches
  and malformed responses all stay at zero across 8,832 timesteps — 4.4× the
  reported experiment. The fallback path fires 5 times and each time the
  building keeps a valid policy.
- **The decision cache improves with scale.** Hit rate rises from 31.7 % over
  three weeks to **59.8 %** over three months, because a longer horizon revisits
  more repeated (hour, occupancy, outdoor, indoor) states. Cost per simulated
  day *falls* as the run gets longer.

Savings are slightly lower than the three-week figure (2.72 % vs 3.32 %) for an
honest reason: June and August are milder than the July peak week, so there is
less cooling load to save on.

Robustness measures already in place:

- **Every handle asserted at resolution**, failing loudly and naming what is missing rather than silently reading zeros
- **Callback exceptions are caught** — a controller fault degrades the decision, it does not kill the simulation
- **Warmup and sizing-period timesteps filtered**, so design days never contaminate results
- **Atomic cross-process state** with bounded retry around the Windows `WinError 5` sharing violation that `os.replace` raises when the MCP server holds the file open
- **Guaranteed fallback** — if the LLM fails twice, the rule-based controller takes that interval; the building is never left without a policy

Verification: `python src/test_mcp_client.py` starts a real simulation in a
background thread and exercises all five tools **while it is running**.

### 2. Energy Efficiency Realized — 25 %

**3.32 % of total facility electricity, 11.50 % of HVAC electricity**, versus
standard baseline scheduling. See the dashboard table above.

The agent also **beats a hand-written rule-based controller** by 1.05 percentage
points, which isolates the contribution of the LLM from the contribution of
simply having any controller at all.

Honest framing: lights and plug loads are roughly 70 % of facility electricity
and no thermostat policy can touch them, so the HVAC-only figure is the fairer
measure of what the controller actually influences. Both are reported.

### 3. Thermal Comfort & Constraints — 20 %

**0.00 % comfort violations** — identical to baseline. The agent did not buy
energy savings with occupant comfort.

Three enforcement layers, in [`src/policy.py`](src/policy.py) and [`src/mcp_server.py`](src/mcp_server.py):

| Layer               | Enforces                                                                                   |
| ------------------- | ------------------------------------------------------------------------------------------ |
| Numeric clamp       | heating 18–22 °C, cooling 23–27 °C, deadband ≥ 1.5 °C                                      |
| Tool-layer check    | Occupied → rejects cooling > 25 °C · Empty → rejects cooling < 26 °C                       |
| Inner-loop envelope | Re-applied **every timestep**, regardless of what the supervisor requested 4 hours earlier |

The third layer matters because the supervisor re-plans only every four hours —
a setback chosen at 04:00 would otherwise still be in force at 06:00 when people
arrive.

`python src/test_clamp.py` proves the clamp holds against 17 hostile inputs
(out-of-range, strings, `None`, `NaN`/`inf`, booleans, inverted setpoints,
unknown zones, four malformed LLM payloads). **Clamp violations in the reported
run: 0.**

### 4. Agentic Autonomy & Code Elegance — 15 %

**Real MCP, not a stand-in.** A FastMCP server runs as its own stdio subprocess.
[`src/mcp_bridge.py`](src/mcp_bridge.py) keeps an async client session alive on a
background event loop so the synchronous EnergyPlus callback can call it — the
agent genuinely talks to a separate process, rather than importing the tool
functions.

**Self-correction driven by real tool feedback.** `set_setpoints` _rejects_
unsafe policies and returns the specific reason. That reason is appended to the
conversation and the model retries:

|                              | Reported 3-week run |
| ---------------------------- | ------------------- |
| Supervisory decisions        | 126                 |
| Malformed JSON               | **0**               |
| Tool rejections              | 17                  |
| Recovered by self-correction | 16                  |
| Fell back to rule-based      | **1**               |
| Cache hit rate               | 31.7 %              |

**Log handling.** [`src/log_tools.py`](src/log_tools.py) compresses a real
EnergyPlus error log **596 lines → 9, a 46.7× token reduction**, so no raw log
ever reaches the model.

**Runs on a 3B local model** — `qwen2.5:3b-instruct`, no cloud, no GPU, no API
cost. That constraint is why the validation layer has real authority instead of
trusting the model.

### 5. Presentation & Documentation — 10 %

- [`README.md`](README.md) — setup, one-command run, results, honest limitations
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the four required sections
- [`docs/presentation.md`](docs/presentation.md) — deck content on the provided template
- [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) — video script
- [`dashboard/report.html`](dashboard/report.html) — five charts, all with hover detail

---

## Reproducing everything

```bash
python -m venv .venv
```

```bash
.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

```bash
python src/prepare_model.py
```

```bash
python src/run_experiment.py --mode all
```

```bash
python src/export_idf.py
```

```bash
python dashboard/build_report.py
```

Build the submission archive:

```bash
python src/package_submission.py
```

---

## Known limitations

- **One building, one climate, cooling season.** The operating envelope is tuned
  for summer; the unoccupied setback cap (heating ≤ 19 °C) is _tighter_ than the
  baseline's own 16.7 °C night setback, so a winter run would likely show the
  agent losing to baseline. Not claimed, and flagged as the first thing to fix.
- **The savings ceiling is low by construction.** With comfort held at 20–25 °C,
  roughly 2.5–3.5 % of facility electricity is available. Letting zones drift to
  27 °C reaches ~7.5 %, but breaks comfort, so it is not claimed.
- **A 3B model is inconsistent.** Run-to-run variance is comparable to the
  effect of prompt changes. The exported IDF outperforms the live agent
  (3.79 % vs 3.32 %) precisely because taking the median discards its outliers —
  that gap _is_ the model's inconsistency, quantified.
- **No weather forecast**, so no pre-cooling ahead of a hot afternoon. The
  clearest next improvement.
- **Comfort is a temperature band, not PMV.** The model's `People` objects do not
  specify a Fanger comfort model.
