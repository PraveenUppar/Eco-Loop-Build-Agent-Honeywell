# Submission map

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

[`README.md § Architecture`](README.md#architecture) — the four required sections are its four subheadings:

1. **Tool-calling architecture** — MCP diagram, the 5 tools, why supervisory rather than per-timestep control, the cross-process bus
2. **Prompt engineering strategy** — teaching the domain rather than dictating answers, JSON schema enforcement, history compaction, context-aware constraints, self-correction
3. **Prompt latency management** — the arithmetic, supervisory cadence, decision cache, measured figures
4. **Handling lengthy simulation logs** — filter → deduplicate → truncate, with measured before/after token counts
