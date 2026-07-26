# Presentation content — SIH Idea Submission

Content mapped **slide-for-slide** onto `IDEA_Presentation_Format.pptx`.

Template rules being followed:
- **Maximum 6 slides**, including the title page (delete the instructions slide)
- **No paragraphs** — points, diagrams, infographics, pictures
- Do not change the template's own heading pointers
- Export to **PDF** before uploading

Fill anything marked `[FILL]` — I don't have those values.

---

## Slide 1 — TITLE PAGE

| Field | Value |
|---|---|
| Problem Statement ID | `[FILL]` |
| Problem Statement Title | `[FILL]` |
| Theme | `[FILL]` |
| PS Category | **Software** |
| Student Name | **Praveen Uppar** |
| Student ID | `[FILL]` |

**Idea title to use throughout:**

> **Eco-Loop — a local AI agent that controls a building's HVAC in a closed loop**

---

## Slide 2 — IDEA TITLE / Proposed Solution

### Detailed explanation of the proposed solution

- A **local language model supervises a live building simulation** — reading sensors and writing thermostat setpoints **while the simulation is running**
- Model talks to the building through an **MCP tool layer** (5 tools), not hard-coded calls
- **Supervisory control:** the AI sets policy every 4 simulated hours; a deterministic loop applies and guards it every 15 minutes
- Runs **fully offline on one laptop** — no cloud, no GPU, no API costs

### How it addresses the problem

- Building thermostats run **fixed schedules written once and rarely revisited**
- They cannot reason about today's weather, or whether anyone is actually in the room
- The stock building holds a **1.7 °C deadband (22.2–23.9 °C)** all day — expensive, and unnecessary when the building is empty
- Eco-Loop reasons per situation and widens the deadband exactly when it is safe to

### Innovation and uniqueness

- **Closed loop, not offline analysis** — data moves *both* directions with a running physics simulation
- **Runs on a 3 B model** — so the safety layer does real work rather than trusting the model
- **The tool layer has authority:** it *rejects* unsafe setpoints and tells the model why → the model self-corrects
- **Exports what it learned** into a standalone building model that runs with no AI at all

### Headline result

> **3.32 % of total electricity and 11.5 % of HVAC electricity saved — with ZERO comfort violations**
> Beats a hand-written rule-based controller by **1.05 percentage points**

**Suggested visual:** the 3 stat callouts above as large numbers (60–72 pt), plus the
cumulative-energy chart from `dashboard/report.html`.

---

## Slide 3 — TECHNICAL APPROACH

### Technologies used

| Layer | Technology |
|---|---|
| Building physics | **EnergyPlus 26.1**, 5-zone VAV office, Chicago TMY3 weather |
| Live control | **`pyenergyplus`** C API — in-process callbacks, not file round-trips |
| Tool layer | **MCP** (Model Context Protocol) — FastMCP server over stdio |
| AI model | **Ollama + `qwen2.5:3b-instruct`**, running locally |
| Model editing | **eppy** — writes the learned policy back into an `.idf` |
| Reporting | **pandas + Plotly** — self-contained HTML report |

### Methodology — the closed loop

```
EnergyPlus (5-zone office)
      |
      |  every 15 min: zone temps, humidity, outdoor air, occupancy, energy
      v
Deterministic inner loop  ──►  applies + guards setpoints  ──►  back into EnergyPlus
      |
      |  every 4 simulated hours
      v
MCP tools ──► local LLM ──► JSON policy ──► VALIDATED ──► applied
      ^                                          |
      └────── rejected, with the reason ─────────┘   (self-correction)
```

### The 5 MCP tools

1. `get_building_state()` — temperatures, humidity, occupancy, energy, active policy
2. `get_energy_summary(window_hours)` — kWh, peak demand, delta vs baseline
3. `set_setpoints(heating, cooling, zone, reason)` — **validates and can reject**
4. `read_simulation_errors(max_lines)` — compresses huge simulation logs
5. `list_zones()` — zone names and floor areas

### Two callbacks, chosen deliberately

- **Write setpoints** *before* the load predictor runs → affects the current timestep
- **Read sensors** at end of zone timestep → avoids double-counting energy

### Three-layer safety envelope

| Layer | Guarantees |
|---|---|
| Numeric clamp | heating 18–22 °C, cooling 23–27 °C, deadband ≥ 1.5 °C |
| Tool-layer check | Occupied → rejects cooling > 25 °C · Empty → rejects cooling < 26 °C |
| Inner-loop guard | Re-applied **every** timestep, whatever the AI asked for 4 hours ago |

**Suggested visual:** the loop diagram above, drawn as boxes and arrows.

---

## Slide 4 — FEASIBILITY AND VIABILITY

### Feasibility — already built and measured

- **Working system, not a concept** — three controllers run from one command
- Simulation of 3 building-weeks completes in **12 seconds**
- Runs on **commodity hardware**: no GPU, no cloud, no per-token cost
- **Zero** safety-clamp violations across 2,016 control timesteps

### Analysis — measured results

| Controller | kWh | Saved | HVAC saved | Comfort violations |
|---|---|---|---|---|
| Baseline (stock schedules) | 3059.74 | — | — | 0.00 % |
| Rule-based (no AI) | 2990.15 | +2.27 % | +7.90 % | 0.00 % |
| **AI agent** | **2958.16** | **+3.32 %** | **+11.50 %** | **0.00 %** |

### Potential challenges and risks

| Risk | Why it is real |
|---|---|
| **Latency** | Calling the AI every timestep = **110 minutes** on a 12-second simulation |
| **Small-model reliability** | A 3 B model produces inconsistent, occasionally wrong answers |
| **Comfort vs energy** | Naively maximising savings pushes rooms to 27 °C — occupants suffer |
| **Log volume** | Simulation error logs repeat the same warning hundreds of times |

### Strategies for overcoming them

| Strategy | Measured outcome |
|---|---|
| **Supervisory cadence** — AI sets policy, fast loop applies it | 2016 timesteps → **126 decisions** |
| **Decision cache** on bucketed state | **31.7 %** of decisions need no AI call at all |
| **JSON schema** forced on the model output | **0** malformed responses |
| **Validate → reject → explain → retry**, then fall back to rules | 17 rejections, 16 self-corrected, **1 fallback in 126** |
| **Occupancy-aware envelope** enforced every timestep | **0.00 %** comfort violations |
| **Filter → deduplicate → truncate** the logs | **596 lines → 9** (46× fewer tokens) |

### Viability / scale-up path

- Policy exports to a **standalone `.idf`** — deployable with no AI in the loop
- Same tool layer maps onto a real BMS (BACnet/Modbus) without redesign
- Bigger model = better judgment; the plumbing already shows 0 JSON failures

---

## Slide 5 — ARTIFACTS

### Code

- **Repository:** `[FILL — repo URL]`
- ~2,600 lines of Python across 15 modules

| Component | File |
|---|---|
| Live simulation runner (sensors + actuators) | `src/eplus_runner.py` |
| MCP server, 5 tools | `src/mcp_server.py` |
| AI supervisory agent | `src/agent.py` |
| Safety clamp + operating envelope | `src/policy.py` |
| Log compression pipeline | `src/log_tools.py` |
| Policy → standalone building model | `src/export_idf.py` |

### Snaps to include

1. **Dashboard** — `dashboard/report.html` → headline tiles + comfort-band chart
2. **Setpoint chart** — every AI decision marked as a dot on the setpoint trace
3. **Live agent log** — terminal showing `[agent] 07-05 12:15 -> 20.0/25.0 (2.9s)`
4. **Self-correction** — a rejected setpoint and the corrected retry
5. **Log compression** — before/after of `python src/log_tools.py`

### Verification (all passing)

| Test | Proves |
|---|---|
| `test_clamp.py` | 17 hostile inputs — all land inside the safety envelope |
| `test_actuation.py` | Loose setpoints **−7.52 %**, tight **+1.47 %** — control works both ways |
| `test_mcp_client.py` | All 5 tools work **against a live, running simulation** |

### Generated artifacts

- `models/agent_optimized.idf` — the learned policy as a real building model (**3.79 %** savings, no AI needed)
- `results/summary.json`, `results/*/timeseries.csv` — all metrics
- `results/agent/llm_calls.jsonl` — every AI decision, latency and retry logged

---

## Slide 6 — RESEARCH AND REFERENCES

### Tools and standards

| Reference | Link |
|---|---|
| EnergyPlus — building simulation engine | https://energyplus.net |
| EnergyPlus Python API (EMS / `pyenergyplus`) | https://nrel.github.io/EnergyPlus/api/python/ |
| Model Context Protocol specification | https://modelcontextprotocol.io |
| Ollama — local model runtime | https://ollama.com |
| Qwen2.5 model family | https://qwenlm.github.io/blog/qwen2.5/ |
| eppy — IDF manipulation | https://eppy.readthedocs.io |

### Domain background

- **ASHRAE Standard 55** — Thermal Environmental Conditions for Human Occupancy (source of the 20–25 °C comfort band used as the constraint)
- **ASHRAE Guideline 36** — High-Performance Sequences of Operation; the basis for supervisory setpoint reset, which is the control pattern this project applies
- **EnergyPlus `5ZoneAirCooled`** reference model — the building under control, shipped with EnergyPlus ExampleFiles
- **Chicago O'Hare TMY3** weather data — typical meteorological year

### Project documentation

- `docs/ARCHITECTURE.md` — tool calling, prompt engineering, latency management, log handling
- `README.md` — setup, one-command run, results, honest limitations

---

# Notes for building the deck

### Slide budget
Template allows **6 slides including the title**. The mapping above is exactly 6 — delete the instructions slide (slide 1 of the template).

### What to cut if slides overflow
- Slide 3: drop the two-callbacks section, keep the loop diagram and the tool list
- Slide 4: drop the viability bullets, keep the results table and the risk/strategy pair
- Slide 5: keep the dashboard snap and the artifacts list; the test table can go

### Numbers to double-check before submitting
All are from `results/summary.json`, regenerated 3-week run:
- 3.32 % facility · 11.5 % HVAC · 0.00 % comfort violations
- 126 decisions · 31.7 % cache hits · 0 malformed JSON · 1 fallback
- Log compression 596 → 9 lines

### Do NOT claim
- That the agent "learns" or "trains" — it reasons per decision, there is no training loop
- The ~7.5 % figure from the loose-setpoint test — that run **breaks comfort** and exists only to prove the actuators work
- If asked why savings are single-digit: comfort is the binding constraint. Facility totals include lights and plug loads, which thermostats cannot touch — hence HVAC-only (11.5 %) is the fairer measure.
