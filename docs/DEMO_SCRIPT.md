# Demo video — script and shot list

Target: **under 3:00**. Script it, do not improvise. Check audio on a 10-second
test take before recording the full thing.

**Numbers to quote:** 3.32 % facility electricity saved, 11.5 % HVAC, 0 %
comfort violations, beats the rule-based controller by 1.05 points. 126
supervisory decisions, 0 malformed JSON, 1 fallback. Log compression 596 → 9
lines (46× fewer tokens).

---

## Before you record

```bash
python src/prepare_model.py && python src/run_experiment.py --mode all
```

```bash
python dashboard/build_report.py
```

Have ready:

- **Window A** — terminal, large font (18pt+), dark theme
- **Window B** — `dashboard/report.html` open in a browser
- **Window C** — `docs/ARCHITECTURE.md` diagram, or the deck's architecture slide

Close notifications. Set the terminal to ~120 columns so log lines do not wrap.

---

## 0:00 – 0:20 · Problem and claim

**On screen:** architecture slide, or a still of the building model.

> "Commercial buildings run on thermostat schedules written once and rarely
> revisited. They cannot reason about today's weather or whether anyone is
> actually in the room.
>
> I put a local language model in a closed loop with a live EnergyPlus
> simulation. It reads sensors and writes setpoints while the simulation runs —
> and it cuts electricity without making anyone uncomfortable."

---

## 0:20 – 0:50 · Architecture

**On screen:** Window C, the diagram. Trace the loop with the cursor as you talk.

> "EnergyPlus runs in-process through its Python API. Two callbacks: one reads
> sensors at the end of every zone timestep, one writes setpoints before the
> predictor runs — that is the correct hook for setpoint control.
>
> Every four simulated hours the agent calls an MCP server over stdio. Five
> tools: read building state, read energy, list zones, read the error log, and
> write setpoints.
>
> The key decision is that the model is a **supervisor**, not a per-timestep
> controller. Calling it every timestep would add ninety minutes to an
> eight-second simulation. It sets policy; a deterministic loop applies and
> guards that policy every fifteen minutes. That is also how real building
> management systems work."

---

## 0:50 – 1:50 · The loop, live *(the most important 60 seconds)*

**On screen:** split screen. Window A left, running the agent. Window B right.

Start the run **on camera** with the demonstration trace enabled:

```bash
python src/run_experiment.py --mode agent --demo
```

`--demo` prints every stage of one control cycle, which is exactly what
deliverable 5 asks to see — data leaving EnergyPlus, and control actions going
back in:

```
==========================================================================
[SENSE] EnergyPlus -> agent   sim time 07-05 12:15
        SPACE1  24.5  SPACE2  24.1  SPACE3  24.5  SPACE4  24.3  SPACE5  24.2  (degC)
        outdoor  31.2 C | OCCUPIED | 1421.3 kWh used so far
[TOOL ] MCP get_energy_summary -> 61.2 kWh over 4 h, peak 17.8 kW
[LLM  ] qwen2.5:3b-instruct -> {"heating_c": 20.0, "cooling_c": 25.0, ...}  (2.9s)
[VALID] accepted -> heating 20.0 C / cooling 25.0 C
[ACT  ] agent -> EnergyPlus: setpoint schedules overwritten, applied every 15 min
```

Narrate over it:

> "The simulation is running now, and you can watch one full control cycle.
>
> **SENSE** — zone temperatures, outdoor air and occupancy coming *out* of
> EnergyPlus while it runs. **TOOL** — the agent querying energy use through
> MCP. **LLM** — the local 3B model returning a policy as structured JSON.
> **VALID** — the tool layer checking it. **ACT** — new setpoints written back
> into the running simulation.
>
> That is the closed loop: data out, decision, control action back in, every
> four simulated hours, with the fast loop applying it every fifteen minutes.
>
> Watch for a REJECTED line — the model asks for 27 degrees while the building
> is occupied, the tool layer refuses and tells it why, and it corrects itself
> on the retry. That rejection is the design working, not a bug: a 3B model gets
> a validation layer with real authority."

**If you need a guaranteed self-correction on camera**, point at the log instead:

```bash
python -c "import json;[print(l['sim_time'],l.get('retries'),(l.get('response') or {}).get('reason','')[:60]) for l in map(json.loads,open('results/agent/llm_calls.jsonl',encoding='utf-8')) if l.get('retries')]"
```

**Recording tip:** speed the playback to 1.5–2× in your editor so the loop
visibly iterates. Keep your narration at normal speed.

---

## 1:50 – 2:30 · Results

**On screen:** Window B, `report.html`. Scroll top to bottom slowly.

> "Same building, same weather, same run period. The only thing that changes is
> the controller.
>
> The agent cut facility electricity by 3.3 percent over three summer weeks —
> and 11.5 percent of the HVAC energy it can actually influence. It also beat a
> hand-written rule-based controller by a point.
>
> This chart is the one that matters — mean zone temperature against the comfort
> band. The agent saves that energy without leaving the band. Comfort violations
> are zero.
>
> And here are the setpoints it actually applied. Every dot is one LLM decision.
> Hover, and you get the model's own reasoning and how many retries it took."

---

## 2:30 – 3:00 · Autonomy, and the artefact

**On screen:** terminal. Run this live — it takes a second and lands hard:

```bash
python src/log_tools.py out/big_log2/eplusout.err 15
```

> "One more thing the agent has to survive: EnergyPlus error logs. They repeat
> the same warning hundreds of times. Feeding that raw into a prompt burns the
> context window on noise.
>
> Filter by severity, collapse repeats into counts, keep the top issues:
> **596 lines to 9 — a 46× token reduction** — and no raw log ever reaches the
> model.
>
> Finally, the agent's learned policy is exported into a standalone EnergyPlus
> model that runs with no LLM, no MCP server and no Python in the loop.
>
> Code is at [REPO LINK]. Thanks for watching."

---

## Checklist

- [ ] Audio checked on a short test take
- [ ] Terminal font ≥ 18pt, ~120 columns, no wrapping
- [ ] Notifications off
- [ ] Live-loop segment shows `[agent]` lines advancing
- [ ] At least one self-correction visible or shown from the log
- [ ] Comfort-band chart on screen while saying "comfort was maintained"
- [ ] Setpoint chart hover demonstrated
- [ ] Repo link on the final frame
- [ ] Total runtime **under 3:00** — verify before export

---

## Things not to claim

Judges will probe these, and the honest answer is stronger than a dodge:

- Do **not** say the agent "learns" or "trains". It reasons per decision; there
  is no training loop.
- Do **not** quote the ~7 % figure from the loose-setpoint diagnostic. That run
  breaks comfort and exists only to prove the actuators work.
- If asked why savings are single-digit: comfort is the binding constraint. The
  honest ceiling with the band intact is roughly 2.5–3 % of facility
  electricity, and about ten times that share of HVAC-only energy.
- If asked whether a bigger model would do better: yes, probably. The plumbing
  shows zero malformed JSON and zero fallbacks, so the limit is model judgment,
  not the loop.
