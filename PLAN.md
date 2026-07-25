# Eco-Loop Building Agents — 24-Hour Build Plan

Starter kit already built and tested: `run_loop.py`, `tools.py`, `llm_agent.py`,
`dashboard/generate_report.py`, `ARCHITECTURE.md`. This plan is about getting
from that scaffold to a submittable, working PoC.

Rubric weights driving priority order: **System Integration (30%) > Energy
Efficiency (25%) > Comfort (20%) > Agentic Autonomy (15%) > Docs (10%)**.
That means: get something that *runs reliably end-to-end* before you polish
anything — a working mock-mode loop beats a half-finished real-EnergyPlus
integration every time.

---

## Hour 0–3: Environment + parallel installs

- [ ] Start EnergyPlus installer download/install in the background (longest single wait of the day — kick it off first, do everything else while it runs)
- [ ] `ollama pull qwen2.5-coder:7b`, confirm `ollama serve` responds
- [ ] `pip install -r requirements.txt --break-system-packages`
- [ ] Run `python3 run_loop.py --mode mock` — confirm it logs cleanly (baseline mock loop, no LLM yet)
- [ ] Find a base IDF: `find /usr/local/EnergyPlus-* -iname "*SmallOffice*.idf"` — copy the smallest one to `models/baseline.idf`, grab matching `.epw`

**Checkpoint:** EnergyPlus installed, mock baseline run produces `logs/baseline.csv`.

---

## Hour 3–7: Wire the LLM into the mock loop first

Prove the agent decision logic works on the cheap simulator before adding
EnergyPlus complexity on top — don't debug two unknowns at once.

- [ ] Run `python3 run_loop.py --mode mock-ai` — confirm the LLM actually gets called, tool calls resolve, `set_zone_setpoint` values look sane
- [ ] Watch for: model refusing to call tools, malformed JSON in tool args, setpoints oscillating every cycle — tune `SYSTEM_PROMPT` in `llm_agent.py` if so
- [ ] Confirm the comfort clamp in `tools.py` actually rejects out-of-band requests (test by asking for e.g. 40°C manually)
- [ ] Generate first dashboard: `python3 dashboard/generate_report.py` against the two mock logs — confirm % savings number appears, even if small/fake at this stage

**Checkpoint:** mock-mode closed loop runs start-to-finish, dashboard produces a savings number. This alone is your fallback demo if EnergyPlus fights you later.

---

## Hour 7–13: Real EnergyPlus integration

- [ ] Open your chosen IDF, note actual zone names (`Zone` objects) and thermostat setpoint schedule names
- [ ] Do a one-off dry run of `api.exchange.get_object_names(state, "Zone")` etc. to confirm handle names resolve — this is the fiddly part, budget real time for it
- [ ] Fill in the `# handles for setpoint actuators` TODO in `run_real_energyplus()` — likely via `Schedule:Compact` actuator override rather than writing EMS Erl code
- [ ] Run `python3 run_loop.py --idf models/baseline.idf --epw models/weather.epw --mode baseline` — get a clean baseline log from the *real* simulator
- [ ] Then `--mode ai` — get the real closed loop running

**Checkpoint:** real EnergyPlus baseline + AI runs both produce logs without crashing. If you're badly behind schedule at hour 13, stop here and ship the mock-mode version — a reliable mock beats a crashed real integration (System Integration is 30% of your score).

---

## Hour 13–17: Comfort + savings validation

- [ ] Re-run dashboard against the real EnergyPlus logs
- [ ] Sanity-check: does the AI run ever push PMV/temp outside the comfort band? If yes, tighten the clamp or the prompt — this is 20% of your score and an easy way to lose points
- [ ] If savings % looks suspiciously large (like >40%), check you're not accidentally letting the AI turn HVAC off entirely — judges will probe this
- [ ] Run the AI mode over a longer horizon (multi-day if time allows) to demonstrate the "extended simulation time horizon" reliability the rubric explicitly asks about

**Checkpoint:** you have real numbers — baseline kWh, AI kWh, % reduction, and evidence comfort bounds held.

---

## Hour 17–20: Documentation

- [ ] Fill in every bracketed section in `ARCHITECTURE.md` with your *actual* numbers and decisions (latency measured, savings achieved, any failure handling you added)
- [ ] Fill the presentation template (mentioned in the brief) with: problem framing, architecture diagram, tool-calling flow, results chart, lessons learned
- [ ] Push code to GitHub, double check `.idf` files (baseline + any runtime-modified versions) are actually committed — it's an explicit deliverable

**Checkpoint:** repo has code + IDFs + dashboard output + architecture doc, all committed.

---

## Hour 20–23: Demo video

- [ ] Script the 3 minutes tightly: (1) show baseline running normally ~20s, (2) show AI agent making a live decision — zoom on a terminal/log showing the LLM's tool call and reasoning ~90s, (3) show the setpoint actually changing in the sim and the dashboard's final % savings number ~50s, (4) one sentence on comfort constraints holding ~20s
- [ ] Record, don't overthink production quality — clarity of the loop actually working live matters far more than editing polish (System Integration + Energy Efficiency are 55% combined; Presentation is only 10%)

---

## Hour 23–24: Buffer + submission

- [ ] Re-read the upload instructions — PDF/ZIP only, convert anything else
- [ ] Submit GitHub URL, upload architecture doc + presentation + video
- [ ] Triple-check nothing depends on a service (Ollama) that won't be running when judges look at the repo — note in the README exactly how to reproduce

---

## Non-negotiable fallback rule

At **any** checkpoint where you're more than ~2 hours behind, drop back to
mock mode and polish that instead of continuing to fight EnergyPlus. A
clean, honest "we validated the closed-loop agent logic on a lightweight
thermal model and have partial real-EnergyPlus integration documented as a
next step" scores far better on System Integration (30%) than a repo that
crashes when judges try to run it.
