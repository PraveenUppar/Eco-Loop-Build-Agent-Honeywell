# Build Log — Eco-Loop Building Agent

Running record of decisions made, why, and what's been verified. Read this
first in any new conversation before touching the code — it explains the
*why* behind choices that aren't obvious from the code alone. Update it
whenever a new non-obvious decision gets made or a checkpoint is hit.

See also: [PLAN.md](PLAN.md) (hour-by-hour build schedule),
[problem details.txt](problem%20details.txt) (original brief),
[ARCHITECTURE.md](ARCHITECTURE.md) (submission deliverable, fill in once
real numbers exist), [README.md](README.md) (setup/run instructions).

---

## Starting state (2026-07-25)

Repo contained only `PLAN.md` and `problem details.txt` — none of the
starter-kit code the plan assumes (`run_loop.py`, `tools.py`,
`llm_agent.py`, `dashboard/generate_report.py`, `ARCHITECTURE.md`)
actually existed. Built from scratch following the plan's own strategy.

## Key decisions

- **Mock-first, EnergyPlus-second.** EnergyPlus isn't installed on this
  machine. Rather than block on that, built a lightweight fake building
  (`MockBuilding` in `tools.py`) that exposes the same interface
  (`get_zone_state`/`set_zone_setpoint`) the real EnergyPlus integration
  will use. This proves the LLM agent logic works before adding
  EnergyPlus's setup complexity on top — isolates which of the two
  systems is broken if something goes wrong. Matches PLAN.md's own
  Hour 0-7 sequencing and its "non-negotiable fallback rule."

- **Direct tool-calling, not a standalone MCP server.** User chose this
  explicitly to minimize moving parts under a deadline. `tools.py`'s
  `TOOL_SCHEMAS` are already in OpenAI/MCP-compatible function-schema
  shape, so migrating to a real MCP server later is a transport swap, not
  a redesign, if there's time left after core deliverables.

- **Model: qwen2.5-coder:7b via Ollama.** Chosen for reliable JSON
  tool-calling at a size that runs locally with reasonable per-step
  latency, and because it's what PLAN.md's own Hour 0-3 step assumes.
  Not uniquely correct — `qwen2.5-coder:1.5b` (~1GB) was the fallback
  discussed if the 7B download kept failing on a flaky connection.
  **Model name lives in one place** — `DEFAULT_MODEL` in
  `llm_agent.py` and the `--model` flag default in `run_loop.py` — change
  both if swapping.

- **Safety clamp is a hard rule in code, not a prompt instruction.**
  `tools.py::clamp_setpoint()` mechanically restricts every setpoint
  request to `[MIN_SETPOINT_C, MAX_SETPOINT_C]` = [18, 26]°C regardless
  of what the LLM asks for. The narrower `[COMFORT_MIN_C, COMFORT_MAX_C]`
  = [20, 24]°C band is what the dashboard checks for violations. This
  two-tier design (hard safety rail vs. comfort target) is the load-bearing
  safety idea judges are likely to probe ("does the AI ever cheat by
  turning HVAC off entirely?").

- **`eppy` dropped from `requirements.txt` for now.** It pulls in a heavy,
  unrelated sphinx/nbsphinx doc-building dependency chain. It's only
  needed once real `.idf` parsing starts (Hour 7-13), so it's commented
  out in `requirements.txt` until then, to keep the install fast on a slow
  connection.

## Bugs found and fixed during testing

- **Thermal model too sluggish (fixed 2026-07-25).** Original
  `MockBuilding` defaults (`envelope_loss=0.06, hvac_gain=0.35,
  thermal_mass=0.9`) meant zone temp took over 5 simulated hours to climb
  from an overnight setback (17.8°C) back toward the occupied setpoint
  (21°C) — never actually reaching the comfort band in a realistic
  workday. This was inflating "comfort violation" counts for reasons that
  had nothing to do with the control strategy. Retuned to
  `envelope_loss=0.08, hvac_gain=0.4, thermal_mass=0.5` — zone now reaches
  the comfort band ~2 hours after occupancy starts, which is a
  believable "morning warm-up lag" a smarter (preheating) AI strategy can
  legitimately improve on. If further tuning is needed, check
  `logs/mock.csv` for the temp trajectory in the first occupied hours of
  each day.

## Verified working (2026-07-25)

- `pip install -r requirements.txt` (ollama, pandas, matplotlib) — done.
- `python run_loop.py --mode mock --hours 24` — runs cleanly, produces
  `logs/mock.csv` (96 steps at 15-min resolution).
- `python dashboard/generate_report.py --baseline logs/mock.csv --ai
  logs/mock.csv` — runs cleanly, produces `dashboard/report.html` +
  two PNG charts (self-comparison smoke test only, 0% savings expected
  since it's mock vs itself).
- **Not yet verified**: `mock-ai` mode end-to-end (needs `ollama serve`
  running + model pulled — see Known issues below), and all of
  `baseline`/`ai` real-EnergyPlus modes (EnergyPlus not installed here).

## Known issues / in-progress

- **Ollama model download unstable on this network.** `ollama pull
  qwen2.5-coder:7b` (4.7GB) repeatedly failed mid-transfer with TLS/
  connection errors to Ollama's Cloudflare R2-backed registry, at ~40
  KB/s-1.8MB/s. Ollama's pull resumes from cached progress on retry, so
  the fix was just "keep re-running the pull command." User resolved
  this by retrying manually.

- **Switched default model to `qwen2.5-coder:1.5b` (hardware constraint,
  2026-07-25).** After the 7B model finished downloading, running it
  crashed every time: first `model requires more system memory (1.8 GiB)
  than is available (1.4 GiB)`, then after freeing RAM (~2.4-3GB free),
  `llama runner process has terminated` with no further detail — same
  crash on repeat attempts. Checked hardware: 8GB total system RAM, and
  the GPU (`nvidia-smi`) is an RTX 2050 with only 4GB VRAM. A 4.7GB model
  file doesn't comfortably fit either the available system RAM headroom
  or the GPU VRAM on this machine while anything else is running — this
  is a hardware ceiling, not a transient resource blip, so **don't retry
  7B on this machine**. Pulled `qwen2.5-coder:1.5b` (986MB) instead and
  updated the default model in `llm_agent.py::DEFAULT_MODEL`,
  `run_loop.py`'s `--model` flag default, and `README.md` accordingly.
  `qwen2.5-coder:7b` is still pulled locally (`ollama list` shows both)
  in case a different, higher-RAM/VRAM machine is used later — just pass
  `--model qwen2.5-coder:7b` there.

## Next steps (in order)

1. Confirm `qwen2.5-coder:7b` finished downloading (`ollama list`) and
   `ollama serve` is running.
2. Run `python run_loop.py --mode mock-ai --hours 24`, watch for the
   failure modes PLAN.md flags: tool refusal, malformed JSON args,
   setpoint oscillation. Tune `SYSTEM_PROMPT` in `llm_agent.py` if seen.
3. Run the real dashboard comparison: `python dashboard/generate_report.py
   --baseline logs/mock.csv --ai logs/mock-ai.csv` — this is the first
   real (non-smoke-test) savings number.
4. Sanity-check the result: if savings >40%, check the AI isn't just
   disabling HVAC; if comfort violations are high, tighten
   `SYSTEM_PROMPT` or the comfort band.
5. Once mock-mode is solid, move to real EnergyPlus (PLAN.md Hour 7-13):
   install EnergyPlus, find zone/schedule names from an example IDF, fill
   in the `TODO` block in `run_loop.py::run_real_energyplus()`.
6. Fill in `ARCHITECTURE.md`'s bracketed sections with real numbers once
   available.
7. `git init`/commit — repo already has `origin` set to
   `https://github.com/PraveenUppar/Eco-Loop-Build-Agent-Honeywell.git`,
   no commits made yet as of this log entry. **Important**: `.gitignore`
   must NOT exclude `dashboard/report.html` or its PNG charts — that
   dashboard is deliverable #3 in the brief. (Confirmed fixed in
   `.gitignore` as of this log entry — double check before pushing.)
