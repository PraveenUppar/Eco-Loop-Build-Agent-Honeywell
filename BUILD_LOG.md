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

- **1.5B model doesn't reliably use Ollama's structured tool-calling
  (fixed 2026-07-25).** First `mock-ai` run: setpoint never left the
  default 21°C across 24 steps. Root cause: `ollama show qwen2.5-coder:1.5b
  --template` requires the model to wrap tool calls in
  `<tool_call>...</tool_call>` tags for Ollama's parser to populate
  `message.tool_calls`; the 1.5B model instead dumps JSON-ish text
  (sometimes markdown-fenced, sometimes with `"arguments": null` or
  `"arguments"` as a JSON-encoded string, sometimes multiple JSON objects
  concatenated) directly into `message.content`, which the loop was
  treating as final reasoning text and discarding. Fixed by adding
  `_parse_fallback_tool_calls()` in `llm_agent.py`: scans message content
  for balanced-brace JSON objects, matches ones with a recognized tool
  `name`, and normalizes malformed `arguments`. Verified against all the
  malformed patterns actually observed - see git history of
  `llm_agent.py` for the exact test. After the fix, setpoints actually
  changed (18/24/23/20/24.5°C across a 6h test) instead of flatlining.

- **Setpoint oscillation with no clear driver (fixed 2026-07-25).** The
  6h test above ran entirely during unoccupied hours (PMV=0 throughout,
  no comfort signal to react to) and the model still swung the setpoint
  almost every cycle (21→18→24→23→20→24.5°C) for no evident reason -
  exactly the "setpoints oscillating every cycle" failure mode PLAN.md
  warns about, and it costs energy (each swing re-chases a new target).
  Tightened `SYSTEM_PROMPT` in `llm_agent.py` to explicitly say: while
  unoccupied, hold one steady low setback value and don't change it
  without a real reason; re-calling `set_zone_setpoint` with the same
  value is fine and expected. **Re-verify this actually reduced
  oscillation** by checking the next full 24h `mock-ai` run's
  `setpoint_c` column during unoccupied hours - it should look mostly
  flat, not jumping every cycle.

- **AI run used MORE energy than baseline despite zero comfort violations
  (fixed 2026-07-25, needs re-verification).** First post-fix 24h run:
  AI = 14.35 kWh vs baseline 12.6 kWh (-13.9%, i.e. worse), but 0 comfort
  violations vs baseline's 11. Root cause: the model held unoccupied
  setpoints around 23-24C instead of near the safe range's low end - it
  followed "stay steady" but not "stay low," because "near the low end of
  the safe range" is too vague for a 1.5B model to translate into an
  actual number. Fixed by making `SYSTEM_PROMPT` state the unoccupied
  setback explicitly as `18.0C`, not a description. **Needs
  re-verification** against a fresh 24h run's kWh totals and comfort
  violations once the run below completes.

- **A single agent call hung indefinitely (fixed 2026-07-25).** During
  the re-verification run above, one LLM call apparently fell into a
  degenerate repetition loop (no output, `ollama ps` stuck showing
  "Stopping..." for 10+ minutes, `python3.13.exe`'s Ollama subprocess CPU
  time still climbing - so not deadlocked, just generating tokens
  forever) - the `ollama.chat()` call had no cap on response length or
  request timeout, so nothing would stop it. Fixed in `llm_agent.py`:
  added `options={"num_predict": 300}` to the chat call (hard cap on
  generated tokens per turn) and `ollama.Client(timeout=60)` (HTTP-level
  timeout as a second safety net). If a run ever hangs again, check
  `ollama ps` - if `UNTIL` is stuck (not counting down and not showing a
  future unload time) for more than ~30s past a call's usual latency,
  it's this failure mode, not a real hang in our own code.

- **CSV log unreadable by pandas (fixed 2026-07-25).** After the hang-fix
  re-run completed cleanly, `pandas.read_csv('logs/mock-ai.csv')` failed
  with `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb0` - the
  model's free-form reasoning text apparently included a `°` character,
  and `run_loop.py::open_log()` opened the file with `open(path, "w",
  newline="")` and no explicit encoding, which defaults to the Windows
  locale codepage (cp1252) rather than UTF-8. cp1252 encodes `°` as byte
  `0xB0`, which isn't valid UTF-8, so it wrote fine but failed to read
  back. Fixed by adding `encoding="utf-8"` to that `open()` call. **Any
  log file written before this fix (i.e. `logs/mock-ai.csv` from the
  hang-fix re-run) is corrupt and must be regenerated**, not just re-read.

- **Agent couldn't anticipate occupancy, causing a severe cold-start
  comfort violation (fixed 2026-07-25).** After the encoding fix, a clean
  24h run showed the agent holding 18.0C right through the start of
  occupancy, only reacting afterward - PMV dropped to -2.62 (way past the
  -1.0 to 1.0 bound) and stayed in violation for ~4 hours (16/40 occupied
  steps). This was worse than the very first "zero violations" run, which
  in hindsight wasn't real anticipation - the model just happened to
  already be at a warm setpoint from earlier oscillation, before that got
  fixed. Root cause: `get_zone_state` gave the agent no way to know
  occupancy was *about to* start - it could only react after the fact,
  never anticipate. Fixed by adding
  `MockBuilding.minutes_until_occupancy_change()` (`tools.py`), exposing
  it in `get_zone_state`'s return value and its tool-schema description,
  and adding an explicit, numeric rule to `SYSTEM_PROMPT`: preheat to
  21.5C when unoccupied with <=120 minutes until occupancy starts,
  otherwise hold 18.0C. **Needs re-verification** against the run below -
  check PMV no longer excurses sharply at occupancy start.

- **CONCLUSION: qwen2.5-coder:1.5b cannot do this task (2026-07-25).**
  After five prompt iterations produced, in order: setpoints stuck at
  default; random oscillation; energy 13.9% WORSE than baseline; 40/40
  occupied steps in comfort violation; and finally a setpoint of `18.915`
  - a value appearing nowhere in the prompt, which permitted only 18.0 or
  22.0, and which is exactly the zone temperature from the preceding step
  echoed back as a setpoint. Decisive isolated test: given a zone with
  `occupied=true, pmv=-2.3` and a prompt whose RULE 1 reads
  `occupied is true -> setpoint 22.0`, the model returned
  `set_zone_setpoint(18.0)`. **Do not spend further effort on prompt
  wording** - there is no phrasing this model will follow more literally
  than the one it ignored. Also fixed along the way: `temperature: 0.0`
  and `seed: 42` in `llm_agent.py`'s chat options, because at Ollama's
  default sampling temperature identical code produced wildly different
  24h outcomes, making every comparison meaningless.

- **Added deterministic supervisory override layer (`supervisor.py`,
  2026-07-25).** User's chosen path forward. The agent proposes a
  setpoint; `supervise()` vetoes proposals that are physically
  indefensible and substitutes a safe value. Three override triggers
  only: (1) occupied zone outside the comfort band with a proposal that
  doesn't target the band, (2) occupancy within `PREHEAT_LEAD_MINUTES`
  with a proposal too low to preheat, (3) empty building with occupancy
  not imminent and a proposal above setback. Anything merely suboptimal
  is left alone, so the agent keeps real authority whenever its choice is
  defensible.
  - **Honesty requirement, do not remove**: `OverrideStats` tracks the
    intervention rate, `run_loop.py` logs `agent_proposed_c`,
    `overridden`, and `override_reason` per step, and the dashboard
    displays the override percentage with a caption stating that a high
    rate means the results reflect the rules rather than the LLM. A
    `--no-supervisor` flag measures unassisted agent quality. Reporting
    savings from this setup *without* the override rate would
    misrepresent what the agent achieved - it is the difference between
    a defensible engineering pattern (real BMS systems do exactly this)
    and overstating the LLM's contribution.
  - **Bug caught by its own unit test**: the first version checked
    `proposed > zone_temp` to decide whether a proposal was "correcting"
    a cold zone, which let the original failure (setpoint parked at 18.0C
    in a 17.5C occupied zone) pass as valid - 18.0 > 17.5, but 18.0 can
    never reach the 20.0C comfort floor. Fixed to test against the
    comfort band, not the current temperature. All 10 supervisor cases
    now pass; re-run that test after touching `supervise()`.

- **The mock energy model was unphysical and exploitable (fixed
  2026-07-25).** It charged `abs(setpoint - zone_temp) + abs(drift)`,
  so holding `setpoint == zone_temp` cost almost nothing - but real
  buildings bleed heat continuously and the HVAC must keep replacing it.
  This is what the LLM's bizarre `18.915` output was: it had found the
  exploit and was echoing the current zone temperature back to zero out
  its energy score. Replaced with a heating-only degree-day model:
  `heat_input = clamp(0, capacity, envelope_loss*(zone-outdoor) +
  hvac_gain*(setpoint-zone))`, energy = heat_input * hours. Verified:
  energy now rises monotonically with setpoint, and setting
  setpoint == zone_temp still costs standing loss. **Every energy number
  measured before this fix is meaningless** - they were scored against a
  broken metric.

- **Controller had steady-state droop (fixed 2026-07-25).** The plain
  proportional term meant a 21.0C setpoint settled the zone at 19.2C, so
  the baseline's nominal 21C was really delivering 19.2C and every
  comfort comparison was skewed. The new formulation (offset envelope
  loss, then close the remaining gap, capped by `hvac_capacity`)
  converges on the setpoint exactly - measured droop now 0.00C.

- **`occupied` and `minutes_until_occupancy_change` were stale at
  decision time (fixed 2026-07-25).** Both read `state.occupied`, which
  is only refreshed inside `step()`, so at the moment a control decision
  was made it described the *previous* step. At exactly 08:00 this made
  the horizon report 1440 minutes instead of 0 and ordered a setback just
  as occupancy began - the direct cause of the last remaining comfort
  violation. Both now derive occupancy from `minute_of_day` via
  `_occupied()`. Boundary behaviour verified at 07:45/08:00/08:15 and
  17:45/18:00/18:15.

- **KEY RESULT: there is almost no energy headroom in this model
  (2026-07-25).** With the physics corrected, comparing the supervisor
  rules against the fixed-schedule baseline over 48h:
  - rules: **0/80 comfort violations**, baseline: **8/80**
  - energy: rules **-0.5%** vs baseline (i.e. marginally worse)
  Swept across climates (base outdoor 14C / 8C / 2C / -4C) the energy
  difference stays within +-1% every time. `OPTIMAL_STOP_MINUTES` was
  measured to save exactly 0.00 kWh here (afternoon outdoor temp exceeds
  the indoor target, so heating is already off before the coast window).
  Interpretation: a well-chosen fixed setback schedule (21C occupied /
  18C setback) already captures nearly all available savings in a
  single-zone building with static occupancy. The honest framing of the
  result is **"strictly better comfort (8 violations -> 0) at
  approximately equal energy"**, not an energy-savings headline.
  - **Implication for the demo**: to show meaningful savings, the
    scenario needs something a static schedule *cannot* exploit. The
    brief explicitly names two: **peak demand thresholds** and **local
    carbon grid intensity**. Adding a time-varying price/carbon signal
    creates real optimization headroom (shift preheating into cheap or
    low-carbon hours, coast through peaks) that no fixed schedule can
    capture. That is the recommended next move, not further tuning of
    these constants - the constants are already at their measured
    optimum and tuning them further is just fitting the benchmark.

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
