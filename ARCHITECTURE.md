# Architecture

A local LLM supervises a live EnergyPlus simulation through an MCP tool layer,
reading sensors and writing thermostat setpoints while the simulation runs.

---

## 1. Tool-calling architecture

### The loop

```mermaid
flowchart TB
    subgraph sim["EnergyPlus process (in-process API)"]
        EP["EnergyPlus 26.1<br/>5-zone VAV office"]
        CB1["callback:<br/>begin_system_timestep_before_predictor"]
        CB2["callback:<br/>end_zone_timestep_after_zone_reporting"]
        INNER["Inner loop (deterministic, every 15 min)<br/>clamp -> operating envelope -> actuators"]
        EP -->|every timestep| CB1 --> INNER -->|set_actuator_value| EP
        EP -->|once per zone timestep| CB2
    end

    subgraph shared["Shared state (atomic JSON, os.replace)"]
        RS[["runtime_state.json<br/>sim -> tools"]]
        PP[["pending_policy.json<br/>tools -> sim"]]
    end

    subgraph mcpsrv["MCP server process (stdio)"]
        T1["get_building_state"]
        T2["get_energy_summary"]
        T3["set_setpoints"]
        T4["read_simulation_errors"]
        T5["list_zones"]
    end

    LLM["qwen2.5:3b-instruct<br/>via Ollama (local)"]

    CB2 -->|publish sensors| RS
    CB2 -->|every 4 sim hours| AGENT["Agent supervisor"]
    AGENT -->|MCP call| T1
    AGENT -->|MCP call| T2
    AGENT -->|prompt| LLM
    LLM -->|JSON policy| AGENT
    AGENT -->|MCP call| T3
    T3 -->|validated| PP
    PP -->|consumed next timestep| INNER
    T3 -.->|rejection + reason| AGENT
    RS --> T1 & T2 & T5
```

### Why two callbacks

| Callback | Job | Why this one |
|---|---|---|
| `begin_system_timestep_before_predictor` | **Actuate** | Fires before the zone predictor computes loads, so a setpoint written here affects the current timestep. This is the correct hook for setpoint control. |
| `end_zone_timestep_after_zone_reporting` | **Sense + decide** | Fires exactly once per zone timestep. EnergyPlus shortens the *system* timestep during difficult HVAC iterations, so reading meters in the actuation callback would double-count energy. |

### Why supervisory, not per-timestep

The LLM sets **policy**; a deterministic inner loop **applies** it every timestep.

This is an engineering decision, not a shortcut:

- **Latency.** At 15-minute timesteps a three-week run is 2016 timesteps. Calling
  a model on each one, at ~2.9 s per call, would add **over 90 minutes** to a
  simulation that itself takes 8 seconds.
- **It matches how buildings are actually controlled.** Real BMS supervisory
  logic resets setpoints on a slow loop; fast local control loops track them.
  Putting a language model in the inner loop would be the wrong design even if
  it were free.
- **Stability.** A policy that holds for four hours cannot oscillate at
  timestep frequency.

### The five tools

| Tool | Returns / does |
|---|---|
| `get_building_state()` | Zone temperatures and RH, outdoor drybulb, occupancy, energy to date, and the policy currently in force |
| `get_energy_summary(window_hours)` | kWh and peak kW over a trailing window, plus the running delta against the baseline run |
| `set_setpoints(heating_c, cooling_c, zone, reason)` | **Validates** and applies a setpoint policy, or rejects it with a specific, actionable reason |
| `read_simulation_errors(max_lines)` | Severity-filtered, deduplicated `.err` summary (see §4) |
| `list_zones()` | Conditioned zone names and floor areas |

### Crossing the process boundary

The MCP server is a separate stdio subprocess, so it cannot see the
simulation's Python objects. The exchange is two small JSON files written with
`os.replace`, which is atomic on NTFS, so a reader never sees a half-written
file.

One real bug surfaced here and is worth recording: on Windows, `os.replace`
raises `WinError 5` if the destination is currently open in another process.
With the server polling state while the simulation writes it 2016 times, this
fired constantly. The fix is a short bounded retry on both sides, treating
state as telemetry — a dropped write is preferable to killing a simulation,
because the next timestep republishes.

### Calling async tools from a sync callback

The EnergyPlus callback is an ordinary synchronous function; the MCP client
session is asyncio-based and must stay open for the whole run. `mcp_bridge.py`
runs the session on its own event loop in a background thread and submits
coroutines with `run_coroutine_threadsafe`. The agent therefore talks to a
**real MCP server over stdio** rather than importing the tool functions
directly.

### The safety envelope

No policy reaches an actuator without passing through validation, whatever its
source. This is load-bearing rather than decorative, because the supervisor is
a 3B model.

```
LLM output -> JSON schema -> clamp_pair() -> tool-layer occupancy check
           -> operating_envelope() every timestep -> actuator
```

| Layer | Enforces |
|---|---|
| `clamp_pair` | heating 18–22 °C, cooling 23–27 °C, deadband ≥ 1.5 °C. Non-numeric, `NaN`, `inf`, booleans and inverted pairs all resolve to a safe policy. |
| `set_setpoints` | Context-aware. **Occupied:** rejects cooling above 25 °C. **Empty:** rejects cooling below 26 °C — running plant for nobody. Rejections carry the reason, which drives self-correction. |
| `operating_envelope` | Runs in the inner loop *every timestep*, independent of what the supervisor asked for 4 hours ago. Pulls setpoints into the comfort band when occupied; pushes them out to setback when empty. |

The middle layer matters because the supervisor decides every four hours: a
setback chosen at 04:00 would otherwise still be in force at 06:00 when people
arrive.

`test_clamp.py` exercises this against deliberately hostile input — out-of-range
values, strings, `None`, `NaN`/`inf`, booleans, inverted setpoints, unknown zone
names and four malformed LLM payloads. Every case lands inside the envelope.

---

## 2. Prompt engineering strategy

### Teach the domain, do not dictate the answer

The first version of the system prompt listed rules ("raise cooling_c to save
energy"). It underperformed, and the failure was diagnosable: on hot days the
model chose cooling 23.5 °C — the most expensive legal option — reasoning that
it was hot outside so more cooling was needed.

That is a **conceptual** error, not an instruction-following error. The model
was treating the cooling setpoint as "amount of cooling" rather than as a
threshold. So the prompt was rewritten to explain the mechanism:

- what a setpoint actually is (a threshold, not a target)
- what the deadband is, and that no energy is spent inside it
- therefore *why* raising `cooling_c` reduces energy, and lowering it increases energy
- the specific misconception, named and corrected
- what occupancy changes about the objective
- why sudden setpoint drops create demand peaks

A model that understands the mechanism generalises to situations the rules do
not enumerate. Rules alone produced a model that pattern-matched "hot → cool
harder".

### Structured output, not prose parsing

Ollama is given an explicit **JSON schema**, not just `format: "json"`:

```python
POLICY_SCHEMA = {
  "type": "object",
  "properties": {"heating_c": {"type": "number"},
                 "cooling_c": {"type": "number"},
                 "reason":    {"type": "string"}},
  "required": ["heating_c", "cooling_c", "reason"],
}
```

This does more for reliability on a 3B model than any amount of polite asking:
**0 malformed JSON responses** across every reported run.

### History compaction

The model receives the last **six** decisions as a fixed-width table — time,
setpoints, cumulative kWh, mean zone temperature — not a transcript. Cost is
bounded regardless of run length, and the trend stays visible.

```
RECENT DECISIONS (most recent last)
  time        htg   clg   kWh/4h  avgC
  07-03 08:15 20.0  25.0  421.0   23.8
```

### Context-aware constraint block

The legal range is restated **per call**, because it depends on occupancy:

```
ALLOWED NOW (people present): heating_c 20.0-22.0, cooling_c 23.0-25.0.
The top of the cooling range is the cheap end; the bottom is the expensive end.
```

Stating the currently-binding constraint in the user turn — where a small model
weights it most — cut tool rejections sharply.

### Self-correction

When the model returns unparseable JSON, or the tool layer rejects the policy,
the **specific failure** is appended to the conversation and the model retries
(up to 2 attempts):

> That was rejected: building is OCCUPIED: cooling_c 27.0 exceeds the comfort
> limit 25.0 C. Reply with corrected JSON only, obeying …

If it still cannot produce a usable policy, the rule-based controller takes
that interval. The building is never left without a policy.

### Budget

System prompt ≈ 620 tokens, per-call user prompt ≈ 360 tokens — roughly **1000
tokens per decision**, comfortably inside the 1500-token target.

---

## 3. Prompt latency management

### The arithmetic

| | Per-timestep control | Supervisory (chosen) |
|---|---|---|
| Timesteps (3 weeks @ 15 min) | 2016 | 2016 |
| Supervisory decisions | 2016 | **126** |
| Model invocations (incl. retries) | ~2050 | **102** |
| At 3.27 s median | **~110 minutes** | **5.8 minutes** |
| Simulation itself | 12 s | 12 s |

Calling the model every timestep would make the LLM roughly **550×** the cost of
the physics it is steering. Re-planning every 4 simulated hours brings that to
something a demo can actually run.

### Decision cache

Situations repeat — 02:00 on consecutive empty weeknights at the same outdoor
temperature is the same decision. Decisions are cached on a **bucketed** key:

```python
(hour // 4, occupied, round(outdoor_temp), round(mean_zone_temp))
```

Rounding is what makes the cache hit at all; exact float state never repeats.

Note that `llm_calls` counts every model invocation, retries included, so it is
not the decision count. One decision costs either a cache hit, or one call plus
however many retries it needed:

```
decisions = llm_calls - retries + cache_hits
          = 102       - 16      + 40         = 126
```

### Measured (3-week run)

| | |
|---|---|
| Supervisory decisions | 126 |
| Model invocations | 102 |
| Cache hits | 40 — **31.7 %** hit rate |
| Median latency | **3.27 s** |
| Total LLM time | 348 s |
| Total wall clock | 360 s |
| **EnergyPlus simulation alone** | **12 s** |

The agent's wall clock is almost entirely LLM time; the physics underneath is
twelve seconds. That ratio is the whole argument for supervisory cadence — and
the cache removes nearly a third of the calls outright.

### What else keeps it off the critical path

- `num_predict` capped at 160 tokens; the reply is three fields
- History fixed at six rows, so prompt size does not grow with run length
- Cache hits cost **0 ms** — no request is made
- A failed call degrades to the rule-based policy rather than blocking

---

## 4. Handling lengthy simulation logs

An EnergyPlus `.err` file is mostly repetition: the same warning re-emitted per
timestep or per object, with only the identifiers changing. Passing it raw into
a prompt spends the context window on noise.

### The pipeline

**1. Filter.** Keep only `** Severe **`, `** Fatal **` and `** Warning **`
records. Drop the informational preamble and the `**   ~~~   **` continuation
lines, which carry no new signal.

**2. Deduplicate.** Normalise the *varying* parts out of each message so repeats
hash together, then count:

- timestamps and date strings → `<TIME>`
- upper-case EnergyPlus object identifiers → `<NAME>`
- quoted strings → `<NAME>`
- all numbers → `#`

Identifier substitution runs **before** quote substitution — the other order
lets the placeholder match itself and produces `<<NAME>>`.

**3. Truncate.** Rank by severity first, then occurrence count, and keep the top
K. The worst problems survive truncation; the tool never returns a raw log.

Collapsed entries keep up to three concrete example names, so the reader retains
enough context to act:

```
[WARNING] Calculated design heating load for zone=<NAME> is zero.  x14
          (e.g. FLOOR 1 KITCHEN, FLOOR 2 CLEAN 2, FLOOR 2 EXAM 3...)
```

### Measured compression

Tested on a genuine EnergyPlus log — an annual run of the `HospitalBaseline`
example with `Output:Diagnostics, DisplayAllWarnings` enabled. Not a synthetic
fixture.

| | Before | After | Reduction |
|---|---|---|---|
| Lines | 596 | 9 | **66×** |
| Approx. tokens | 13,769 | 295 | **46.7×** |
| Distinct issues | — | 8 (from 56 occurrences) | — |

Reproduce with:

```bash
python src/log_tools.py out/big_log2/eplusout.err 15
```

The project's own model is well-conditioned and produces a clean 16-line `.err`,
which would demonstrate nothing — hence testing the pipeline against a large,
genuinely warning-heavy model.

---

## 5. Robustness over an extended horizon

The three-week experiment is the reported result, but it does not by itself show
that the loop survives a long run. `src/endurance_run.py` runs the identical
pipeline over a full cooling season and records whether anything degrades.

| | 3-week experiment | **Endurance run** |
|---|---|---|
| Period | 1–21 July | **1 June – 31 August** |
| Timesteps | 2,016 | **8,832** |
| Supervisory decisions | 126 | **552** |
| Crashes / hangs | 0 | **0** |
| Clamp violations | 0 | **0** |
| Comfort violations | 0.00 % | **0.00 %** |
| Malformed JSON | 0 | **0** |
| Fallbacks | 1 (0.8 %) | **5 (0.9 %)** |
| Cache hit rate | 31.7 % | **59.8 %** |
| Savings | +3.32 % | +2.72 % |

**The failure rate does not grow with horizon length** — fallbacks stay under
1 % of decisions, and the three "must never happen" counters (clamp violations,
comfort breaches, malformed JSON) stay at zero across 4.4× the timesteps.

**The cache gets better with scale.** Hit rate nearly doubles over the longer
run, because more (hour-block, occupancy, outdoor, indoor) states repeat. The
marginal cost of each additional simulated day *falls* as the run lengthens —
the opposite of the usual scaling worry with an LLM in the loop.

What makes the loop survive:

- **Handles asserted at resolution**, failing loudly and naming the missing one
  rather than silently reading zeros
- **Callback bodies wrapped** — a controller fault degrades one decision instead
  of killing the simulation
- **Warmup and sizing periods filtered** by an explicit run-period guard, so
  design days never contaminate results
- **Bounded retry** around the Windows `WinError 5` sharing violation, treating
  published state as telemetry: a dropped write is preferable to a dead run
- **Guaranteed fallback** — two failed LLM attempts hand the interval to the
  rule-based controller

Savings are lower over the season (2.72 % vs 3.32 %) for an honest reason: June
and August are milder than the July peak week, so there is less cooling load to
save on.

---

## Repository map

| Path | Purpose |
|---|---|
| `src/config.py` | Single source of truth: zones, schedules, limits, run period |
| `src/prepare_model.py` | Trims the stock example to a run week and instruments it |
| `src/eplus_runner.py` | In-process EnergyPlus runner; sensing and actuation callbacks |
| `src/policy.py` | `Policy`, `clamp_pair`, `operating_envelope` — the safety layer |
| `src/controllers.py` | Deterministic controllers (baseline, fixed, rule-based) |
| `src/mcp_server.py` | FastMCP stdio server exposing the five tools |
| `src/mcp_bridge.py` | Sync facade over the async MCP session |
| `src/agent.py` | LLM supervisor: prompt, schema, self-correction, cache |
| `src/log_tools.py` | `.err` filter / dedupe / truncate pipeline |
| `src/metrics.py` | Metrics computed from `eplusout.csv` |
| `src/run_experiment.py` | Runs the three modes and writes comparable results |
| `src/export_idf.py` | Bakes the agent's policy into `agent_optimized.idf` |
| `src/compare_models.py` | Side experiment: same loop, different LLM |
| `src/test_clamp.py` | Safety-clamp unit tests |
| `src/test_actuation.py` | Proves actuation moves energy in both directions |
| `src/test_mcp_client.py` | Standalone MCP client; exercises tools mid-simulation |
| `dashboard/build_report.py` | Builds the self-contained HTML report |
