# Presentation Content — Eco-Loop Building Agent

Slide-by-slide content to paste into the provided template. Keep the
speaker notes; they answer the questions judges actually ask.

---

## Slide 1 — Title

**Eco-Loop Building Agent**
Closed-loop LLM supervisory control for building HVAC

Praveen Uppar · github.com/PraveenUppar/Eco-Loop-Build-Agent-Honeywell

---

## Slide 2 — Problem

Buildings are ~40 % of global energy use.

Conventional BMS run **fixed schedules**: 21 °C at 08:00, 18 °C at 18:00,
every day — blind to weather, occupancy, and grid conditions.

A fixed schedule cannot react to anything it wasn't programmed with.

> **Notes:** Lead with why a *schedule* is the limitation, not the
> hardware. That framing sets up why an agent helps.

---

## Slide 3 — Architecture

```
Simulation ──state──> LLM agent ──proposed setpoint──> Supervisor ──> Simulation
(zone temp,          (Ollama,                        (comfort +          ↑
 occupancy,           tool calling)                    load-shift        │
 PMV, price)                                           rules)  ──────────┘
```

- **Tools:** `get_zone_state`, `get_grid_forecast`, `set_zone_setpoint`
- **Two safety layers:** situational supervisor + hard clamp (18–26 °C)
- **Model:** `qwen2.5-coder:1.5b`, local via Ollama — no cloud, no API key

> **Notes:** Emphasise that `set_zone_setpoint` is the *only* write path
> into the simulation. The LLM cannot bypass either safety layer.

---

## Slide 4 — Closed loop in action

Per cycle: read state → reason → call tool → setpoint injected → next
cycle observes the consequence.

- 192 timesteps, 48 simulated hours, zero crashes
- ~2 s mean decision latency
- Decision cadence decoupled from simulation step (`--agent-every-n-steps`)

> **Notes:** This is the 30 %-weighted System Integration criterion.
> Stress reliability over an extended horizon — that's the exact wording
> in the brief.

---

## Slide 5 — The key insight

**With flat electricity pricing, a good fixed schedule is already near
optimal.** We measured this: ±1 % across every climate tested.

Real headroom appears only when something varies that a static schedule
cannot track — **time-of-use price and grid carbon intensity**.

> **Notes:** This is the strongest slide. It shows you measured before
> optimising, and that you understand *why* the naive approach is hard to
> beat. Most teams will claim big savings against a strawman baseline.

---

## Slide 6 — Results

| | Cost | CO₂ | Peak-hour energy | Comfort violations |
|---|---|---|---|---|
| Fixed schedule | — | — | 7.57 kWh | 8 / 80 |
| Eco-Loop | **−4.0 %** | **−2.1 %** | **−23.3 %** | **0 / 80** |

Total energy rises ~1 %. **The strategy shifts *when* energy is bought,
not how much.**

*(Insert `dashboard/load_shift.png`)*

> **Notes:** Explain the chart in one sentence: taller orange bars before
> the peak = buying cheap energy and storing it as heat; shorter bars
> inside the shaded window = coasting on it. Expect a question on why
> total kWh went up — the answer is thermal losses while storing, paid
> for by a 4× cheaper price per kWh.

---

## Slide 7 — Honest evaluation

We report the **supervisor override rate: 95.8 %**.

A capability probe settles why. Reduced to two booleans and a three-line
lookup table, the model scored 3/6 — and the three correct answers were
exactly the three where the answer was the fallback line. It emits a
constant rather than reading its input.

**The savings belong to the deterministic control logic, not the LLM.**

> **Notes:** Do not skip this slide. Judges probe exactly here, and a team
> that surfaces its own limitation is far more credible than one caught
> out. We also ship a `mock-rules` arm — the same rules with no LLM — as
> an experimental control, which is what makes the claim checkable.

---

## Slide 8 — Engineering rigour

Bugs found and fixed, each documented in `BUILD_LOG.md`:

- **Exploitable energy model** — cost was charged only on the
  setpoint/temperature gap, so holding them equal was nearly free. The
  LLM found the exploit and echoed the zone temperature back as its
  setpoint. Replaced with a degree-day model.
- **Controller droop** — a 21 °C setpoint settled at 19.2 °C, making the
  baseline's comfort fictitious.
- **Stale occupancy at 08:00** — the horizon reported 1440 minutes
  instead of 0, ordering a setback as occupants arrived.
- **Stacked drift rules** — two savings rules fired together and pushed
  the zone through the comfort floor.

> **Notes:** This slide is the Code Elegance part of the 15 % criterion.
> Finding and fixing your own measurement bugs is the strongest possible
> evidence that the numbers are real.

---

## Slide 9 — Limitations & next steps

- **EnergyPlus integration scaffolded, not complete** — results come from
  a lightweight thermal model; PyEnergyPlus callbacks are in place with
  handle resolution left as `TODO`
- **Model too small** — `qwen2.5-coder:7b` needs ~16 GB RAM; the dev
  machine has 8 GB and a 4 GB GPU. One `--model` flag away.
- **Single zone, static occupancy** — multi-zone with stochastic
  occupancy is where LLM reasoning would plausibly beat a rule table

> **Notes:** Frame the model limit as a hardware constraint you measured,
> not a project failure. You have unusually good evidence for it.

---

## Slide 10 — Summary

- Working closed-loop agentic pipeline, 192 steps, zero crashes
- Load shifting cuts peak-hour energy 23 % and cost 4 %
- Comfort violations eliminated (8 → 0)
- Every claim backed by a logged, reproducible run — including the
  claims that went against us
