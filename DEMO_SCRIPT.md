# 3-Minute Demo Video — Shot List

Target: 2 min 50 s. The brief weights System Integration (30 %) and
Energy Efficiency (25 %) far above presentation polish (10 %), so film
the loop actually running rather than editing anything fancy.

**Before recording**

```bash
ollama serve                 # separate terminal, leave running
python run_loop.py --mode mock --hours 48
python run_loop.py --mode mock-rules --hours 48
python dashboard/generate_report.py --baseline logs/mock.csv --ai logs/mock-ai.csv
```

Have ready: one terminal (large font, ~140 chars wide), and
`dashboard/report.html` open in a browser tab.

---

## Shot 1 — The problem (0:00–0:20)

*Screen: `dashboard/load_shift.png`, or the slide with it.*

> "Buildings are about 40 % of global energy use, and most run on fixed
> schedules — heat to 21 degrees at 8am, set back at 6pm, every day,
> regardless of weather, occupancy, or what the electricity grid is
> doing. We built a closed-loop agent that replaces that fixed rule."

---

## Shot 2 — The loop running live (0:20–1:20) — **the most important shot**

*Screen: terminal. Run this live and let it scroll.*

```bash
python run_loop.py --mode mock-ai --hours 48 --agent-every-n-steps 4
```

Point at a line as it appears:

```
step 24: agent 23.5C -> applied 23.5C (1.9s) [OVERRIDE: ...]
```

> "Every cycle, the LLM calls `get_zone_state`, reads zone temperature,
> occupancy and grid price, and calls `set_zone_setpoint` — a real tool
> call, parsed and executed against the simulation. That setpoint feeds
> straight back in, and the next cycle sees the consequences. This is
> the closed loop: sense, reason, act, repeat — 192 timesteps without
> intervention."

Let it run visibly for 15–20 s. **Do not cut away** — the point is that
it doesn't crash.

---

## Shot 3 — Safety layer (1:20–1:45)

*Screen: same terminal, point at an `[OVERRIDE: ...]` line.*

> "Every setpoint passes a deterministic safety layer before it reaches
> the building. If the model asks for something that would breach the
> comfort band, it gets corrected, and the correction is logged. We
> report that override rate rather than hiding it — for this model it's
> 95.8 %, which tells you honestly how much of the result is the LLM
> versus the rules."

---

## Shot 4 — Results (1:45–2:35)

*Screen: `dashboard/report.html`, then scroll to the load-shift chart.*

> "Against the fixed-schedule baseline over 48 simulated hours: cost down
> 4 %, carbon down 2 %, and peak-hour energy down 23 %."

*Point at hours 14–15, then the shaded band.*

> "Here's the mechanism. The agent buys extra energy here, before the
> peak, while power is cheap and clean — and stores it in the building's
> thermal mass. Then during the expensive hours it coasts on that stored
> heat and barely runs. Total energy is actually slightly higher; what
> changed is *when* we bought it. A fixed schedule can't do this — it has
> no idea what hour it is in price terms."

*Point at the comfort stat.*

> "And comfort held throughout — zero violations across 80 occupied
> steps, versus 8 for the baseline."

---

## Shot 5 — Honest close (2:35–2:50)

> "Two limits worth stating. EnergyPlus integration is scaffolded but not
> complete — these results come from a lightweight thermal model. And the
> 1.5-billion-parameter model we could run on this hardware isn't strong
> enough to drive the loop on its own; we measured that, documented it,
> and the architecture takes a larger model with a single flag."

---

## Rules for filming

- **Do not fake the live run.** Judges look for exactly this. Real
  terminal, real latency, real scroll.
- **Do not claim the LLM produced the savings.** It didn't — the
  supervisor did, and your own dashboard says so. Claiming otherwise is
  the fastest way to lose credibility under questioning.
- One take is fine. Clarity of the working loop beats editing polish.
- Check audio before the real take. Bad audio ruins an otherwise fine
  demo more often than bad video does.
