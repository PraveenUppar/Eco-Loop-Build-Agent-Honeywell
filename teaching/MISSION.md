# Mission

**Who:** Praveen, who vibe-coded the Eco-Loop project (a local-LLM building
energy controller) end to end but has not deliberately studied the concepts
underneath it — no prior background in AI/ML or in building control.

**Why:** An interview is coming up about this project. The goal is not to
memorize talking points but to actually understand, from first principles, why
the system is built the way it is — well enough to answer follow-up questions
that go off-script.

**Scope, in the learner's own words:**
1. The problem the project solves
2. The approach taken
3. The workflow (how it runs, end to end)
4. The system design and architecture
5. The decisions made, and why
6. What every folder and file contains and does
7. The results and outputs, and what they mean
8. The packages and tools used, and why
9. Current drawbacks and possible improvements
10. What ARCHITECTURE.md, SUBMISSION.md and the other docs are for
11. Baseline vs. rule-based vs. LLM approaches, compared
12. A live grill: 10 interview questions, answered out loud, with feedback

**Assumed background:** none. No AI/ML, no prior EnergyPlus/BMS knowledge.
Every lesson must build from first principles — define a term the first time
it is used.

**Format constraint:** the learner wants this broken into discrete steps/parts
with diagrams and workflow visuals, not one long wall of text.

**Session shape:** single sitting, front-loaded scope (unusual for this
skill's normal one-lesson-at-a-time pacing) — the learner already knows what
they want covered, so lessons 1-9 below were generated as a full set rather
than one at a time. Item 12 (the grill) stays a live, turn-by-turn
conversation, not a static lesson.

## Curriculum map

| # | Lesson file | Covers |
|---|---|---|
| 1 | `lessons/0001-the-problem.html` | Why building HVAC control is a real problem; what a setpoint/deadband is |
| 2 | `lessons/0002-the-approach.html` | Supervisory LLM + deterministic inner loop, and why |
| 3 | `lessons/0003-the-workflow.html` | End-to-end run sequence, macro and one-decision-cycle |
| 4 | `lessons/0004-system-design-architecture.html` | Processes, data bus, safety layers |
| 5 | `lessons/0005-key-decisions.html` | Every non-obvious design choice and its reason |
| 6 | `lessons/0006-repo-map.html` | Every folder/file, plus what the .md docs are for |
| 7 | `lessons/0007-results-and-outputs.html` | The numbers, what they mean, honest caveats |
| 8 | `lessons/0008-packages-and-tools.html` | Every dependency, EnergyPlus, Ollama, MCP |
| 9 | `lessons/0009-drawbacks-and-improvements.html` | Known limitations and the fix roadmap |
| 10 | `lessons/0010-three-controllers-compared.html` | Baseline vs rule-based vs LLM, side by side |
| — | `reference/glossary.html` | Every term used across the lessons, one place |
| 12 | live conversation | Interview grill, 10 questions |

Mission may be revisited if a follow-up study session targets something
narrower (e.g. just the MCP layer, or just prompt design).
