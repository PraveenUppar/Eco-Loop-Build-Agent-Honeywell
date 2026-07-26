# Resources

External, high-trust material to ground the lessons. Project-internal claims
are cited to the actual source file instead (README.md, ARCHITECTURE.md,
SUBMISSION.md, and src/*.py) since that is the highest-trust source for
project-specific behavior — these external resources are for the general
concepts underneath it.

## Building control / HVAC fundamentals
- EnergyPlus documentation — <https://energyplus.net/documentation> — the
  simulation engine this whole project sits on top of. Input/Output Reference
  explains `Schedule:Compact`, `RunPeriod`, meters, and the EMS/Python API
  hooks used by `src/eplus_runner.py`.
- EnergyPlus Python API (`pyenergyplus`) guide — bundled with the EnergyPlus
  install at `<ENERGYPLUS_DIR>/ExampleFiles` and the PythonPlugins docs —
  covers the callback model (`callback_begin_system_timestep_before_predictor`
  etc.) used directly in `src/eplus_runner.py`.
- ASHRAE 55 (thermal comfort) — background for why "comfort" in this project
  is a temperature band rather than a full PMV/Fanger model (a stated,
  deliberate simplification — see lesson 9).

## LLMs, agents, tool calling
- Ollama documentation — <https://ollama.com> — local model serving; the
  `format` (JSON schema) parameter used in `src/agent.py` is documented under
  Ollama's structured-outputs guide.
- Model Context Protocol (MCP) spec — <https://modelcontextprotocol.io> — the
  protocol behind `src/mcp_server.py` / `src/mcp_bridge.py`. Read the
  "Architecture" and "Tools" pages first; that's the whole surface this
  project uses.
- Qwen2.5 model card (Alibaba/Qwen team) — background on the `qwen2.5:3b-instruct`
  model this project runs locally, its size class, and instruction-tuning.

## Software patterns used in this codebase
- Python `asyncio` docs, specifically `run_coroutine_threadsafe` — the
  exact mechanism `src/mcp_bridge.py` uses to call an async MCP session from a
  synchronous EnergyPlus callback.
- `os.replace` atomicity — Python docs note it is atomic on POSIX and (as of
  recent Windows/Python versions) on Windows via `MoveFileEx` — relevant to
  the `WinError 5` handling in `src/shared_state.py`.

## Where to go for wisdom (real-world practice)
- r/ControlTheory and r/homeautomation (Reddit) for building-control practitioner
  discussion — useful once the fundamentals are solid, to see how real BMS
  supervisory logic compares.
- MCP's own GitHub Discussions — active community for the protocol layer if
  deeper agentic-tooling questions come up in the interview.

*(Populate further as specific gaps show up in the grill session — if a
question exposes a shaky spot, add the resource that would have filled it.)*
