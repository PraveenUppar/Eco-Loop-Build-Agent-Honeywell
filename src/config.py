"""Central configuration for the Eco-Loop building agent.

Everything that the runner, MCP server, agent and dashboard need to agree on
lives here so there is exactly one source of truth for zone names, schedule
names and safety limits.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
OUT = ROOT / "out"

ENERGYPLUS_DIR = Path(r"C:\EnergyPlusV26-1-0")
IDD_PATH = ENERGYPLUS_DIR / "Energy+.idd"

BASELINE_IDF = MODELS / "baseline.idf"          # untouched ExampleFiles copy
SIM_IDF = MODELS / "simulation.idf"             # trimmed, instrumented model
AGENT_IDF = MODELS / "agent_optimized.idf"      # written after an agent run
WEATHER = MODELS / "weather.epw"

# --------------------------------------------------------------------------
# Simulation window
# --------------------------------------------------------------------------
# A summer week in Chicago: cooling-dominated, which is where the tight
# baseline deadband leaves the most savings headroom.
RUN_START = (7, 1)      # (month, day)
RUN_END = (7, 21)       # three weeks: 15 weekdays + 6 weekend days, 2016 steps
START_DAY_OF_WEEK = "Monday"
TIMESTEPS_PER_HOUR = 4  # 15-minute control timestep

# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------
# The five conditioned zones. PLENUM-1 is an unconditioned return plenum and is
# deliberately excluded from control and from the comfort metric.
ZONES = ["SPACE1-1", "SPACE2-1", "SPACE3-1", "SPACE4-1", "SPACE5-1"]
PLENUM = "PLENUM-1"

# Schedule:Compact objects that the thermostats read. These are the actuation
# targets -- writing them overrides the schedule for that timestep.
HEATING_SCHEDULE = "Htg-SetP-Sch"
COOLING_SCHEDULE = "Clg-SetP-Sch"
OCCUPANCY_SCHEDULE = "OCCUPY-1"

# --------------------------------------------------------------------------
# Meters
# --------------------------------------------------------------------------
# The EnergyPlus API will not return a handle for the "Electricity:Facility"
# aggregate meter even though it appears in list_available_api_data_csv. The
# three sector meters below ARE resolvable and sum to exactly the facility
# total (verified against eplusout.csv: 953.04 kWh over the run week).
FACILITY_METERS = [
    "Electricity:Building",
    "Electricity:HVAC",
    "Electricity:Plant",
]

# Setpoint-controllable end uses. This is the signal the supervisor actually
# steers -- lights and plug loads are not affected by thermostat policy.
HVAC_METERS = [
    "Cooling:Electricity",
    "Fans:Electricity",
    "Heating:Electricity",
    "Pumps:Electricity",
]

# Authoritative column in eplusout.csv for reporting metrics.
FACILITY_CSV_COLUMN = "Electricity:Facility [J](TimeStep)"

# --------------------------------------------------------------------------
# Safety envelope -- hard clamps applied to every policy, whatever its source
# --------------------------------------------------------------------------
HEATING_MIN, HEATING_MAX = 18.0, 22.0
COOLING_MIN, COOLING_MAX = 23.0, 27.0
MIN_DEADBAND = 1.5          # cooling_sp - heating_sp must be >= this

# Comfort band used for the reported comfort metric (occupied hours only).
COMFORT_LOW, COMFORT_HIGH = 20.0, 25.0

# Setback envelope for an empty building. With zones floating around 20-22 C
# overnight, heating at or below 19 and cooling at or above 26 guarantees no
# equipment runs at all. This is the economic mirror of the comfort band: one
# stops the supervisor making occupants uncomfortable, the other stops it
# running plant in an empty building.
SETBACK_HEATING_MAX = 19.0
SETBACK_COOLING_MIN = 26.0

# Occupied hours, local time. Matches OCCUPY-1 in the model (weekdays 6-20).
OCCUPIED_START_HOUR = 6
OCCUPIED_END_HOUR = 20

# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
OLLAMA_MODEL = "qwen2.5:3b-instruct"
LLM_INTERVAL_HOURS = 4      # supervisor re-plans every 4 simulated hours
LLM_TIMEOUT_S = 60
LLM_MAX_RETRIES = 2         # self-correction attempts after an invalid reply

# Baseline setpoints, used as the safe fallback whenever the LLM cannot
# produce a valid policy.
FALLBACK_HEATING = 21.0
FALLBACK_COOLING = 24.0

J_TO_KWH = 1.0 / 3.6e6
