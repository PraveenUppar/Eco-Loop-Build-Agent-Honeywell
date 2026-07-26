"""Turn the stock 5ZoneAirCooled example into a fast, instrumented model.

Three changes, all reproducible from the untouched ExampleFiles copy:
  1. trim the RunPeriod from a full year to a single summer week
  2. strip the ~40 hourly Output:Variable objects the example ships with
  3. add back only the timestep-resolution outputs the control loop needs

Run:  python src/prepare_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppy.modeleditor import IDF

import config as cfg

# Variables we need at every timestep. "*" means all applicable keys.
TIMESTEP_VARIABLES = [
    ("*", "Zone Mean Air Temperature"),
    ("*", "Zone Air Relative Humidity"),
    ("*", "Zone Thermostat Heating Setpoint Temperature"),
    ("*", "Zone Thermostat Cooling Setpoint Temperature"),
    ("*", "Site Outdoor Air Drybulb Temperature"),
    (cfg.OCCUPANCY_SCHEDULE, "Schedule Value"),
]

TIMESTEP_METERS = [
    "Electricity:Facility",
    "Cooling:Electricity",
    "Fans:Electricity",
]


def prepare() -> Path:
    IDF.setiddname(str(cfg.IDD_PATH))
    idf = IDF(str(cfg.BASELINE_IDF))

    # -- 1. run period -----------------------------------------------------
    rp = idf.idfobjects["RUNPERIOD"][0]
    rp.Begin_Month, rp.Begin_Day_of_Month = cfg.RUN_START
    rp.End_Month, rp.End_Day_of_Month = cfg.RUN_END
    rp.Day_of_Week_for_Start_Day = cfg.START_DAY_OF_WEEK

    idf.idfobjects["TIMESTEP"][0].Number_of_Timesteps_per_Hour = (
        cfg.TIMESTEPS_PER_HOUR
    )

    # -- 2. strip the example's reporting ---------------------------------
    removed = 0
    for key in ("OUTPUT:VARIABLE", "OUTPUT:METER",
                "OUTPUT:METER:METERFILEONLY", "OUTPUT:SQLITE"):
        for obj in list(idf.idfobjects.get(key, [])):
            idf.removeidfobject(obj)
            removed += 1

    # -- 3. add only what the control loop and dashboard consume ----------
    for key, name in TIMESTEP_VARIABLES:
        obj = idf.newidfobject("OUTPUT:VARIABLE")
        obj.Key_Value = key
        obj.Variable_Name = name
        obj.Reporting_Frequency = "Timestep"

    for meter in TIMESTEP_METERS:
        obj = idf.newidfobject("OUTPUT:METER")
        obj.Key_Name = meter
        obj.Reporting_Frequency = "Timestep"

    idf.saveas(str(cfg.SIM_IDF))

    added = len(TIMESTEP_VARIABLES) + len(TIMESTEP_METERS)
    print(f"removed {removed} example output objects, added {added} timestep outputs")
    print(f"run period {cfg.RUN_START} -> {cfg.RUN_END}, "
          f"{cfg.TIMESTEPS_PER_HOUR} timesteps/hour")
    print(f"wrote {cfg.SIM_IDF}")
    return cfg.SIM_IDF


if __name__ == "__main__":
    prepare()
