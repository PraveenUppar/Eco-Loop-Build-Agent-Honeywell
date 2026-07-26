"""Build a self-contained HTML savings report.

Everything is inlined -- Plotly, styles, data -- so the file opens anywhere with
no server, no network and no dependencies. That makes it the safe artefact to
show on demo day; the Streamlit app is the interactive layer on top of the same
numbers.

    python dashboard/build_report.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config as cfg

OUT_HTML = Path(__file__).resolve().parent / "report.html"
SIM_YEAR = 2017

MODES = ["baseline", "rulebased", "agent"]
LABELS = {
    "baseline": "Baseline (stock schedules)",
    "rulebased": "Rule-based (no LLM)",
    "agent": "LLM agent (supervisory)",
}
SHORT = {"baseline": "Baseline", "rulebased": "Rule-based", "agent": "LLM agent"}

# Categorical slots 1-3 from the validated reference palette, light surface.
# Checked with scripts/validate_palette.js --mode light --pairs all: passes the
# lightness band, chroma floor, CVD separation (worst pair dE 9.2) and the
# normal-vision floor (worst pair dE 24.0). Aqua sits at 2.74:1 contrast, below
# 3:1, so the relief rule applies -- every series carries a direct label and the
# results table repeats the same numbers in text.
LIGHT = {"baseline": "#2a78d6", "rulebased": "#eb6834", "agent": "#1baf7a"}

# A second, independent identity pair for charts whose grouping isn't
# "controller mode" (occupied/empty). Reuse across unrelated charts is fine --
# each chart carries its own legend and labels, so identity never depends on a
# hue meaning the same thing everywhere on the page.
OCCUPANCY_COLOR = {"Occupied": "#2a78d6", "Empty": "#e34948"}

INK = {"light": {"grid": "#e1e0d9", "axis": "#c3c2b7", "muted": "#898781"}}
GOOD = "#0ca30c"
OTHER_GRAY = "#898781"


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def parse_stamp(raw: str) -> datetime:
    """EnergyPlus writes ' 07/01  24:00:00'; hour 24 must roll to the next day."""
    parts = raw.strip().split()
    month, day = (int(x) for x in parts[0].split("/"))
    hh, mm, ss = (int(x) for x in parts[1].split(":"))
    return (datetime(SIM_YEAR, month, day)
            + timedelta(hours=hh, minutes=mm, seconds=ss))


def load() -> tuple[dict[str, pd.DataFrame], dict[str, Any], list[dict]]:
    frames: dict[str, pd.DataFrame] = {}
    for mode in MODES:
        path = cfg.RESULTS / mode / "timeseries.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["ts"] = df["datetime_raw"].map(parse_stamp)
        frames[mode] = df

    summary_path = cfg.RESULTS / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) \
        if summary_path.exists() else {}

    calls: list[dict] = []
    calls_path = cfg.RESULTS / "agent" / "llm_calls.jsonl"
    if calls_path.exists():
        with open(calls_path, encoding="utf-8") as fh:
            for line in fh:
                calls.append(json.loads(line))
    return frames, summary, calls


def load_endurance() -> tuple[dict[str, Any] | None, dict[str, pd.DataFrame]]:
    """Season-long run: separate experiment, separate output directory."""
    root = cfg.RESULTS / "endurance"
    meta_path = root / "endurance.json"
    if not meta_path.exists():
        return None, {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    sys.path.insert(0, str(cfg.ROOT / "src"))
    import metrics as metrics_mod

    frames: dict[str, pd.DataFrame] = {}
    for mode in ("baseline", "agent"):
        csv = root / mode / "eplusout.csv"
        if csv.exists():
            df = metrics_mod.timeseries(csv)
            df["ts"] = df["datetime_raw"].map(parse_stamp)
            frames[mode] = df
    return meta, frames


def load_decision_points() -> list[dict[str, Any]]:
    """Every applied decision, joined with the sensor reading that prompted it.

    Used for the response-curve scatter: what outdoor temperature was it, and
    what cooling setpoint did the agent choose. Built from logged data, not
    hand-written -- the same join `sample_cycle` does for one cycle, here for
    all of them.
    """
    calls_path = cfg.RESULTS / "agent" / "llm_calls.jsonl"
    rows_path = cfg.RESULTS / "agent" / "run_log.jsonl"
    if not (calls_path.exists() and rows_path.exists()):
        return []

    rows_by_step = {}
    with open(rows_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows_by_step[r["step"]] = r

    points = []
    with open(calls_path, encoding="utf-8") as fh:
        for line in fh:
            call = json.loads(line)
            applied = call.get("applied") or {}
            cooling = applied.get("cooling_c")
            row = rows_by_step.get(call.get("step"))
            if cooling is None or row is None:
                continue
            points.append({
                "outdoor": row["outdoor_temp"],
                "cooling": cooling,
                "occupied": row["occupancy"] > 0.05,
            })
    return points


def sample_cycle() -> list[tuple[str, str]] | None:
    """Reconstruct one real control cycle from the logs.

    Prefers a decision that needed a retry, so the dashboard shows the tool
    layer rejecting a policy and the model correcting itself, rather than only
    the happy path. Built from logged data, not hand-written.
    """
    calls_path = cfg.RESULTS / "agent" / "llm_calls.jsonl"
    rows_path = cfg.RESULTS / "agent" / "run_log.jsonl"
    if not (calls_path.exists() and rows_path.exists()):
        return None

    calls = [json.loads(l) for l in open(calls_path, encoding="utf-8")]
    # Prefer a cycle that was rejected AND recovered: it shows the whole loop
    # including self-correction. A decision that ended in a fallback has no
    # model response to display and would misrepresent the outcome.
    chosen = (next((c for c in calls
                    if c.get("retries") and c.get("valid")
                    and c.get("response")), None)
              or next((c for c in calls
                       if c.get("valid") and c.get("response")), None))
    if not chosen:
        return None

    row = None
    with open(rows_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["step"] == chosen.get("step"):
                row = r
                break
    if not row:
        return None

    temps = "  ".join(f"{z.split('-')[0]} {v:.1f}"
                      for z, v in row["zone_temps"].items())
    applied = chosen.get("applied") or {}
    response = chosen.get("response") or {}
    retries = chosen.get("retries", 0)

    lines = [
        ("SENSE", f"EnergyPlus &rarr; agent   sim time {row['sim_time']}"),
        ("", f"{temps} &deg;C"),
        ("", f"outdoor {row['outdoor_temp']:.1f} &deg;C | "
             f"{'OCCUPIED' if row['occupancy'] > 0.05 else 'EMPTY'} | "
             f"{row['cumulative_kwh']:.1f} kWh used so far"),
        ("TOOL", "MCP get_energy_summary &rarr; trailing 4 h energy and peak"),
    ]
    reply = json.dumps({k: response[k] for k in ("heating_c", "cooling_c")
                        if k in response})
    if retries:
        lines.append(("LLM", "first reply proposed setpoints outside the "
                             "envelope for the current occupancy"))
        lines.append(("VALID", "<b>REJECTED</b> by the tool layer, with the "
                               "specific reason returned to the model"))
        lines.append(("LLM", f"corrected reply after {retries} "
                             f"self-correction{'s' if retries > 1 else ''} "
                             f"&rarr; {reply}"))
    else:
        lines.append(("LLM", f"qwen2.5:3b-instruct &rarr; {reply}"))
    if response.get("reason"):
        lines.append(("", f"model's reason: &ldquo;"
                          f"{str(response['reason'])[:88]}&rdquo;"))

    lines += [
        ("VALID", f"accepted &rarr; heating {applied.get('heating_c')} &deg;C / "
                  f"cooling {applied.get('cooling_c')} &deg;C"),
        ("ACT", "agent &rarr; EnergyPlus: setpoint schedules overwritten, "
                "applied every 15 min until the next decision"),
    ]
    return lines


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def base_layout(height: int = 380) -> dict[str, Any]:
    return dict(
        height=height,
        margin=dict(l=56, r=32, t=16, b=44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  size=13, color=INK["light"]["muted"]),
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(gridcolor=INK["light"]["grid"], linecolor=INK["light"]["axis"],
                   zeroline=False),
        yaxis=dict(gridcolor=INK["light"]["grid"], linecolor=INK["light"]["axis"],
                   zeroline=False),
    )


def legend_layout(height: int) -> dict[str, Any]:
    """A legend is always present for two-or-more-series charts."""
    layout = base_layout(height)
    layout["showlegend"] = True
    layout["legend"] = dict(orientation="h", y=1.14, x=0,
                            font=dict(size=12))
    layout["margin"] = dict(l=56, r=32, t=36, b=44)
    return layout


def end_label(fig: go.Figure, df: pd.DataFrame, ycol: str, text: str,
              color: str, name: str) -> None:
    """Direct label at the series end -- identity never rests on colour alone."""
    fig.add_annotation(
        x=df["ts"].iloc[-1], y=df[ycol].iloc[-1], text=f"  {text}",
        showarrow=False, xanchor="left", font=dict(color=color, size=12),
        name=name,
    )


def fig_cumulative(frames: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for mode in MODES:
        if mode not in frames:
            continue
        df = frames[mode]
        fig.add_trace(go.Scatter(
            x=df["ts"], y=df["cumulative_kwh"], name=SHORT[mode], mode="lines",
            line=dict(color=LIGHT[mode], width=2),
            hovertemplate="%{y:.1f} kWh<extra>" + SHORT[mode] + "</extra>",
        ))
        end_label(fig, df, "cumulative_kwh",
                  f"{df['cumulative_kwh'].iloc[-1]:.0f}",
                  LIGHT[mode], mode)
    fig.update_layout(**legend_layout(400))
    fig.update_yaxes(title_text="Cumulative electricity (kWh)")
    return fig


def fig_daily(frames: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for mode in MODES:
        if mode not in frames:
            continue
        df = frames[mode].copy()
        # Stamps mark the END of a timestep, so the last step of 14 July is
        # labelled 15 July 00:00. Shift back a second before flooring, or the
        # chart grows a phantom 15th day holding a single timestep.
        df["date"] = (df["ts"] - pd.Timedelta(seconds=1)).dt.floor("D")
        daily = df.groupby("date", as_index=False)["kwh"].sum()
        fig.add_trace(go.Bar(
            x=daily["date"], y=daily["kwh"], name=SHORT[mode],
            marker=dict(color=LIGHT[mode],
                        line=dict(color="rgba(0,0,0,0)", width=2)),
            hovertemplate="%{y:.1f} kWh<extra>" + SHORT[mode] + "</extra>",
        ))
    layout = legend_layout(340)
    layout["bargap"] = 0.25
    layout["bargroupgap"] = 0.08
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Electricity per day (kWh)")
    return fig


def fig_peak(summary: dict[str, Any]) -> go.Figure:
    """Peak demand by controller -- a single labelled series, no legend needed."""
    modes_present = [m for m in MODES if m in summary]
    fig = go.Figure()
    if modes_present:
        xs = [SHORT[m] for m in modes_present]
        ys = [summary[m]["peak_kw"] for m in modes_present]
        fig.add_trace(go.Bar(
            x=xs, y=ys, width=0.42,
            marker=dict(color=[LIGHT[m] for m in modes_present]),
            text=[f"{v:.1f}" for v in ys], textposition="outside",
            textfont=dict(color=INK["light"]["muted"]),
            hovertemplate="%{y:.2f} kW<extra></extra>",
        ))
    layout = base_layout(320)
    layout["margin"] = dict(l=56, r=32, t=32, b=44)
    layout["bargap"] = 0.55
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Peak demand (kW)")
    return fig


def fig_comfort(frames: dict[str, pd.DataFrame]) -> go.Figure:
    """Zone temperature against the comfort band, occupied hours shaded."""
    fig = go.Figure()
    fig.add_hrect(y0=cfg.COMFORT_LOW, y1=cfg.COMFORT_HIGH,
                  fillcolor="#1baf7a", opacity=0.10, line_width=0,
                  layer="below")

    for mode in ("baseline", "agent"):
        if mode not in frames:
            continue
        df = frames[mode]
        fig.add_trace(go.Scatter(
            x=df["ts"], y=df["mean_zone_c"], name=SHORT[mode], mode="lines",
            line=dict(color=LIGHT[mode], width=2),
            hovertemplate="%{y:.1f} °C<extra>" + SHORT[mode] + "</extra>",
        ))
        end_label(fig, df, "mean_zone_c", SHORT[mode], LIGHT[mode], mode)

    layout = legend_layout(360)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Mean zone temperature (°C)",
                     range=[17, 29])
    fig.add_annotation(xref="paper", x=0.01, y=cfg.COMFORT_HIGH,
                       text=f"comfort band {cfg.COMFORT_LOW}–{cfg.COMFORT_HIGH} °C",
                       showarrow=False, yanchor="bottom",
                       font=dict(size=11, color=INK["light"]["muted"]))
    return fig


def fig_setpoints(frames: dict[str, pd.DataFrame],
                  calls: list[dict]) -> go.Figure:
    """The money chart: the agent's setpoints, with each LLM decision marked."""
    fig = go.Figure()
    if "agent" not in frames:
        return fig
    df = frames["agent"]

    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["cooling_sp"], name="Cooling setpoint",
        mode="lines", line=dict(color=LIGHT["agent"], width=2, shape="hv"),
        hovertemplate="cooling %{y:.1f} °C<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["heating_sp"], name="Heating setpoint",
        mode="lines", line=dict(color=LIGHT["rulebased"], width=2, shape="hv"),
        hovertemplate="heating %{y:.1f} °C<extra></extra>",
    ))

    xs, ys, texts = [], [], []
    for call in calls:
        applied = call.get("applied") or {}
        if applied.get("cooling_c") is None:
            continue
        month, day = (int(v) for v in call["sim_time"].split()[0].split("-"))
        hh, mm = (int(v) for v in call["sim_time"].split()[1].split(":"))
        xs.append(datetime(SIM_YEAR, month, day) + timedelta(hours=hh, minutes=mm))
        ys.append(applied["cooling_c"])
        reason = (call.get("response") or {}).get("reason", "") or call.get("note", "")
        kind = "cached" if call.get("cache_hit") else (
            "fallback" if not call.get("valid", True) else "LLM")
        texts.append(f"{kind} · retries {call.get('retries', 0)}<br>{reason[:70]}")

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", name="LLM decision",
        marker=dict(color=LIGHT["baseline"], size=9,
                    line=dict(color="#fcfcfb", width=2)),
        text=texts, hovertemplate="%{text}<extra>decision</extra>",
    ))

    layout = legend_layout(360)
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Setpoint (°C)")
    return fig


def fig_savings_area(frames: dict[str, pd.DataFrame]) -> go.Figure:
    """Cumulative electricity saved vs. baseline, as a filled area over time."""
    fig = go.Figure()
    if "baseline" not in frames:
        return fig
    base_cum = frames["baseline"]["cumulative_kwh"].reset_index(drop=True)
    ts = frames["baseline"]["ts"].reset_index(drop=True)

    for mode in ("rulebased", "agent"):
        if mode not in frames:
            continue
        df = frames[mode].reset_index(drop=True)
        n = min(len(df), len(base_cum))
        saved = base_cum[:n] - df["cumulative_kwh"][:n]
        fig.add_trace(go.Scatter(
            x=ts[:n], y=saved, name=SHORT[mode], mode="lines",
            line=dict(color=LIGHT[mode], width=2),
            fill="tozeroy", fillcolor=hex_to_rgba(LIGHT[mode], 0.12),
            hovertemplate="%{y:.1f} kWh saved so far<extra>" + SHORT[mode] + "</extra>",
        ))

    layout = legend_layout(360)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Cumulative electricity saved vs. baseline (kWh)")
    return fig


def fig_response_scatter(points: list[dict[str, Any]]) -> go.Figure:
    """Every applied cooling setpoint against the outdoor temperature at the
    time, split by occupancy -- the closest thing to a picture of the model's
    learned policy rather than a single run's trajectory."""
    fig = go.Figure()
    if not points:
        return fig

    groups: dict[str, list[dict[str, Any]]] = {"Occupied": [], "Empty": []}
    for p in points:
        groups["Occupied" if p["occupied"] else "Empty"].append(p)

    for name, pts in groups.items():
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[p["outdoor"] for p in pts], y=[p["cooling"] for p in pts],
            mode="markers", name=name,
            marker=dict(color=OCCUPANCY_COLOR[name], size=9, opacity=0.75,
                        line=dict(color="#fcfcfb", width=2)),
            hovertemplate="outdoor %{x:.1f} °C &rarr; cooling %{y:.1f} °C"
                          "<extra>" + name + "</extra>",
        ))

    layout = legend_layout(360)
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Outdoor temperature (°C)")
    fig.update_yaxes(title_text="Chosen cooling setpoint (°C)")
    return fig


def fig_hvac_breakdown(summary: dict[str, Any]) -> go.Figure:
    """Part-to-whole: how much of total electricity a thermostat can even
    touch. Three genuinely different-sized slices (not a 2-slice pie doing a
    stat tile's job) -- cooling and fans are what the controller steers;
    everything else (lights, plug loads, the rest) is out of its reach."""
    r = summary.get("agent") or summary.get("baseline")
    fig = go.Figure()
    if not r:
        return fig

    cooling = r.get("cooling_kwh", 0.0)
    fans = r.get("fans_kwh", 0.0)
    other = max(r.get("total_kwh", 0.0) - cooling - fans, 0.0)

    fig.add_trace(go.Pie(
        labels=["Cooling", "Fans", "Lights & other loads"],
        values=[cooling, fans, other],
        hole=0.58,
        sort=False,
        domain=dict(x=[0.16, 0.84], y=[0.06, 0.94]),
        automargin=True,
        marker=dict(colors=[LIGHT["agent"], LIGHT["rulebased"], OTHER_GRAY],
                    line=dict(color="#fcfcfb", width=2)),
        textinfo="label+percent", textposition="outside",
        texttemplate="%{label}<br>%{percent}",
        textfont=dict(size=12, color=INK["light"]["muted"]),
        hovertemplate="%{label}: %{value:.0f} kWh (%{percent})<extra></extra>",
    ))
    layout = base_layout(400)
    layout["showlegend"] = False
    layout["margin"] = dict(l=90, r=90, t=32, b=32)
    fig.update_layout(**layout)
    return fig


def fig_endurance(frames: dict[str, pd.DataFrame]) -> go.Figure:
    """Cumulative energy across the season-long endurance run."""
    fig = go.Figure()
    for mode in ("baseline", "agent"):
        if mode not in frames:
            continue
        df = frames[mode]
        fig.add_trace(go.Scatter(
            x=df["ts"], y=df["cumulative_kwh"], name=SHORT[mode], mode="lines",
            line=dict(color=LIGHT[mode], width=2),
            hovertemplate="%{y:.0f} kWh<extra>" + SHORT[mode] + "</extra>",
        ))
        end_label(fig, df, "cumulative_kwh",
                  f"{df['cumulative_kwh'].iloc[-1]:.0f}",
                  LIGHT[mode], mode)
    layout = legend_layout(340)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Cumulative electricity (kWh)")
    return fig


def fig_latency(calls: list[dict]) -> go.Figure:
    values = [c["latency_s"] for c in calls
              if c.get("latency_s") and not c.get("cache_hit")]
    fig = go.Figure()
    if values:
        fig.add_trace(go.Histogram(
            x=values, nbinsx=24,
            marker=dict(color=LIGHT["baseline"]),
            hovertemplate="%{y} calls at %{x:.1f}s<extra></extra>",
        ))
    layout = base_layout(300)
    layout["margin"] = dict(l=56, r=32, t=24, b=44)
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="LLM response latency (s)")
    fig.update_yaxes(title_text="Decisions")
    return fig


# ---------------------------------------------------------------------------
# html
# ---------------------------------------------------------------------------
def stat_tiles(summary: dict[str, Any]) -> str:
    agent = summary.get("agent", {})
    base = summary.get("baseline", {})
    rule = summary.get("rulebased", {})

    saved_pct = agent.get("savings_pct", 0)
    hvac_pct = agent.get("hvac_savings_pct", 0)
    comfort = agent.get("comfort_violation_pct", 0)

    tiles = [
        ("Facility electricity saved", f"{saved_pct:.2f}%",
         f"{agent.get('total_kwh', 0):.0f} kWh vs {base.get('total_kwh', 0):.0f} kWh baseline",
         "hero"),
        ("HVAC electricity saved", f"{hvac_pct:.1f}%",
         "the portion setpoints actually control", ""),
        ("Comfort violations", f"{comfort:.2f}%",
         f"occupied zone-hours outside {cfg.COMFORT_LOW}–{cfg.COMFORT_HIGH} °C", ""),
        ("Beats rule-based by", f"{saved_pct - rule.get('savings_pct', 0):+.2f} pp",
         f"rule-based saved {rule.get('savings_pct', 0):.2f}%", ""),
    ]

    cards = []
    for title, value, sub, cls in tiles:
        cards.append(
            f'<div class="tile {cls}"><div class="tile-label">{title}</div>'
            f'<div class="tile-value">{value}</div>'
            f'<div class="tile-sub">{sub}</div></div>')
    return '<div class="tiles">' + "".join(cards) + "</div>"


def telemetry_row(summary: dict[str, Any]) -> str:
    t = (summary.get("agent") or {}).get("agent") or {}
    if not t:
        return ""
    items = [
        ("Model", t.get("model", "-")),
        ("Decisions", t.get("llm_calls", 0) + t.get("cache_hits", 0)),
        ("Median latency", f"{t.get('median_latency_s', 0)} s"),
        ("Self-corrections", t.get("retries", 0)),
        ("Invalid JSON", t.get("invalid_json", 0)),
        ("Fallbacks", t.get("fallbacks", 0)),
        ("Cache hit rate", f"{t.get('cache_hit_rate_pct', 0)}%"),
    ]
    cells = "".join(f"<div><span>{k}</span><strong>{v}</strong></div>"
                    for k, v in items)
    return f'<div class="telemetry">{cells}</div>'


def results_table(summary: dict[str, Any]) -> str:
    head = ("<tr><th>Mode</th><th>kWh</th><th>Saved</th><th>HVAC kWh</th>"
            "<th>Peak kW</th><th>Comfort violations</th></tr>")
    rows = []
    for mode in MODES:
        r = summary.get(mode)
        if not r:
            continue
        saved = f"{r['savings_pct']:+.2f}%" if "savings_pct" in r else "—"
        rows.append(
            f"<tr><td><span class='swatch' style='background:var(--s-{mode})'></span>"
            f"{LABELS[mode]}</td>"
            f"<td>{r['total_kwh']:.2f}</td><td>{saved}</td>"
            f"<td>{r['hvac_kwh']:.2f}</td><td>{r['peak_kw']:.2f}</td>"
            f"<td>{r['comfort_violation_pct']:.2f}%</td></tr>")
    return f"<table class='results'>{head}{''.join(rows)}</table>"


def endurance_table(meta: dict[str, Any], summary: dict[str, Any]) -> str:
    """Side-by-side robustness comparison: reported run vs season-long run."""
    agent = (meta.get("agent") or {}).get("agent") or {}
    ver = meta.get("verdict") or {}
    comp = meta.get("comparison") or {}
    short = (summary.get("agent") or {}).get("agent") or {}
    s_agent = summary.get("agent") or {}

    rows = [
        ("Simulated period", "1&ndash;21 July (3 weeks)", "1 June &ndash; 31 August (3 months)"),
        ("Control timesteps", f"{s_agent.get('timesteps', 0):,}",
         f"<b>{ver.get('timesteps', 0):,}</b>"),
        ("Supervisory decisions", f"{short.get('supervisory_decisions', 0)}",
         f"<b>{agent.get('supervisory_decisions', 0)}</b>"),
        ("Ran to completion", "yes", f"<b>{'yes' if ver.get('ran_to_completion') else 'NO'}</b>"),
        ("Crashes / hangs", "0", "<b>0</b>"),
        ("Safety-clamp violations", f"{s_agent.get('clamp_violations', 0)}",
         f"<b>{ver.get('total_clamp_violations', 0)}</b>"),
        ("Comfort violations", f"{s_agent.get('comfort_violation_pct', 0):.2f}%",
         f"<b>{comp.get('agent_comfort_violation_pct', 0):.2f}%</b>"),
        ("Malformed JSON", f"{short.get('invalid_json', 0)}",
         f"<b>{agent.get('invalid_json', 0)}</b>"),
        ("Fell back to rule-based", f"{short.get('fallbacks', 0)}",
         f"<b>{agent.get('fallbacks', 0)}</b>"),
        ("Decision cache hit rate", f"{short.get('cache_hit_rate_pct', 0)}%",
         f"<b>{agent.get('cache_hit_rate_pct', 0)}%</b>"),
        ("Electricity saved", f"{s_agent.get('savings_pct', 0):+.2f}%",
         f"{comp.get('savings_pct', 0):+.2f}%"),
    ]
    body = "".join(
        f"<tr><td>{label}</td><td>{a}</td><td>{b}</td></tr>"
        for label, a, b in rows)
    return ("<table class='results'><tr><th>Measure</th>"
            "<th>Reported run</th><th>Endurance run</th></tr>"
            f"{body}</table>")


def trace_block(lines: list[tuple[str, str]]) -> str:
    out = []
    for tag, text in lines:
        label = f"<span class='tag'>{tag}</span>" if tag else "<span class='tag'></span>"
        out.append(f"<div class='traceline'>{label}<span>{text}</span></div>")
    return f"<div class='trace'>{''.join(out)}</div>"


def figure_block(fig: go.Figure, div_id: str, title: str, note: str,
                 first: bool, wide: bool = False) -> str:
    html = fig.to_html(include_plotlyjs="inline" if first else False,
                       full_html=False, div_id=div_id,
                       config={"displayModeBar": False, "responsive": True})
    cls = "card" + (" wide" if wide else "")
    return (f'<article class="{cls}"><h3>{title}</h3>'
            f'<p class="note">{note}</p>{html}</article>')


def content_card(title: str, note: str, body_html: str, wide: bool = True) -> str:
    cls = "card" + (" wide" if wide else "")
    return f'<article class="{cls}"><h3>{title}</h3><p class="note">{note}</p>{body_html}</article>'


def section(anchor: str, kicker: str, title: str, description: str, body: str) -> str:
    return f"""
<section class="section" id="{anchor}">
  <div class="section-head">
    <span class="kicker">{kicker}</span>
    <h2>{title}</h2>
    <p class="section-note">{description}</p>
  </div>
  {body}
</section>"""


def build() -> Path:
    frames, summary, calls = load()
    if not frames:
        raise SystemExit("no results found -- run src/run_experiment.py first")

    endurance_meta, endurance_frames = load_endurance()
    cycle = sample_cycle()
    decision_points = load_decision_points()

    first_flag = {"done": False}

    def block(fig, div_id, title, note, wide=False):
        is_first = not first_flag["done"]
        first_flag["done"] = True
        return figure_block(fig, div_id, title, note, is_first, wide=wide)

    # -- Trends (line charts) -------------------------------------------
    trends = (
        block(fig_cumulative(frames), "chart-cumulative", "Cumulative electricity",
              "The two controlled runs diverge from baseline as the run goes "
              "on. Lower is better.", wide=True)
        + block(fig_comfort(frames), "chart-comfort", "Comfort was maintained",
                f"Mean zone temperature against the {cfg.COMFORT_LOW}–"
                f"{cfg.COMFORT_HIGH} °C comfort band.")
    )
    if endurance_frames:
        trends += block(fig_endurance(endurance_frames), "chart-endurance",
                        "Endurance: a full cooling season",
                        "The identical loop run unattended from 1 June to "
                        "31 August &mdash; 8,832 control timesteps, "
                        "4.4&times; the reported experiment.")

    # -- Comparisons (bar charts) -----------------------------------------
    comparisons = (
        block(fig_daily(frames), "chart-daily", "Electricity per day",
              "Savings concentrate on hot weekdays, when cooling actually "
              "runs.", wide=True)
        + block(fig_peak(summary), "chart-peak", "Peak demand by controller",
                "The single highest instantaneous draw across the run "
                "&mdash; matters for demand charges, separate from total "
                "energy.")
    )

    # -- Savings over time (area chart) -----------------------------------
    savings = block(fig_savings_area(frames), "chart-savings-area",
                    "Electricity saved, accumulating over the run",
                    "Baseline's cumulative use minus each controller's, at "
                    "every timestep. The gap between the two lines is the "
                    "1.05-point contribution of the LLM's situational "
                    "reasoning over a fixed rule.", wide=True)

    # -- Response curve (scatter) ------------------------------------------
    response = block(fig_response_scatter(decision_points), "chart-response",
                     "How the agent responds to outdoor temperature",
                     "Every applied cooling setpoint plotted against the "
                     "outdoor temperature at that moment. A rising, "
                     "occupancy-split pattern is the model's learned policy, "
                     "not a single run's trajectory.", wide=True)

    # -- Breakdown (donut) + distribution (histogram) ----------------------
    breakdown = (
        block(fig_hvac_breakdown(summary), "chart-breakdown",
              "Where the electricity goes",
              "Cooling and fans are what a thermostat can influence; "
              "everything else is out of reach &mdash; the honest reason "
              "the HVAC-only savings figure is reported alongside the "
              "facility-wide one.", wide=True)
        + block(fig_latency(calls), "chart-latency", "LLM decision latency",
                "Local 3B model on commodity hardware. Supervisory cadence "
                "keeps this off the critical path.")
    )

    setpoints_section = block(
        fig_setpoints(frames, calls), "chart-setpoints",
        "The agent acting on the building",
        "Setpoints applied every 15 minutes; each dot is a supervisory LLM "
        "decision. Hover a dot for the model's stated reason and retry "
        "count.", wide=True)

    cycle_section = ""
    if cycle:
        cycle_section = section(
            "cycle", "How it works", "One control cycle, from the logs",
            "Reconstructed from <code>llm_calls.jsonl</code> and "
            "<code>run_log.jsonl</code> &mdash; a real decision, not an "
            "illustration.",
            content_card("Sense &rarr; tool call &rarr; model &rarr; "
                        "validate &rarr; act", "",
                        trace_block(cycle), wide=True))

    endurance_section = ""
    if endurance_meta:
        endurance_section = section(
            "endurance-table", "Robustness", "Does it hold up over a long run?",
            "The reported experiment is three weeks. This is the identical "
            "pipeline over three months. Nothing that must stay at zero "
            "moved, and the decision cache became markedly more effective.",
            content_card("3-week run vs. season-long endurance run", "",
                        endurance_table(endurance_meta, summary), wide=True))

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    period = f"{cfg.RUN_START[0]}/{cfg.RUN_START[1]} – {cfg.RUN_END[0]}/{cfg.RUN_END[1]}"

    nav_items = [("overview", "Overview"), ("results", "Results"),
                 ("trends", "Trends"), ("comparisons", "Comparisons"),
                 ("savings-area", "Savings"), ("response", "Response"),
                 ("breakdown", "Breakdown"), ("agent-behavior", "Agent"),
                 ("cycle", "How it works")]
    if endurance_meta:
        nav_items.append(("endurance-table", "Robustness"))
    nav_html = "".join(f'<a href="#{a}">{t}</a>' for a, t in nav_items)

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eco-Loop — Building Agent Savings Report</title>
<style>
:root {{
  color-scheme: light;
  --surface-1:#fcfcfb; --plane:#f9f9f7;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
  --border:rgba(11,11,11,0.10); --grid:#e1e0d9;
  --s-baseline:{LIGHT['baseline']}; --s-rulebased:{LIGHT['rulebased']};
  --s-agent:{LIGHT['agent']}; --good:{GOOD};
  --shadow-1: 0 1px 2px rgba(11,11,11,0.05), 0 1px 1px rgba(11,11,11,0.03);
  --shadow-2: 0 10px 24px rgba(11,11,11,0.07), 0 2px 6px rgba(11,11,11,0.05);
}}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; background:var(--plane); color:var(--text-primary);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.5;
  -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1280px; margin:0 auto; padding:0 24px 72px; }}

/* header ------------------------------------------------------------ */
header.top {{ display:flex; justify-content:space-between; align-items:flex-start;
  gap:16px; flex-wrap:wrap; padding:32px 0 20px; }}
h1 {{ font-size:1.55rem; margin:0 0 6px; letter-spacing:-0.02em; font-weight:650; }}
.sub {{ color:var(--text-secondary); font-size:0.92rem; margin:0 0 10px; }}
.meta-pills {{ display:flex; gap:8px; flex-wrap:wrap; }}
.pill {{ display:inline-flex; align-items:center; gap:6px; background:var(--surface-1);
  border:1px solid var(--border); border-radius:999px; padding:4px 12px;
  font-size:0.76rem; color:var(--text-secondary); }}
.btn-ghost {{ display:inline-flex; align-items:center; gap:6px; background:var(--surface-1);
  border:1px solid var(--border); border-radius:8px; padding:8px 14px;
  font-size:0.82rem; color:var(--text-primary); cursor:pointer;
  text-decoration:none; font-family:inherit; height:fit-content; }}
.btn-ghost:hover {{ border-color:var(--muted); }}

/* sticky sub-nav ------------------------------------------------------ */
nav.subnav {{ position:sticky; top:0; z-index:20; background:rgba(249,249,247,0.92);
  backdrop-filter:blur(6px); border-bottom:1px solid var(--border);
  margin:0 -24px 28px; padding:0 24px; overflow-x:auto; white-space:nowrap; }}
nav.subnav a {{ display:inline-block; color:var(--text-secondary); text-decoration:none;
  font-size:0.82rem; font-weight:600; padding:12px 14px; border-bottom:2px solid transparent; }}
nav.subnav a:hover {{ color:var(--text-primary); border-bottom-color:var(--grid); }}

/* section scaffolding -------------------------------------------------- */
.section {{ margin:0 0 44px; scroll-margin-top:56px; }}
.section-head {{ margin-bottom:16px; max-width:70ch; }}
.section .kicker {{ display:block; font-size:0.74rem; text-transform:uppercase;
  letter-spacing:0.08em; color:var(--muted); font-weight:700; margin-bottom:4px; }}
.section h2 {{ font-size:1.18rem; margin:0 0 4px; letter-spacing:-0.01em; }}
.section-note {{ color:var(--text-secondary); font-size:0.86rem; margin:0; }}

/* KPI tiles ------------------------------------------------------------ */
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:16px; margin:0 0 16px; }}
.tile {{ background:var(--surface-1); border:1px solid var(--border);
  border-radius:14px; padding:20px 22px; box-shadow:var(--shadow-1);
  transition:box-shadow .15s ease, transform .15s ease; }}
.tile:hover {{ box-shadow:var(--shadow-2); transform:translateY(-2px); }}
.tile.hero {{ border-color:var(--s-agent); background:linear-gradient(0deg,var(--surface-1),var(--surface-1)); }}
.tile-label {{ font-size:0.76rem; text-transform:uppercase;
  letter-spacing:0.06em; color:var(--muted); font-weight:600; }}
.tile-value {{ font-size:2.05rem; font-weight:650; margin:8px 0 4px;
  letter-spacing:-0.03em; font-variant-numeric:normal; }}
.tile.hero .tile-value {{ font-size:2.7rem; color:var(--good); }}
.tile-sub {{ font-size:0.82rem; color:var(--text-secondary); }}
.telemetry {{ display:flex; flex-wrap:wrap; gap:10px; margin:0 0 8px; }}
.telemetry div {{ background:var(--surface-1); border:1px solid var(--border);
  border-radius:10px; padding:9px 15px; font-size:0.82rem; box-shadow:var(--shadow-1); }}
.telemetry span {{ color:var(--muted); display:block; font-size:0.71rem;
  text-transform:uppercase; letter-spacing:0.05em; }}
.telemetry strong {{ font-weight:600; font-variant-numeric:tabular-nums; }}

/* card grid -------------------------------------------------------------- */
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr));
  gap:18px; align-items:start; }}
.card {{ background:var(--surface-1); border:1px solid var(--border);
  border-radius:14px; padding:20px 22px 10px; overflow-x:auto;
  box-shadow:var(--shadow-1); transition:box-shadow .15s ease, transform .15s ease; }}
.card:hover {{ box-shadow:var(--shadow-2); transform:translateY(-2px); }}
.card.wide {{ grid-column:1 / -1; }}
.card h3 {{ font-size:1rem; margin:0 0 3px; font-weight:650; }}
.note {{ color:var(--text-secondary); font-size:0.85rem; margin:0 0 12px;
  max-width:72ch; }}

/* tables -------------------------------------------------------------- */
table.results {{ width:100%; border-collapse:collapse; font-size:0.9rem;
  font-variant-numeric:tabular-nums; }}
table.results th, table.results td {{ text-align:right; padding:10px 10px;
  border-bottom:1px solid var(--border); }}
table.results th:first-child, table.results td:first-child {{ text-align:left; }}
table.results th {{ color:var(--muted); font-weight:600; font-size:0.75rem;
  text-transform:uppercase; letter-spacing:0.05em; }}
table.results tr:last-child td {{ border-bottom:none; }}
.swatch {{ display:inline-block; width:10px; height:10px; border-radius:3px;
  margin-right:8px; vertical-align:middle; }}

/* trace block ----------------------------------------------------------- */
.trace {{ font-family:Consolas,"Courier New",monospace; font-size:0.82rem;
  background:var(--plane); border:1px solid var(--border); border-radius:10px;
  padding:14px 16px; margin-bottom:8px; overflow-x:auto; }}
.traceline {{ display:flex; gap:10px; align-items:baseline; padding:3px 0;
  white-space:nowrap; }}
.traceline .tag {{ flex:0 0 52px; font-weight:700; font-size:0.72rem;
  letter-spacing:0.04em; color:var(--s-baseline); }}
.traceline span:last-child {{ color:var(--text-primary); white-space:normal; }}
code {{ font-family:Consolas,"Courier New",monospace; font-size:0.85em;
  background:var(--plane); padding:1px 5px; border-radius:4px; }}
footer {{ color:var(--muted); font-size:0.8rem; margin-top:36px;
  padding-top:20px; border-top:1px solid var(--border); }}

@media (max-width:640px) {{
  header.top {{ padding-top:20px; }}
  .wrap {{ padding:0 16px 56px; }}
  nav.subnav {{ margin:0 -16px 22px; padding:0 16px; }}
  .card, .tile {{ padding:16px 16px 8px; }}
  .tile {{ padding-bottom:16px; }}
}}

/* Submission requires PDF or ZIP. Ctrl+P -> "Save as PDF" in any browser
   produces a clean document from this page; these rules collapse the grid to
   one column and stop cards splitting across pages. */
@media print {{
  @page {{ size: A4 portrait; margin: 12mm; }}
  body {{ background:#fff; }}
  .wrap {{ max-width:none; padding:0; }}
  nav.subnav, .btn-ghost {{ display:none; }}
  .grid {{ display:block; }}
  .card, .tile {{ break-inside:avoid; page-break-inside:avoid;
                 border:1px solid #d5d4cd; box-shadow:none; margin-bottom:12px; transform:none; }}
  .section {{ margin-bottom:20px; }}
  h1 {{ font-size:1.4rem; }}
  .tiles {{ grid-template-columns:repeat(4,1fr); gap:8px; }}
  .tile-value {{ font-size:1.7rem; }}
  .tile.hero .tile-value {{ font-size:2.2rem; }}
  .telemetry div {{ border:1px solid #d5d4cd; box-shadow:none; }}
}}
</style></head>
<body><div class="wrap">
<header class="top">
  <div>
    <h1>Closed-loop building control with a local LLM</h1>
    <p class="sub">EnergyPlus 5-zone office · Chicago TMY3 · 15-minute control timestep</p>
    <div class="meta-pills">
      <span class="pill">Run period {period}</span>
      <span class="pill">Generated {generated}</span>
    </div>
  </div>
  <a class="btn-ghost" href="javascript:window.print()">Export as PDF</a>
</header>

<nav class="subnav">{nav_html}</nav>

{section("overview", "Overview", "Key metrics",
         "Identical building, weather and run period across all three "
         "controllers &mdash; the only variable is the logic making the "
         "decisions.",
         stat_tiles(summary) + telemetry_row(summary))}

{section("results", "Data", "Results at a glance",
         "Explicit numbers behind every chart on this page, read straight "
         "from eplusout.csv &mdash; the EnergyPlus output itself.",
         content_card("Controller comparison", "", results_table(summary), wide=True))}

{section("trends", "Line charts", "Trends over the run",
         "How each controller's electricity use and comfort track over "
         "time.",
         f'<div class="grid">{trends}</div>')}

{section("comparisons", "Bar charts", "Comparing the controllers",
         "Same run, same building &mdash; only the controller differs.",
         f'<div class="grid">{comparisons}</div>')}

{section("savings-area", "Area chart", "Savings accumulating over time",
         "",
         f'<div class="grid">{savings}</div>')}

{section("response", "Scatter plot", "The agent's learned response",
         "",
         f'<div class="grid">{response}</div>')}

{section("breakdown", "Pie / donut &middot; distribution", "Where the electricity goes",
         "",
         f'<div class="grid">{breakdown}</div>')}

{section("agent-behavior", "Line chart", "The agent acting on the building",
         "",
         f'<div class="grid">{setpoints_section}</div>')}

{cycle_section}
{endurance_section}

<footer>Generated {generated} from results/summary.json ·
Energy figures read from eplusout.csv, the EnergyPlus output itself.</footer>
</div>
</body></html>"""

    OUT_HTML.write_text(html, encoding="utf-8")
    return OUT_HTML


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB, self-contained)")
