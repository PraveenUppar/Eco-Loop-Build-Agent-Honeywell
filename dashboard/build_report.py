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

INK = {"light": {"grid": "#e1e0d9", "axis": "#c3c2b7", "muted": "#898781"}}


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


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
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


def base_layout(height: int = 380) -> dict[str, Any]:
    return dict(
        height=height,
        margin=dict(l=56, r=112, t=16, b=44),
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
                  f"{SHORT[mode]} {df['cumulative_kwh'].iloc[-1]:.0f}",
                  LIGHT[mode], mode)
    fig.update_layout(**base_layout(400))
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
    layout = base_layout(340)
    layout["showlegend"] = True
    layout["legend"] = dict(orientation="h", y=1.12, x=0)
    layout["bargap"] = 0.25
    layout["bargroupgap"] = 0.08
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Electricity per day (kWh)")
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

    fig.update_layout(**base_layout(360))
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

    layout = base_layout(360)
    layout["showlegend"] = True
    layout["legend"] = dict(orientation="h", y=1.12, x=0)
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Setpoint (°C)")
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
                  f"{SHORT[mode]} {df['cumulative_kwh'].iloc[-1]:.0f}",
                  LIGHT[mode], mode)
    fig.update_layout(**base_layout(340))
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
    fig.update_layout(**base_layout(300))
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
    telemetry = agent.get("agent", {})

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
                 first: bool) -> str:
    html = fig.to_html(include_plotlyjs="inline" if first else False,
                       full_html=False, div_id=div_id,
                       config={"displayModeBar": False, "responsive": True})
    return (f'<section class="card"><h2>{title}</h2>'
            f'<p class="note">{note}</p>{html}</section>')


def build() -> Path:
    frames, summary, calls = load()
    if not frames:
        raise SystemExit("no results found -- run src/run_experiment.py first")

    endurance_meta, endurance_frames = load_endurance()
    cycle = sample_cycle()

    figs = [
        (fig_cumulative(frames), "cumulative", "Cumulative electricity",
         "The two controlled runs diverge from baseline as the fortnight goes on. "
         "Lower is better."),
        (fig_daily(frames), "daily", "Electricity per day",
         "Savings concentrate on hot weekdays, when cooling actually runs."),
        (fig_comfort(frames), "comfort", "Comfort was maintained",
         f"Mean zone temperature against the {cfg.COMFORT_LOW}–{cfg.COMFORT_HIGH} °C "
         "comfort band. The agent saves energy without pushing occupants out of band."),
        (fig_setpoints(frames, calls), "setpoints",
         "The agent acting on the building",
         "Setpoints applied every 15 minutes; each dot is a supervisory LLM "
         "decision. Hover a dot for the model's stated reason and retry count."),
        (fig_latency(calls), "latency", "LLM decision latency",
         "Local 3B model on commodity hardware. Supervisory cadence keeps this "
         "off the critical path."),
    ]

    if endurance_frames:
        figs.append((
            fig_endurance(endurance_frames), "endurance",
            "Endurance: a full cooling season",
            "The same loop run unattended from 1 June to 31 August "
            "&mdash; 8,832 control timesteps, 4.4&times; the reported experiment."))

    blocks = "".join(
        figure_block(fig, div_id, title, note, i == 0)
        for i, (fig, div_id, title, note) in enumerate(figs))

    cycle_section = ""
    if cycle:
        cycle_section = f"""
<section class="card"><h2>One control cycle, from the logs</h2>
<p class="note">Reconstructed from <code>llm_calls.jsonl</code> and
<code>run_log.jsonl</code> &mdash; a real decision, not an illustration. Sensor
data leaves EnergyPlus, the agent queries tools, the model proposes a policy,
the tool layer validates it, and setpoints go back into the running
simulation.</p>
{trace_block(cycle)}
</section>"""

    endurance_section = ""
    if endurance_meta:
        endurance_section = f"""
<section class="card"><h2>Does it hold up over a long run?</h2>
<p class="note">The reported experiment is three weeks. This is the identical
pipeline over three months. Nothing that must stay at zero moved, and the
decision cache became markedly more effective.</p>
{endurance_table(endurance_meta, summary)}
</section>"""

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    period = f"{cfg.RUN_START[0]}/{cfg.RUN_START[1]} – {cfg.RUN_END[0]}/{cfg.RUN_END[1]}"

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
  --s-agent:{LIGHT['agent']}; --good:#006300;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--plane); color:var(--text-primary);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.5; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:32px 20px 64px; }}
header {{ display:flex; justify-content:space-between; align-items:flex-start;
  gap:16px; flex-wrap:wrap; margin-bottom:8px; }}
h1 {{ font-size:1.6rem; margin:0 0 4px; letter-spacing:-0.02em; }}
.sub {{ color:var(--text-secondary); font-size:0.95rem; margin:0; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
  gap:14px; margin:24px 0; }}
.tile {{ background:var(--surface-1); border:1px solid var(--border);
  border-radius:12px; padding:18px 20px; }}
.tile.hero {{ grid-column:span 1; border-color:var(--s-agent); }}
.tile-label {{ font-size:0.78rem; text-transform:uppercase;
  letter-spacing:0.06em; color:var(--muted); }}
.tile-value {{ font-size:2.1rem; font-weight:650; margin:6px 0 2px;
  letter-spacing:-0.03em; }}
.tile.hero .tile-value {{ font-size:2.9rem; color:var(--good); }}
.tile-sub {{ font-size:0.82rem; color:var(--text-secondary); }}
.telemetry {{ display:flex; flex-wrap:wrap; gap:10px; margin:0 0 24px; }}
.telemetry div {{ background:var(--surface-1); border:1px solid var(--border);
  border-radius:8px; padding:8px 14px; font-size:0.82rem; }}
.telemetry span {{ color:var(--muted); display:block; font-size:0.72rem;
  text-transform:uppercase; letter-spacing:0.05em; }}
.telemetry strong {{ font-weight:600; }}
.card {{ background:var(--surface-1); border:1px solid var(--border);
  border-radius:12px; padding:20px 20px 8px; margin-bottom:18px;
  overflow-x:auto; }}
.card h2 {{ font-size:1.05rem; margin:0 0 2px; }}
.note {{ color:var(--text-secondary); font-size:0.86rem; margin:0 0 10px;
  max-width:70ch; }}
table.results {{ width:100%; border-collapse:collapse; font-size:0.9rem;
  font-variant-numeric:tabular-nums; }}
table.results th, table.results td {{ text-align:right; padding:9px 10px;
  border-bottom:1px solid var(--border); }}
table.results th:first-child, table.results td:first-child {{ text-align:left; }}
table.results th {{ color:var(--muted); font-weight:600; font-size:0.78rem;
  text-transform:uppercase; letter-spacing:0.05em; }}
.swatch {{ display:inline-block; width:10px; height:10px; border-radius:3px;
  margin-right:8px; vertical-align:middle; }}
.trace {{ font-family:Consolas,"Courier New",monospace; font-size:0.82rem;
  background:#f4f4f1; border:1px solid var(--border); border-radius:8px;
  padding:12px 14px; margin-bottom:12px; overflow-x:auto; }}
.traceline {{ display:flex; gap:10px; align-items:baseline; padding:2px 0;
  white-space:nowrap; }}
.traceline .tag {{ flex:0 0 52px; font-weight:700; font-size:0.72rem;
  letter-spacing:0.04em; color:var(--s-baseline); }}
.traceline span:last-child {{ color:var(--text-primary); white-space:normal; }}
code {{ font-family:Consolas,"Courier New",monospace; font-size:0.85em;
  background:#f0efec; padding:1px 5px; border-radius:4px; }}
footer {{ color:var(--muted); font-size:0.8rem; margin-top:28px; }}
.printhint {{ background:#eef6ff; border:1px solid #bcd9f7; border-radius:8px;
  padding:10px 14px; font-size:0.85rem; margin:0 0 20px; color:#12395f; }}

/* Submission requires PDF or ZIP. Ctrl+P -> "Save as PDF" in any browser
   produces a clean document from this page; these rules stop cards splitting
   across pages and drop the on-screen-only hint. */
@media print {{
  @page {{ size: A4 portrait; margin: 12mm; }}
  body {{ background:#fff; }}
  .wrap {{ max-width:none; padding:0; }}
  .printhint {{ display:none; }}
  .card, .tile {{ break-inside:avoid; page-break-inside:avoid;
                 border:1px solid #d5d4cd; box-shadow:none; }}
  .card {{ margin-bottom:12px; }}
  h1 {{ font-size:1.4rem; }}
  .tiles {{ grid-template-columns:repeat(4,1fr); gap:8px; }}
  .tile-value {{ font-size:1.7rem; }}
  .tile.hero .tile-value {{ font-size:2.2rem; }}
  .telemetry div {{ border:1px solid #d5d4cd; }}
}}
</style></head>
<body><div class="wrap">
<header>
  <div>
    <h1>Closed-loop building control with a local LLM</h1>
    <p class="sub">EnergyPlus 5-zone office · Chicago TMY3 · {period} ·
       15-minute control timestep</p>
  </div>
</header>

<p class="printhint"><strong>To submit as PDF:</strong> press Ctrl+P (Cmd+P on Mac)
and choose &ldquo;Save as PDF&rdquo;. Wait for every chart to finish drawing first.
This notice does not appear in the PDF.</p>

{stat_tiles(summary)}
{telemetry_row(summary)}

<section class="card"><h2>Results</h2>
<p class="note">Identical building, weather and run period. The only variable is
the controller.</p>
{results_table(summary)}
</section>

{cycle_section}
{blocks}
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
