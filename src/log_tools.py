"""Compress EnergyPlus .err logs into something an LLM can actually read.

An EnergyPlus error file is mostly repetition: the same warning re-emitted once
per timestep with only the numbers changing. Feeding that raw into a prompt
burns the whole context window on noise. The pipeline here is:

  1. FILTER   -- keep only ** Severe **, ** Fatal ** and ** Warning ** records,
                 discarding the informational preamble and the continuation
                 chatter that carries no new signal.
  2. DEDUPE   -- normalise the varying numbers out of each message so that the
                 same warning at 400 different timesteps collapses to one line
                 tagged "x400".
  3. TRUNCATE -- rank by severity first, then by occurrence count, and keep the
                 top K.

`compress_err_file` reports the before/after sizes so the compression ratio can
be measured rather than asserted.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SEVERITY_RE = re.compile(r"\*\*\s*(Severe|Fatal|Warning)\s*\*\*", re.IGNORECASE)
CONTINUATION_RE = re.compile(r"\*\*\s*~~~\s*\*\*")

# Rank order for truncation: the worst problems survive truncation first.
SEVERITY_RANK = {"fatal": 0, "severe": 1, "warning": 2}

# Patterns whose *values* vary between otherwise-identical messages. Replacing
# them with placeholders is what makes deduplication actually collapse.
_NUMBER = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")
_TIMESTAMP = re.compile(
    r"\b\d{2}/\d{2}\s+\d{2}:\d{2}(?::\d{2})?\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d+\b",
    re.IGNORECASE,
)


# EnergyPlus object names are upper-case identifiers. The same warning emitted
# once per zone differs only by that name, so collapsing the name turns 13
# near-identical lines into one line plus a count.
_QUOTED = re.compile(r'"[^"]*"')
_UPPER_IDENT = re.compile(r"\b[A-Z][A-Z0-9_\-]{2,}(?:\s+[A-Z0-9_\-]+)*\b")


def _normalise(message: str) -> str:
    """Strip the varying parts so repeats of one warning hash together."""
    text = _TIMESTAMP.sub("<TIME>", message)
    # Identifiers first: substituting inside the quotes turns "FOO BAR" into
    # "<NAME>", which the quote rule then folds to a single <NAME>. Doing it
    # the other way round lets the placeholder match itself -> "<<NAME>>".
    text = _UPPER_IDENT.sub("<NAME>", text)
    text = _QUOTED.sub("<NAME>", text)
    text = _NUMBER.sub("#", text)
    return re.sub(r"\s+", " ", text).strip()


def _entities(message: str) -> list[str]:
    """Pull the identifier(s) out of a message so examples can be shown."""
    found = _QUOTED.findall(message) + _UPPER_IDENT.findall(message)
    return [f.strip('"').strip() for f in found if f.strip('"').strip()]


def parse_err(text: str) -> list[dict[str, Any]]:
    """Return deduplicated severity records, most severe and frequent first."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}

    for raw in text.splitlines():
        match = SEVERITY_RE.search(raw)
        if not match:
            continue                     # drops preamble and ~~~ continuations
        if CONTINUATION_RE.search(raw):
            continue

        severity = match.group(1).lower()
        message = raw[match.end():].strip()
        if not message:
            continue

        normalised = _normalise(message)
        key = (severity, normalised)
        entry = groups.get(key)
        if entry is None:
            groups[key] = {
                "severity": severity,
                "message": message,          # first concrete instance
                "template": normalised,
                "count": 1,
                "examples": _entities(message)[:1],
            }
        else:
            entry["count"] += 1
            for name in _entities(message)[:1]:
                if name not in entry["examples"] and len(entry["examples"]) < 3:
                    entry["examples"].append(name)

    records = list(groups.values())
    records.sort(key=lambda r: (SEVERITY_RANK.get(r["severity"], 9), -r["count"]))
    return records


def format_records(records: list[dict[str, Any]], max_lines: int) -> list[str]:
    lines = []
    for rec in records[:max_lines]:
        tag = rec["severity"].upper()
        if rec["count"] > 1:
            # Show the collapsed template plus a couple of concrete names, so
            # the reader keeps enough context to act without the raw log.
            body = rec.get("template", rec["message"])
            suffix = f"  x{rec['count']}"
            examples = rec.get("examples") or []
            if examples:
                shown = ", ".join(examples[:3])
                more = "..." if rec["count"] > len(examples) else ""
                suffix += f"  (e.g. {shown}{more})"
        else:
            body = rec["message"]
            suffix = ""
        lines.append(f"[{tag}] {body}{suffix}")
    return lines


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) -- good enough to report a ratio."""
    return max(1, len(text) // 4)


def compress_err_file(path: str | Path, max_lines: int = 25) -> dict[str, Any]:
    """Compress a .err file and report the before/after cost."""
    p = Path(path)
    if not p.exists():
        return {
            "ok": False,
            "error": f"no error file at {p}",
            "summary": "",
            "records": [],
        }

    raw = p.read_text(encoding="utf-8", errors="replace")
    records = parse_err(raw)
    lines = format_records(records, max_lines)

    total_occurrences = sum(r["count"] for r in records)
    counts = {"fatal": 0, "severe": 0, "warning": 0}
    for r in records:
        counts[r["severity"]] = counts.get(r["severity"], 0) + r["count"]

    header = (f"{counts['fatal']} fatal, {counts['severe']} severe, "
              f"{counts['warning']} warning "
              f"({len(records)} distinct, {total_occurrences} total)")
    summary = header + ("\n" + "\n".join(lines) if lines else "\n(no problems reported)")

    raw_lines = len(raw.splitlines())
    kept_lines = len(lines) + 1
    return {
        "ok": True,
        "summary": summary,
        "records": records[:max_lines],
        "distinct": len(records),
        "total_occurrences": total_occurrences,
        "counts": counts,
        "lines_before": raw_lines,
        "lines_after": kept_lines,
        "tokens_before": approx_tokens(raw),
        "tokens_after": approx_tokens(summary),
        "compression_ratio": round(approx_tokens(raw) / max(1, approx_tokens(summary)), 1),
        "truncated": max(0, len(records) - max_lines),
    }


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "out/actuation_baseline/eplusout.err"
    result = compress_err_file(target, max_lines=int(sys.argv[2]) if len(sys.argv) > 2 else 25)
    if not result["ok"]:
        print(result["error"])
        raise SystemExit(1)
    print(f"file            : {target}")
    print(f"lines  {result['lines_before']:>6} -> {result['lines_after']}")
    print(f"tokens {result['tokens_before']:>6} -> {result['tokens_after']} "
          f"({result['compression_ratio']}x reduction)")
    print(f"distinct issues : {result['distinct']} "
          f"(from {result['total_occurrences']} occurrences)")
    print("-" * 60)
    print(result["summary"])
