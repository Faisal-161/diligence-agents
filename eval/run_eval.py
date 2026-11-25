"""Benchmark: multi-agent graph vs single-agent baseline on the same cases and tools.

Per architecture:
- recommendation_accuracy: recommendation matches the expected one
- flag_recall: share of must-find risk flags that were found
- false_flags: must-not flags raised (e.g. another company's tax lien)
- claim_support_rate: share of report claims that pass the citation checks
- wrong_entity_citations: claims citing documents about a different company
- llm_calls, tokens, cost_usd, wall_ms per company

Run: python eval/run_eval.py   (DILIGENCE_LLM=claude for real model numbers)
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diligence.agents import Deps, verify  # noqa: E402
from diligence.config import Settings  # noqa: E402
from diligence.local import run_multi_agent, run_single_agent  # noqa: E402

EVAL_DATE = date(2026, 11, 25)  # fixed so "incorporated N months ago" stays stable


def load_cases(path: Path = ROOT / "eval" / "cases.jsonl") -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def score(deps: Deps, case: dict, res) -> dict:
    r = res.report
    checks, _, _ = verify(deps, case["company"], r)
    flags = set(r.risk_flags)
    return {
        "company": case["company"],
        "architecture": res.architecture,
        "expected": case["expected"],
        "got": r.recommendation,
        "correct": r.recommendation == case["expected"],
        "flags": sorted(flags),
        "missed_flags": sorted(set(case["must_flag"]) - flags),
        "false_flags": sorted(set(case["must_not_flag"]) & flags),
        "claims": len(checks),
        "supported": sum(c.supported for c in checks),
        "wrong_entity": sum(c.reason.startswith("cites a source about a different") for c in checks),
        "llm_calls": res.llm_calls,
        "tokens": res.tokens,
        "cost_usd": res.cost_usd,
        "wall_ms": res.wall_ms,
        "rounds": len(res.critiques),
    }


def summarize(rows: list[dict], cases: list[dict]) -> dict:
    must = sum(len(c["must_flag"]) for c in cases)
    return {
        "recommendation_accuracy": round(sum(r["correct"] for r in rows) / len(rows), 3),
        "flag_recall": round(1 - sum(len(r["missed_flags"]) for r in rows) / must, 3) if must else None,
        "false_flags": sum(len(r["false_flags"]) for r in rows),
        "claim_support_rate": round(sum(r["supported"] for r in rows) / max(1, sum(r["claims"] for r in rows)), 3),
        "wrong_entity_citations": sum(r["wrong_entity"] for r in rows),
        "avg_llm_calls": round(statistics.mean(r["llm_calls"] for r in rows), 1),
        "avg_tokens": round(statistics.mean(r["tokens"] for r in rows)),
        "avg_cost_usd": round(statistics.mean(r["cost_usd"] for r in rows), 5),
        "avg_wall_ms": round(statistics.mean(r["wall_ms"] for r in rows), 1),
    }


def run(settings: Settings | None = None) -> dict:
    deps = Deps.from_settings(settings or Settings.from_env())
    deps.today = EVAL_DATE
    cases = load_cases()
    rows = {"multi_agent": [], "single_agent": []}
    for case in cases:
        rows["multi_agent"].append(score(deps, case, run_multi_agent(deps, case["company"])))
        rows["single_agent"].append(score(deps, case, run_single_agent(deps, case["company"])))
    return {"llm": deps.settings.llm, "summary": {k: summarize(v, cases) for k, v in rows.items()}, "rows": rows}


if __name__ == "__main__":
    report = run()
    (ROOT / "eval" / "results.json").write_text(json.dumps(report, indent=2))
    print(f"LLM: {report['llm']}")
    print(json.dumps(report["summary"], indent=2))
    for arch, rs in report["rows"].items():
        for r in rs:
            if not r["correct"] or r["false_flags"] or r["missed_flags"]:
                print(
                    f"  {arch:<13} {r['company']:<30} expected={r['expected']:<24} got={r['got']:<24} "
                    f"missed={r['missed_flags']} false={r['false_flags']}"
                )
