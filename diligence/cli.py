"""Command line.

diligence run "Bluepeak Logistics Ltd"            # multi-agent, prints the report
diligence run "Summit Print Co" --review          # pause for you to approve the plan
diligence run "Orion Data Labs Inc." --single     # single-agent baseline
diligence run "..." --engine graph                # use LangGraph (default: graph if installed)
"""

from __future__ import annotations

import argparse
import sys

from .agents import Deps
from .local import PlanRejected, run_multi_agent, run_single_agent
from .report import to_markdown


def _review(plan):
    print("Proposed research plan:")
    for t in plan.tasks:
        print(f"  {t.id} [{t.topic}/{t.source}] {t.question}")
    return plan if input("Approve? [y/N] ").strip().lower() == "y" else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="diligence")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("company")
    r.add_argument("--single", action="store_true", help="single-agent baseline")
    r.add_argument("--review", action="store_true", help="approve the plan before research starts")
    r.add_argument("--engine", choices=["auto", "graph", "local"], default="auto")
    r.add_argument("--trace", action="store_true", help="print each agent step")
    args = p.parse_args(argv)

    deps = Deps.from_settings()
    try:
        if args.single:
            res = run_single_agent(deps, args.company)
        elif args.engine in ("auto", "graph"):
            try:
                from .graph import run_graph
            except ImportError:
                if args.engine == "graph":
                    raise
                res = run_multi_agent(deps, args.company, _review if args.review else None)
            else:
                if args.review:  # use the local hook for an interactive terminal prompt
                    res = run_multi_agent(deps, args.company, _review)
                else:
                    res = run_graph(deps, args.company)
        else:
            res = run_multi_agent(deps, args.company, _review if args.review else None)
    except PlanRejected:
        print("Plan rejected; nothing was researched.")
        return 1
    if args.trace:
        print("\n".join(res.log) + "\n")
    print(to_markdown(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
