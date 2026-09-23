import argparse
import os
import sys
from pathlib import Path

from .adapters.http import HttpAdapter
from .dataset import load_dataset
from .metrics.faithfulness import FaithfulnessJudge
from .report import compare_reports, format_summary, load_report, write_report
from .runner import check_gates, run_cases, summarize


def _run(args: argparse.Namespace) -> int:
    url = args.url or os.getenv("EVAL_TARGET_URL")
    if not url:
        print("error: set --url or EVAL_TARGET_URL", file=sys.stderr)
        return 2

    cases = load_dataset(args.dataset)
    adapter = HttpAdapter(url, token=os.getenv("EVAL_TARGET_TOKEN"))
    judge = None if args.no_judge else FaithfulnessJudge()

    results = run_cases(cases, adapter, k=args.k, judge=judge)
    summary = summarize(results)

    out = Path(args.out or f"results/{args.label}.json")
    write_report(
        out, label=args.label, dataset=args.dataset, k=args.k, summary=summary, results=results
    )
    print(format_summary(summary, args.k))
    print(f"saved {out}")

    failures = check_gates(summary, args.min_hit_rate, args.min_faithfulness)
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


def _compare(args: argparse.Namespace) -> int:
    print(compare_reports(load_report(args.before), load_report(args.after)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalharness")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="score a RAG API against a dataset")
    run.add_argument("--dataset", required=True)
    run.add_argument("--url", help="defaults to $EVAL_TARGET_URL")
    run.add_argument("--k", type=int, default=5)
    run.add_argument("--label", default="run")
    run.add_argument("--out", help="defaults to results/<label>.json")
    run.add_argument("--no-judge", action="store_true", help="skip faithfulness scoring")
    run.add_argument("--min-hit-rate", type=float)
    run.add_argument("--min-faithfulness", type=float)
    run.set_defaults(func=_run)

    cmp_ = sub.add_parser("compare", help="diff two saved runs")
    cmp_.add_argument("before")
    cmp_.add_argument("after")
    cmp_.set_defaults(func=_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
