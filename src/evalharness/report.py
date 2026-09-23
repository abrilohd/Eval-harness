import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .runner import CaseResult


def write_report(
    path: Path,
    *,
    label: str,
    dataset: str,
    k: int,
    summary: dict,
    results: list[CaseResult],
) -> None:
    payload = {
        "label": label,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "k": k,
        "summary": summary,
        "cases": [asdict(r) for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_summary(summary: dict, k: int) -> str:
    return "\n".join(
        [
            f"cases         {summary['cases']} ({summary['errors']} errors)",
            f"hit@{k:<9} {summary['hit_rate']:.1%}",
            f"mrr@{k:<9} {summary['mrr']:.3f}",
            f"faithfulness  {_fmt(summary['faithfulness'])}",
        ]
    )


def compare_reports(before: dict, after: dict) -> str:
    lines = [
        f"{before['label']} -> {after['label']}",
        f"{'metric':<14}{'before':>9}{'after':>9}{'delta':>9}",
    ]
    for metric in ("hit_rate", "mrr", "faithfulness"):
        b, a = before["summary"][metric], after["summary"][metric]
        delta = "n/a" if b is None or a is None else f"{a - b:+.3f}"
        lines.append(f"{metric:<14}{_fmt(b):>9}{_fmt(a):>9}{delta:>9}")
    return "\n".join(lines)
