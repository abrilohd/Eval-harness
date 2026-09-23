from dataclasses import dataclass, field
from statistics import mean
from typing import Protocol

from .adapters.base import RagAdapter
from .dataset import Case
from .metrics.retrieval import hit_at_k, reciprocal_rank


class Judge(Protocol):
    def score(self, question: str, answer: str, contexts: list[str]) -> tuple[float, list[str]]: ...


@dataclass
class CaseResult:
    id: str
    question: str
    hit: bool
    rr: float
    retrieved: list[str]
    faithfulness: float | None = None
    unsupported: list[str] = field(default_factory=list)
    error: str | None = None


def run_cases(
    cases: list[Case], adapter: RagAdapter, k: int, judge: Judge | None = None
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        try:
            resp = adapter.query(case.question, k)
            retrieved = [s.id for s in resp.sources][:k]
            faith, unsupported = None, []
            if judge is not None:
                contexts = [s.text for s in resp.sources[:k]]
                faith, unsupported = judge.score(case.question, resp.answer, contexts)
            results.append(
                CaseResult(
                    id=case.id,
                    question=case.question,
                    hit=hit_at_k(retrieved, case.expected_sources, k),
                    rr=reciprocal_rank(retrieved, case.expected_sources, k),
                    retrieved=retrieved,
                    faithfulness=faith,
                    unsupported=unsupported,
                )
            )
        except Exception as e:
            results.append(
                CaseResult(
                    id=case.id,
                    question=case.question,
                    hit=False,
                    rr=0.0,
                    retrieved=[],
                    error=f"{type(e).__name__}: {e}",
                )
            )
    return results


def summarize(results: list[CaseResult]) -> dict:
    n = len(results)
    scored = [r.faithfulness for r in results if r.faithfulness is not None]
    return {
        "cases": n,
        "errors": sum(r.error is not None for r in results),
        "hit_rate": sum(r.hit for r in results) / n,
        "mrr": sum(r.rr for r in results) / n,
        "faithfulness": mean(scored) if scored else None,
    }


def check_gates(
    summary: dict, min_hit_rate: float | None = None, min_faithfulness: float | None = None
) -> list[str]:
    failures: list[str] = []
    if summary["errors"]:
        failures.append(f"{summary['errors']} case(s) errored")
    if min_hit_rate is not None and summary["hit_rate"] < min_hit_rate:
        failures.append(f"hit_rate {summary['hit_rate']:.3f} < {min_hit_rate}")
    if min_faithfulness is not None:
        faith = summary["faithfulness"]
        if faith is None:
            failures.append("faithfulness gate set but no faithfulness scores")
        elif faith < min_faithfulness:
            failures.append(f"faithfulness {faith:.3f} < {min_faithfulness}")
    return failures
