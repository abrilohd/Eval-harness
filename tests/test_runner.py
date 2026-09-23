from evalharness.adapters.base import RagResponse, Source
from evalharness.dataset import Case
from evalharness.runner import check_gates, run_cases, summarize


class FakeAdapter:
    def query(self, question, top_k):
        if question == "boom":
            raise RuntimeError("down")
        return RagResponse(answer="a", sources=[Source("s1", "t1"), Source("s2", "t2")])


class FakeJudge:
    def score(self, question, answer, contexts):
        return 0.8, []


CASES = [
    Case(id="1", question="ok", expected_sources=["s2"]),
    Case(id="2", question="miss", expected_sources=["zzz"]),
]


def test_summary_metrics():
    s = summarize(run_cases(CASES, FakeAdapter(), k=2, judge=FakeJudge()))
    assert s["hit_rate"] == 0.5
    assert s["mrr"] == 0.25
    assert s["faithfulness"] == 0.8
    assert s["errors"] == 0


def test_adapter_error_is_recorded_not_raised():
    boom = [Case(id="3", question="boom", expected_sources=["s1"])]
    results = run_cases(boom, FakeAdapter(), k=2)
    assert results[0].error and not results[0].hit
    assert summarize(results)["errors"] == 1


def test_gates():
    s = summarize(run_cases(CASES, FakeAdapter(), k=2, judge=FakeJudge()))
    assert check_gates(s, min_hit_rate=0.4, min_faithfulness=0.7) == []
    assert len(check_gates(s, min_hit_rate=0.9)) == 1
    no_judge = summarize(run_cases(CASES, FakeAdapter(), k=2))
    assert check_gates(no_judge, min_faithfulness=0.7)
