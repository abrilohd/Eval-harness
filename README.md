# Eval-harness

Regression tests for RAG quality. Runs a fixed question set against a RAG API, scores retrieval and answer faithfulness, and fails CI when the scores drop.

Built to test [Voice AI Study Coach](https://github.com/abrilohd). Works with any API that follows the contract below.

> **Status:** scaffold. Metrics, CLI, and CI are in place. The real dataset and first baseline are next (see [Roadmap](#roadmap)).

## What it measures

| Metric | How |
|---|---|
| **Hit@k** | Share of questions where an expected source appears in the top k retrieved chunks |
| **MRR@k** | Mean of 1/rank of the first expected source |
| **Faithfulness** | An LLM judge scores 0 to 1 how much of the answer is supported by the retrieved chunks |

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env        # set EVAL_TARGET_URL and ANTHROPIC_API_KEY

evalharness run --dataset datasets/example.jsonl --k 5 --label baseline
evalharness run --dataset datasets/example.jsonl --k 5 --label chunk-256 \
  --min-hit-rate 0.75 --min-faithfulness 0.80
evalharness compare results/baseline.json results/chunk-256.json
```

Retrieval only, no judge calls: add `--no-judge`.

Exit code is `1` if any case errors or a threshold is missed.

## Dataset

One JSON object per line in `datasets/*.jsonl`:

```json
{"id": "q001", "question": "What does the ivfflat index trade off?", "expected_sources": ["pgvector-notes.md"]}
```

`expected_sources` are source IDs as your API returns them. A hit means any one of them was retrieved.

## Target API contract

`POST $EVAL_TARGET_URL`

```json
// request
{"query": "string", "top_k": 5}

// response
{"answer": "string", "sources": [{"id": "string", "text": "string"}]}
```

`sources` must be in ranked order. For a new system, write a small class implementing `RagAdapter` in `src/evalharness/adapters/`.

## CI

- `ci.yml` runs Ruff and pytest on every push and PR.
- The `eval` job runs the harness against a live API on manual dispatch. It needs `EVAL_TARGET_URL`, `EVAL_TARGET_TOKEN`, and `ANTHROPIC_API_KEY` as repo secrets.

## Results

Filled in as experiments run. One change per row.

| Run | Change | Hit@5 | MRR@5 | Faithfulness |
|---|---|---|---|---|
| baseline | none | — | — | — |

## Layout

```
Eval-harness/
├── .github/workflows/ci.yml
├── datasets/
│   └── example.jsonl
├── results/                    # run outputs, baseline.json is committed
├── src/evalharness/
│   ├── adapters/
│   │   ├── base.py             # RagAdapter protocol
│   │   └── http.py             # generic HTTP target
│   ├── metrics/
│   │   ├── retrieval.py        # hit@k, reciprocal rank
│   │   └── faithfulness.py     # LLM judge
│   ├── cli.py                  # run, compare
│   ├── dataset.py              # JSONL loader and validation
│   ├── report.py               # save, load, compare runs
│   └── runner.py               # per-case loop, summary, gates
├── tests/
├── .env.example
├── LICENSE
└── pyproject.toml
```

## Roadmap

- [ ] 30 to 50 cases built from real documents
- [ ] `POST /eval/query` endpoint on Study Coach
- [ ] Baseline run, committed as `results/baseline.json`
- [ ] Experiment: chunk size
- [ ] Experiment: top-k and reranking
- [ ] Run the harness from Study Coach's CI
- [ ] Latency and cost per case in the report

## License

MIT
