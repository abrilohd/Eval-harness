# Eval-harness

A lightweight Retrieval-Augmented Generation (RAG) evaluation harness for testing whether a blog or knowledge-base assistant retrieves the right sources and answers faithfully.

This project runs a fixed set of questions against a RAG API, measures retrieval quality and answer grounding, and fails CI when a change causes regression. It is designed for teams iterating on prompts, chunking, retrieval settings, and document structure without sacrificing answer quality.

## Why this project exists

When building a RAG system for a blog, knowledge base, or documentation search experience, the real risk is not just whether the model answers at all — it is whether it:

- retrieves the correct source material,
- ranks the most relevant chunks first,
- stays grounded in the retrieved context, and
- avoids hallucinating unsupported claims.

`Eval-harness` gives you a repeatable way to catch those regressions early.

## What it measures

| Metric | What it checks |
| --- | --- |
| Hit@k | Whether at least one expected source appears in the top-k retrieved chunks |
| MRR@k | How early the first expected source appears in the ranking |
| Faithfulness | How much of the generated answer is supported by the retrieved context |

This makes it useful for evaluating RAG pipelines on real content, including blog posts, product docs, tutorials, and internal knowledge sources.

## Features

- JSONL dataset format for test cases and expected sources
- HTTP adapter for calling a RAG endpoint
- Retrieval metrics for hit rate and reciprocal rank
- LLM-based faithfulness judging
- CLI for running, saving, and comparing experiments
- CI-ready thresholds to fail builds on regressions

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env
# set EVAL_TARGET_URL and ANTHROPIC_API_KEY

# run a baseline evaluation
 evalharness run --dataset datasets/example.jsonl --k 5 --label baseline

# check a new configuration against thresholds
 evalharness run --dataset datasets/example.jsonl --k 5 --label chunk-256 \
   --min-hit-rate 0.75 --min-faithfulness 0.80

# compare results from two runs
 evalharness compare results/baseline.json results/chunk-256.json
```

You can disable faithfulness judging for a retrieval-only check:

```bash
evalharness run --dataset datasets/example.jsonl --k 5 --label baseline --no-judge
```

If any threshold is missed, the command exits with a non-zero status code.

## Dataset format

Each line in `datasets/*.jsonl` is a JSON object:

```json
{"id": "q001", "question": "What does the ivfflat index trade off?", "expected_sources": ["pgvector-notes.md"]}
```

`expected_sources` are source identifiers returned by your RAG system. A hit is recorded when any expected source is present in the top-k results.

## Target API contract

The harness expects a RAG endpoint that accepts a query and returns an answer plus ranked sources:

```http
POST $EVAL_TARGET_URL
```

Request:

```json
{"query": "string", "top_k": 5}
```

Response:

```json
{"answer": "string", "sources": [{"id": "string", "text": "string"}]}
```

The `sources` array should be in ranked order. If you are integrating a new backend, implement a small adapter in `src/evalharness/adapters/`.

## Project layout

```text
Eval-harness/
├── .github/workflows/ci.yml
├── datasets/
│   └── example.jsonl
├── results/
├── src/evalharness/
│   ├── adapters/
│   │   ├── base.py
│   │   └── http.py
│   ├── metrics/
│   │   ├── retrieval.py
│   │   └── faithfulness.py
│   ├── cli.py
│   ├── dataset.py
│   ├── report.py
│   └── runner.py
├── tests/
├── .env.example
├── LICENSE
├── pyproject.toml
├── README.md
└── .gitignore
```

## CI pipeline

The repository includes CI checks to keep the RAG evaluation pipeline reliable:

- Ruff for linting
- pytest for test validation
- optional live evaluation runs using a configured RAG endpoint and API credentials

## Example use cases

This project is useful for evaluating:

- blog Q&A systems,
- internal docs assistants,
- customer support copilots,
- documentation retrieval pipelines,
- prompt and chunking experiments for RAG.

## Roadmap

- [ ] Build a realistic dataset from production content
- [ ] Add more retrieval and reranking experiments
- [ ] Compare chunk sizes and top-k values
- [ ] Track latency and cost per evaluation run
- [ ] Run the harness in CI for regression prevention

## License

MIT
