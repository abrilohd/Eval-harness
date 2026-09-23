import json
import os

from anthropic import Anthropic

PROMPT = """You are grading a RAG answer for faithfulness.
Score how much of the answer is supported by the context, from 0.0 to 1.0.
1.0 = every claim is supported. 0.0 = nothing is supported or it contradicts the context.
Ignore style. Do not use outside knowledge.

Question: {question}

Context:
{context}

Answer:
{answer}

Reply with JSON only: {{"score": <float>, "unsupported": [<claims not supported>]}}"""


def parse_judgement(text: str) -> tuple[float, list[str]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    data = json.loads(cleaned)
    score = min(1.0, max(0.0, float(data["score"])))
    return score, [str(c) for c in data.get("unsupported", [])]


class FaithfulnessJudge:
    def __init__(self, model: str | None = None, client: Anthropic | None = None):
        self._client = client or Anthropic()
        self._model = model or os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001")

    def score(self, question: str, answer: str, contexts: list[str]) -> tuple[float, list[str]]:
        if not contexts:
            return 0.0, ["no retrieved context"]
        context = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=400,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(question=question, context=context, answer=answer),
                }
            ],
        )
        return parse_judgement(msg.content[0].text)
