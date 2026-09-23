from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Source:
    id: str
    text: str


@dataclass(frozen=True)
class RagResponse:
    answer: str
    sources: list[Source]


class RagAdapter(Protocol):
    def query(self, question: str, top_k: int) -> RagResponse: ...
