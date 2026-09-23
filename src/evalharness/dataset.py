from pathlib import Path

from pydantic import BaseModel, Field


class Case(BaseModel):
    id: str
    question: str
    expected_sources: list[str] = Field(min_length=1)


def load_dataset(path: str | Path) -> list[Case]:
    cases: list[Case] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(Case.model_validate_json(line))
        except ValueError as e:
            raise ValueError(f"{path}:{n}: {e}") from e
    if not cases:
        raise ValueError(f"{path}: no cases found")
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate case ids")
    return cases
