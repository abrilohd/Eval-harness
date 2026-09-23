import httpx

from .base import RagResponse, Source


class HttpAdapter:
    """POST {"query", "top_k"} -> {"answer", "sources": [{"id", "text"}]}"""

    def __init__(self, url: str, token: str | None = None, timeout: float = 60.0):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._url = url
        self._client = httpx.Client(headers=headers, timeout=timeout)

    def query(self, question: str, top_k: int) -> RagResponse:
        r = self._client.post(self._url, json={"query": question, "top_k": top_k})
        r.raise_for_status()
        data = r.json()
        return RagResponse(
            answer=data["answer"],
            sources=[Source(id=s["id"], text=s["text"]) for s in data["sources"]],
        )
