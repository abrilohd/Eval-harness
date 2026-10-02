from __future__ import annotations

from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from pydantic import BaseModel, Field

from .base import RagResponse, Source


class RAGState(BaseModel):
    question: str
    retrieved_docs: list[Document] = Field(default_factory=list)
    context: str = ""
    answer: str = ""
    sources: list[Source] = Field(default_factory=list)


class LangGraphRAGAdapter:
    """A concrete LangGraph-based RAG adapter for the eval harness."""

    def __init__(
        self,
        documents: list[Document],
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "claude-3-5-sonnet-20241022",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
    ):
        if not documents:
            raise ValueError("documents must not be empty")

        self.top_k = top_k
        self.llm = ChatAnthropic(model=llm_model, temperature=0)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        split_docs: list[Document] = []
        for doc in documents:
            chunks = splitter.split_text(doc.page_content)
            for chunk in chunks:
                split_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            "source_id": doc.metadata.get(
                                "source_id", doc.metadata.get("source", "unknown")
                            ),
                        },
                    )
                )

        embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = FAISS.from_documents(split_docs, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 8})
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(RAGState)

        def retrieve_docs(state: RAGState) -> dict[str, Any]:
            docs = self.retriever.invoke(state.question)
            return {"retrieved_docs": docs}

        def build_context(state: RAGState) -> dict[str, Any]:
            selected = state.retrieved_docs[: self.top_k]
            context = "\n\n".join(
                f"[Source: {doc.metadata.get('source_id', 'unknown')}]\n{doc.page_content}"
                for doc in selected
            )
            return {"retrieved_docs": selected, "context": context}

        def generate_answer(state: RAGState) -> dict[str, Any]:
            prompt = f"""
You are a helpful assistant answering questions using only the provided context.
If the answer is not in the context, say: "I cannot find this information in the provided context."

Context:
{state.context}

Question:
{state.question}
"""
            response = self.llm.invoke(prompt)
            return {"answer": response.content}

        def format_sources(state: RAGState) -> dict[str, Any]:
            sources: list[Source] = []
            seen: set[str] = set()
            for doc in state.retrieved_docs:
                source_id = doc.metadata.get("source_id", "unknown")
                if source_id not in seen:
                    sources.append(Source(id=source_id, text=doc.page_content))
                    seen.add(source_id)
            return {"sources": sources[: self.top_k]}

        graph.add_node("retrieve", retrieve_docs)
        graph.add_node("build_context", build_context)
        graph.add_node("generate_answer", generate_answer)
        graph.add_node("format_sources", format_sources)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "build_context")
        graph.add_edge("build_context", "generate_answer")
        graph.add_edge("generate_answer", "format_sources")

        return graph.compile()

    def query(self, question: str, top_k: int) -> RagResponse:
        result = self.graph.invoke(RAGState(question=question))
        return RagResponse(
            answer=result["answer"],
            sources=result["sources"][:top_k],
        )
