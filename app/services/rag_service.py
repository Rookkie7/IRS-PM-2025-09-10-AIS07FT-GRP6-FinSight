from __future__ import annotations

from typing import List, Dict
from app.adapters.llm.openai_llm import OpenAICompatLLM
from app.adapters.rag.retriever import Retriever
from app.adapters.rag.ranker import SimpleRanker
from app.adapters.rag.prompt_templates import build_finance_prompt
from app.ports.vector_index import VectorIndexPort
from app.adapters.db.news_repo import NewsRepo
from app.ports.embedding import EmbeddingProviderPort

class RagService:
    def __init__(
        self,
        index: VectorIndexPort,
        news_repo: NewsRepo,
        query_embedder: EmbeddingProviderPort,
        llm: OpenAICompatLLM,
        dim: int = 32,
    ):
        self.index = index
        self.news_repo = news_repo
        self.query_embedder = query_embedder
        self.llm = llm
        self.dim = dim
        self.retriever = Retriever(index, dim=dim)
        self.ranker = SimpleRanker(alpha=0.8, half_life_hours=72)

    async def answer(
        self,
        query_text: str | None = None,
        query_vector: List[float] | None = None,
        k: int = 5,
        temperature: float = 0.2,
        max_tokens: int = 512,
        filters: dict | None = None,
    ) -> dict:
        # 1) searching vector
        if query_vector is None:
            if not query_text:
                raise ValueError("either query_text or query_vector must be provided")
            vecs = await self.query_embedder.embed([query_text], dim=self.dim)
            query_vector = vecs[0]

        # 2) Retrieval Candidates
        hits = await self.retriever.retrieve(query_vector, top_k=max(20, k*3), filters=filters)

        # 3) Pull the original text (one-time batch get)
        ids = [hid for hid, _ in hits]
        docs = await self.news_repo.get_many(ids)
        id2doc: Dict[str, dict] = {str(d["_id"]): d for d in docs}

        # 4) rerank
        reranked = self.ranker.rerank(hits, id2doc, top_k=k)
        contexts = [id2doc[_id] for _id, _ in reranked if _id in id2doc]

        # 5) Constructing prompt words & tuning LLM
        prompt = build_finance_prompt(question=query_text or "N/A", contexts=contexts)
        answer = await self.llm.generate(prompt, temperature=temperature, max_tokens=max_tokens)

        # 6) Back to Answer + Quote
        citations = [
            {"id": _id, "score": score, "title": id2doc.get(_id, {}).get("title", "")}
            for _id, score in reranked
            if _id in id2doc
        ]
        return {"answer": answer, "citations": citations}