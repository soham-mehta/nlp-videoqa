from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any, Protocol

from openai import OpenAI

from src.agentic.agent import AgentResult, run_agent
from src.agentic.backend import Chunk
from src.agentic.benchmark_backend import BenchmarkRetrievalBackend
from src.config.settings import GenerationConfig
from src.rag.answering import BaselineRAGAnsweringService
from src.rag.schemas import AnswerRequest, GeneratedAnswer, RAGRunResult
from src.retrieval.schemas import RetrievedItem
from src.retrieval.service import RetrieverService


class RetrievalRunner(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ): ...


def _dedupe_items(items: list[RetrievedItem]) -> list[RetrievedItem]:
    seen: set[str] = set()
    out: list[RetrievedItem] = []
    for item in items:
        if item.item_id in seen:
            continue
        seen.add(item.item_id)
        out.append(item)
    return out


def _chunk_to_retrieved_item(chunk: Chunk) -> RetrievedItem:
    return RetrievedItem(
        item_id=chunk.item_id,
        video_id=chunk.video_id,
        modality=chunk.modality,
        score=float(chunk.score or 0.0),
        timestamp_start=chunk.timestamp_start,
        timestamp_end=chunk.timestamp_end,
        text=chunk.text,
        frame_path=chunk.frame_path,
        source_id=chunk.source_id,
    )


def _format_initial_context(items: list[RetrievedItem]) -> str:
    lines = ["Transcript evidence:"]
    text_items = [item for item in items if item.modality == "text"]
    frame_items = [item for item in items if item.modality == "frame"]
    if text_items:
        for item in text_items:
            lines.append(
                f"- [{item.item_id}] {item.timestamp_start:.2f}-{item.timestamp_end:.2f}s: {item.text or '(no text)'}"
            )
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("Frame evidence:")
    if frame_items:
        for item in frame_items:
            frame_ref = item.frame_path or "(no frame path)"
            lines.append(
                f"- [{item.item_id}] {item.timestamp_start:.2f}-{item.timestamp_end:.2f}s: {frame_ref}"
            )
    else:
        lines.append("- (none)")
    return "\n".join(lines)


class AgenticRAGAnsweringService:
    """
    Tool-augmented answerer that starts from the baseline retrieval bundle, then lets
    the model call retrieval tools over the same benchmark index before answering.
    """

    def __init__(
        self,
        *,
        retriever: RetrievalRunner,
        generation_config: GenerationConfig,
        metadata_jsonl_path: str | None = None,
        backend: Any | None = None,
        max_tool_calls: int = 8,
    ) -> None:
        self.retriever = retriever
        self.generation_config = generation_config
        self.max_tool_calls = max_tool_calls
        self.client = OpenAI(
            base_url=generation_config.base_url,
            api_key=generation_config.api_key,
            timeout=generation_config.timeout_sec,
        )
        if backend is not None:
            self.backend = backend
        elif isinstance(retriever, RetrieverService):
            if metadata_jsonl_path is None:
                raise ValueError("metadata_jsonl_path is required for local benchmark backend")
            self.backend = BenchmarkRetrievalBackend(
                embedder=retriever.embedder,
                vector_store=retriever.vector_store,
                metadata_jsonl_path=metadata_jsonl_path,
            )
        else:
            raise ValueError("backend is required when retriever is not a local RetrieverService")

    @staticmethod
    def _collect_observed_items(agent_result: AgentResult) -> list[RetrievedItem]:
        items: list[RetrievedItem] = []
        for step in agent_result.trajectory:
            try:
                payload = json.loads(step.result)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, list):
                continue
            for row in payload:
                if not isinstance(row, dict) or "item_id" not in row:
                    continue
                try:
                    items.append(
                        RetrievedItem(
                            item_id=str(row["item_id"]),
                            video_id=str(row["video_id"]),
                            modality=str(row["modality"]),  # type: ignore[arg-type]
                            score=float(row.get("score", 0.0)),
                            timestamp_start=float(row["timestamp_start"]),
                            timestamp_end=float(row["timestamp_end"]),
                            text=row.get("text"),
                            frame_path=row.get("frame_path"),
                            source_id=row.get("source_id"),
                        )
                    )
                except Exception:
                    continue
        return _dedupe_items(items)

    def _resolve_final_items(
        self,
        *,
        agent_result: AgentResult,
        initial_items: list[RetrievedItem],
    ) -> list[RetrievedItem]:
        if agent_result.evidence_chunk_ids:
            resolved = [
                _chunk_to_retrieved_item(chunk)
                for chunk in self.backend.resolve_chunks(agent_result.evidence_chunk_ids)
            ]
            if resolved:
                return _dedupe_items(resolved)

        observed = self._collect_observed_items(agent_result)
        if observed:
            return observed

        return _dedupe_items(initial_items)

    def run(self, request: AnswerRequest) -> RAGRunResult:
        t0 = time.perf_counter()
        filters = {"video_id": request.video_id} if request.video_id else None
        retrieval_result = self.retriever.retrieve(
            query=request.question,
            top_k=request.retrieval_policy.top_k_total,
            filters=filters,
        )
        initial_items = BaselineRAGAnsweringService._apply_retrieval_policy(
            retrieval_result.items,
            max_text_items=request.retrieval_policy.max_text_items,
            max_frame_items=request.retrieval_policy.max_frame_items,
            dedupe_seconds=request.retrieval_policy.dedupe_seconds,
        )
        agent_result = run_agent(
            question=request.question,
            video_id=request.video_id,
            initial_context=_format_initial_context(initial_items),
            backend=self.backend,
            client=self.client,
            model=self.generation_config.model_name,
            max_tool_calls=self.max_tool_calls,
            temperature=self.generation_config.temperature,
        )
        final_items = self._resolve_final_items(
            agent_result=agent_result,
            initial_items=initial_items,
        )
        evidence_bundle = BaselineRAGAnsweringService._build_evidence_bundle(
            question=request.question,
            items=final_items,
        )
        dt = time.perf_counter() - t0

        return RAGRunResult(
            request=request,
            final_answer=agent_result.answer,
            retrieved_items=final_items,
            evidence_bundle=evidence_bundle,
            generated_answer=GeneratedAnswer(
                answer_text=agent_result.answer,
                model_name=self.generation_config.model_name,
                prompt_text=agent_result.prompt_text,
                used_evidence_ids=[item.item_id for item in final_items],
                raw_response_text=agent_result.answer,
                metadata={
                    "agent_total_tool_calls": agent_result.total_tool_calls,
                    "agent_stopped_reason": agent_result.stopped_reason,
                },
            ),
            debug_info={
                "duration_sec": round(dt, 3),
                "retrieved_count_before_policy": len(retrieval_result.items),
                "retrieved_count_after_policy": len(initial_items),
                "final_retrieved_count": len(final_items),
                "num_transcript_evidence": len(evidence_bundle.transcripts),
                "num_frame_evidence": len(evidence_bundle.frames),
                "retrieval_policy": asdict(request.retrieval_policy),
                "agent_trajectory": [asdict(step) for step in agent_result.trajectory],
                "agent_total_tool_calls": agent_result.total_tool_calls,
                "agent_stopped_reason": agent_result.stopped_reason,
                "initial_retrieved_item_ids": [item.item_id for item in initial_items],
                "final_retrieved_item_ids": [item.item_id for item in final_items],
            },
        )
