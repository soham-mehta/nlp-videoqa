from __future__ import annotations

import time
from dataclasses import asdict

from src.rag.generator_base import MultimodalAnswerGenerator
from src.rag.schemas import (
    AnswerRequest,
    EvidenceBundle,
    FrameEvidence,
    RAGRunResult,
    TranscriptEvidence,
)
from src.retrieval.schemas import RetrievedItem
from src.retrieval.service import RetrieverService


class BaselineRAGAnsweringService:
    """
    Baseline single-pass multimodal RAG:
    retrieve -> build evidence -> generate grounded answer.
    """

    def __init__(self, retriever: RetrieverService, generator: MultimodalAnswerGenerator) -> None:
        self.retriever = retriever
        self.generator = generator

    @staticmethod
    def _is_near_duplicate(a: RetrievedItem, b: RetrievedItem, dedupe_seconds: float) -> bool:
        if a.video_id != b.video_id or a.modality != b.modality:
            return False
        return abs(a.timestamp_start - b.timestamp_start) <= dedupe_seconds

    @classmethod
    def _apply_retrieval_policy(
        cls,
        items: list[RetrievedItem],
        *,
        max_text_items: int,
        max_frame_items: int,
        dedupe_seconds: float,
    ) -> list[RetrievedItem]:
        selected: list[RetrievedItem] = []
        text_count = 0
        frame_count = 0
        for item in items:
            if any(cls._is_near_duplicate(item, prev, dedupe_seconds) for prev in selected):
                continue
            if item.modality == "text":
                if text_count >= max_text_items:
                    continue
                text_count += 1
            elif item.modality == "frame":
                if frame_count >= max_frame_items:
                    continue
                frame_count += 1
            selected.append(item)
            if text_count >= max_text_items and frame_count >= max_frame_items:
                break
        return selected

    @staticmethod
    def _build_evidence_bundle(question: str, items: list[RetrievedItem]) -> EvidenceBundle:
        transcript_items: list[TranscriptEvidence] = []
        frame_items: list[FrameEvidence] = []
        for item in items:
            if item.modality == "text" and item.text:
                transcript_items.append(
                    TranscriptEvidence(
                        evidence_id=item.item_id,
                        video_id=item.video_id,
                        timestamp_start=item.timestamp_start,
                        timestamp_end=item.timestamp_end,
                        text=item.text,
                        source_id=item.source_id,
                    )
                )
            elif item.modality == "frame" and item.frame_path:
                frame_items.append(
                    FrameEvidence(
                        evidence_id=item.item_id,
                        video_id=item.video_id,
                        timestamp_start=item.timestamp_start,
                        timestamp_end=item.timestamp_end,
                        frame_path=item.frame_path,
                        source_id=item.source_id,
                    )
                )
        return EvidenceBundle(
            question=question,
            transcripts=transcript_items,
            frames=frame_items,
            retrieved_items=items,
        )

    def run(self, request: AnswerRequest) -> RAGRunResult:
        t0 = time.perf_counter()
        filters = {"video_id": request.video_id} if request.video_id else None
        retrieval_result = self.retriever.retrieve(
            query=request.question,
            top_k=request.retrieval_policy.top_k_total,
            filters=filters,
        )
        policy_items = self._apply_retrieval_policy(
            retrieval_result.items,
            max_text_items=request.retrieval_policy.max_text_items,
            max_frame_items=request.retrieval_policy.max_frame_items,
            dedupe_seconds=request.retrieval_policy.dedupe_seconds,
        )
        evidence_bundle = self._build_evidence_bundle(
            question=request.question,
            items=policy_items,
        )
        generated = self.generator.generate_answer(
            question=request.question,
            retrieved_evidence=evidence_bundle,
            system_prompt=request.system_prompt,
        )
        dt = time.perf_counter() - t0
        return RAGRunResult(
            request=request,
            final_answer=generated.answer_text,
            retrieved_items=policy_items,
            evidence_bundle=evidence_bundle,
            generated_answer=generated,
            debug_info={
                "duration_sec": round(dt, 3),
                "retrieved_count_before_policy": len(retrieval_result.items),
                "retrieved_count_after_policy": len(policy_items),
                "num_transcript_evidence": len(evidence_bundle.transcripts),
                "num_frame_evidence": len(evidence_bundle.frames),
                "retrieval_policy": asdict(request.retrieval_policy),
            },
        )
