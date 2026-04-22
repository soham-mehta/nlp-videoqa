from __future__ import annotations

from src.rag.schemas import EvidenceBundle


DEFAULT_GROUNDED_SYSTEM_PROMPT = """You are answering questions about a specific video using only the provided retrieved evidence.
The evidence may include transcript excerpts and video frames.
Base your answer only on that evidence.
If the evidence is incomplete or ambiguous, say that clearly.
Prefer precise answers over broad guesses.
When useful, mention which timestamps or evidence items support the answer.
Do not claim certainty when the evidence does not support it.
Do not include hidden reasoning; output only the final grounded answer."""


def build_grounded_user_prompt(question: str, evidence: EvidenceBundle) -> str:
    lines: list[str] = []
    lines.append("Question:")
    lines.append(question.strip())
    lines.append("")
    lines.append("Transcript evidence:")
    if evidence.transcripts:
        for t in evidence.transcripts:
            lines.append(
                f"- [{t.evidence_id}] {t.timestamp_start:.2f}-{t.timestamp_end:.2f}s: {t.text}"
            )
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("Frame evidence:")
    if evidence.frames:
        for f in evidence.frames:
            lines.append(
                f"- [{f.evidence_id}] {f.timestamp_start:.2f}-{f.timestamp_end:.2f}s: {f.frame_path}"
            )
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append(
        "Answer using only the evidence above. If evidence is insufficient, explicitly say what is missing."
    )
    return "\n".join(lines)
