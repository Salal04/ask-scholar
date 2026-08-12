"""
Retrieval-augmented answer generation for `/askQuestion/{scholar_id}`.

Input: the user's question, optional free-text conversation history, and a
list of already-retrieved chunks (embedding_service.query_similar) — a mix
of YouTube-transcript chunks and document chunks, possibly from several
different videos/documents.

Output: one grounded answer, already translated into the caller's preferred
language, plus a list of the specific chunks the model actually relied on.

Design note on citations: the LLM is only ever asked to return the INDEX of
a chunk (its position in the numbered list we hand it in the prompt), never
asked to reproduce a URL, timestamp, page number, or title itself. We look
those up ourselves from the chunk metadata we already have. This means a
citation can never point to a hallucinated link/page — worst case the model
picks an irrelevant index, but it can't invent one that doesn't exist. The
model itself decides which chunks (if any) matter enough to cite, and can
cite several distinct videos and/or documents at once.
"""
import logging
from typing import Dict, List, Optional

from pydantic import BaseModel

from ..gemini_manager import key_manager
from . import json_utils

logger = logging.getLogger(__name__)


class AnswerResult(BaseModel):
    has_relevant_information: bool
    answer: str
    citations: List[int]

ANSWER_PROMPT_TEMPLATE = """\
You are helping answer a question submitted to an Islamic scholar's Q&A \
platform, using ONLY the numbered source excerpts below (pulled from that \
scholar's own videos and documents). These excerpts are your entire body \
of knowledge for this task — do not use outside knowledge.

Critical rule: you must NEVER issue a religious ruling (fatwa) or state \
your own religious opinion. Your only job is to neutrally summarize what \
the provided excerpts say that is relevant to the question, so the person \
has something useful to read while they wait for the scholar to personally \
review and answer. If the excerpts don't actually address the question, \
say so plainly instead of guessing or filling gaps with general knowledge.

Conversation history so far (may be empty):
{history}

User's question:
{question}

Numbered source excerpts:
{sources}

Reply with ONLY a JSON object (no markdown fences, no commentary), shaped \
exactly like this:
{{
  "has_relevant_information": <true or false>,
  "answer": "<your summary of what the relevant excerpts say, written in \
{language}. Empty string if has_relevant_information is false.>",
  "citations": [<index numbers (integers) of the excerpts your answer \
actually draws on, most relevant first. Empty list if none.>]
}}

Rules for "answer":
- Write it entirely in {language}, regardless of what language the source \
excerpts or the question were in.
- Only include information actually present in the cited excerpts.
- Do not mention "excerpt", "source", "chunk", or index numbers in the \
answer text itself — citations are returned separately in "citations".
- Keep it concise and directly relevant to the question.
"""


def _describe_chunk(chunk: Dict) -> str:
    """One-line human-readable locator for a chunk, used only inside the
    prompt so the model has some sense of where each excerpt is from —
    it never has to repeat this back to us."""
    if chunk.get("source_type") == "youtube":
        loc = f'YouTube video "{chunk.get("title") or chunk.get("source_url")}"'
        if chunk.get("start_time") is not None:
            loc += f", around {int(chunk['start_time'])}s"
    else:
        loc = f'document "{chunk.get("title") or chunk.get("source_url")}"'
        if chunk.get("page_number") is not None:
            loc += f", page {chunk['page_number']}"
    return loc


def _format_sources(chunks: List[Dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks):
        blocks.append(f"[{i}] ({_describe_chunk(chunk)})\n{chunk.get('text', '')}")
    return "\n\n".join(blocks)


def _chunk_to_citation(chunk: Dict) -> Dict:
    """Build the citation the router hands back to the client, sourced
    entirely from our own retrieved metadata — never from anything the LLM
    said. Documents just need a name; videos need url + timestamp."""
    citation = {
        "type": chunk.get("source_type"),
        "title": chunk.get("title") or None,
        "url": chunk.get("source_url"),
    }
    if chunk.get("source_type") == "youtube":
        citation["start_seconds"] = int(chunk["start_time"]) if chunk.get("start_time") is not None else 0
        citation["end_seconds"] = int(chunk["end_time"]) if chunk.get("end_time") is not None else None
        citation["page_number"] = None
    else:
        citation["start_seconds"] = None
        citation["end_seconds"] = None
        citation["page_number"] = chunk.get("page_number")
    return citation


def generate_answer(
    question: str,
    chunks: List[Dict],
    history: Optional[str] = None,
    target_language: Optional[str] = None,
) -> Dict:
    """
    Returns: {"answer": str | None, "citations": [<citation dict>, ...]}
    "answer" is None (and citations empty) when there were no chunks to
    work with, or the model judged none of them relevant — callers should
    treat that the same as "no RAG match" and fall back accordingly.
    """
    logger.info(
        "generate_answer called: question_len=%d, %d chunk(s), target_language=%s",
        len(question), len(chunks), target_language or "(same as question)",
    )

    if not chunks:
        logger.info("No chunks provided, skipping LLM call and returning no-answer")
        return {"answer": None, "citations": []}

    prompt = ANSWER_PROMPT_TEMPLATE.format(
        history=history or "(none)",
        question=question,
        sources=_format_sources(chunks),
        language=target_language or "the same language the question was asked in",
    )

    def fn(client, model):
        logger.info("Requesting answer generation from model=%s", model)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            # response_schema constrains generation to this exact shape
            # (not just "please output JSON"), which is what actually
            # fixes "model didn't return JSON" — the mime type alone was
            # a request, this is an enforced constraint. loads_relaxed()
            # is still the fallback for whatever still slips through.
            config=json_utils.build_json_config(schema=AnswerResult),
        )

        parsed = json_utils.extract_parsed(response)
        if parsed is not None:
            return parsed.model_dump()

        raw = (response.text or "").strip()
        data = json_utils.loads_relaxed(raw)
        # Normalize loosely-shaped dicts through the same pydantic model
        # so downstream code always sees the expected keys/types.
        return AnswerResult.model_validate(data).model_dump()

    try:
        result = key_manager.call("answer", fn)
    except Exception:
        logger.exception("Answer generation failed for question (len=%d)", len(question))
        raise

    if not result.get("has_relevant_information") or not (result.get("answer") or "").strip():
        logger.info("Model judged no relevant information among %d chunk(s)", len(chunks))
        return {"answer": None, "citations": []}

    citations = []
    seen_indices = set()
    for raw_idx in result.get("citations", []):
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            logger.warning("Ignoring non-integer citation index from model: %r", raw_idx)
            continue
        if idx in seen_indices or not (0 <= idx < len(chunks)):
            logger.warning("Ignoring out-of-range or duplicate citation index: %d", idx)
            continue
        seen_indices.add(idx)
        citations.append(_chunk_to_citation(chunks[idx]))

    logger.info(
        "generate_answer succeeded: answer_len=%d, %d citation(s)",
        len(result["answer"].strip()), len(citations),
    )
    return {"answer": result["answer"].strip(), "citations": citations}