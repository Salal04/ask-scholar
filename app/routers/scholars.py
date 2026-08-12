import asyncio
import re
from datetime import datetime, timezone
from math import ceil

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user
from app.models import Fiqah, Scholar, ScholarStatus, User, Video
from app.rag import config as rag_config
from app.rag.services import answer_service, embedding_service
from app.serializers import scholar_public_out
from app.schemas import (
    AskQuestionData,
    AskQuestionIn,
    Citation,
    Envelope,
    Pagination,
    ScholarDetailData,
    ScholarListData,
    VideoRef,
)

router = APIRouter(prefix="/scholars", tags=["scholars"])


# --------------------------------------------------------------------------
# Greetings ("assalam o alaikum", "hi", ...) get a plain, friendly reply
# instead of the full RAG pipeline — there's no question for a content
# summary (or its disclaimer) to attach to. Only matches when the WHOLE
# message is just a greeting; a real question that happens to start with
# "salam, ..." still goes through the normal flow below.
# --------------------------------------------------------------------------
_GREETING_PATTERNS = [
    r"^(as)?salam(u|o)?\s*(o\s*)?alaikum(\s*wa\s*rahmatullah(\s*wa\s*barakatuh)?)?$",
    r"^(wa)?alaikum\s*(as)?salam(\s*wa\s*rahmatullah(\s*wa\s*barakatuh)?)?$",
    r"^assalamualaikum$",
    r"^waalaikumsalam$",
    r"^salam$",
    r"^hi+$",
    r"^he(l+o+|y+)$",
    r"^hello+$",
    r"^(good\s*)?(morning|evening|afternoon)$",
    r"^السلام عليكم(\s*ورحمة الله(\s*وبركاته)?)?$",
    r"^وعليكم السلام(\s*ورحمة الله(\s*وبركاته)?)?$",
]
_GREETING_RE = re.compile("|".join(_GREETING_PATTERNS), re.IGNORECASE)


def _is_pure_greeting(text: str) -> bool:
    """True only when the message is nothing more than a greeting — no
    question mark and not much more than the greeting itself. A message
    like "salam, can you tell me about X?" is a real question and must
    NOT be treated as a greeting."""
    normalized = (text or "").strip()
    if not normalized or "?" in normalized:
        return False
    normalized = re.sub(r"[!.,؟]+$", "", normalized).strip()
    if len(normalized.split()) > 6:
        return False
    return bool(_GREETING_RE.match(normalized))


def _greeting_reply(scholar_name: str | None) -> str:
    name = scholar_name or "the scholar"
    return (
        f"Wa alaikum assalam! Welcome. Feel free to ask your question and "
        f"I'll share what {name}'s own videos and documents say about it."
    )


# Every RAG-generated answer carries this disclaimer instead of the old
# "it has been received and will be reviewed by {scholar}" wording — that
# line implied a human review step that doesn't actually happen here.
# This is honest about what's really going on: the reply is an automated
# summary of the scholar's own content, not the scholar personally
# answering.
def _automated_disclaimer(scholar_name: str | None) -> str:
    name = scholar_name or "the scholar"
    return (
        f"Note: this is an automated explanation generated from {name}'s own "
        f"content (videos/documents) — {name} has not personally written or "
        f"reviewed this reply."
    )


async def _rag_answer_for_scholar(
    scholar_id: str,
    question: str,
    history: str | None,
    preferred_language: str | None,
):
    """
    Retrieves the scholar's own ingested chunks (see admin.add_scholar_video
    and the /rag/ingest/document/* endpoints, both of which ingest into the
    `scholar_id` Pinecone namespace) — youtube AND document chunks together,
    ranked by relevance — then hands them + the question/history to Gemini
    to compose one grounded, already-translated answer. The model itself
    decides which of the retrieved chunks (if any, possibly several
    different videos/documents at once) are actually relevant enough to
    cite; see answer_service.generate_answer.

    Best-effort only: RAG is optional infrastructure (needs
    GEMINI_API_KEY_*/PINECONE_API_KEY configured, and at least one of the
    scholar's videos/documents to have finished ingesting), so any failure
    here just means we fall back to the plain acknowledgement — it must
    never break askQuestion itself.
    """
    try:
        matches = await asyncio.to_thread(
            embedding_service.query_similar,
            query=question,
            top_k=rag_config.ANSWER_RETRIEVAL_TOP_K,
            namespace=scholar_id,
            source_type=["youtube", "document"],
        )
        if not matches:
            return None
        return await asyncio.to_thread(
            answer_service.generate_answer,
            question=question,
            chunks=matches,
            history=history,
            target_language=preferred_language,
        )
    except Exception:
        return None


def _visible_scholars_clause():
    """Only scholars who have finished onboarding and aren't suspended show up publicly."""
    return (Scholar.status == ScholarStatus.ACTIVE) & (Scholar.is_active.is_(True))


@router.get("", response_model=Envelope[ScholarListData])
async def browse_scholars(
    db: AsyncSession = Depends(get_db),
    search: str | None = Query(default=None),
    fiqah: Fiqah | None = Query(default=None),
    location: str | None = Query(default=None),
    language: str | None = Query(default=None),
    sort: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=12, ge=1, le=100),
):
    conditions = [_visible_scholars_clause()]

    if search:
        like = f"%{search}%"
        conditions.append(or_(Scholar.name.ilike(like), Scholar.specialization.ilike(like)))
    if fiqah:
        conditions.append(Scholar.fiqah == fiqah)
    if location:
        conditions.append(Scholar.location.ilike(f"%{location}%"))
    if language:
        conditions.append(Scholar.languages.any(language))

    base_query = select(Scholar).where(*conditions)

    total = await db.scalar(select(func.count()).select_from(base_query.subquery()))

    if sort == "popularity":
        base_query = base_query.order_by(Scholar.popularity_score.desc())
    else:
        base_query = base_query.order_by(Scholar.created_at.desc())

    base_query = base_query.offset((page - 1) * limit).limit(limit)
    scholars = (await db.scalars(base_query)).all()

    total = total or 0
    pagination = Pagination(page=page, limit=limit, total=total, totalPages=max(1, ceil(total / limit)))
    return Envelope(data=ScholarListData(scholars=[scholar_public_out(s) for s in scholars], pagination=pagination))


@router.get("/{scholar_id}", response_model=Envelope[ScholarDetailData])
async def get_scholar(scholar_id: str, db: AsyncSession = Depends(get_db)):
    scholar = await db.get(Scholar, scholar_id)
    if not scholar or not (scholar.status == ScholarStatus.ACTIVE and scholar.is_active):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scholar not found.")
    return Envelope(data=ScholarDetailData(scholar=scholar_public_out(scholar)))


@router.post("/askQuestion/{scholar_id}", response_model=Envelope[AskQuestionData])
async def ask_question(
    scholar_id: str,
    payload: AskQuestionIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    scholar = await db.get(Scholar, scholar_id)
    if not scholar or not (scholar.status == ScholarStatus.ACTIVE and scholar.is_active):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scholar not found.")

    # A plain greeting isn't a question — reply simply and skip the RAG
    # pipeline (and its disclaimer, which has nothing to attach to here).
    if _is_pure_greeting(payload.question):
        return Envelope(
            data=AskQuestionData(
                answer=_greeting_reply(scholar.name),
                video=None,
                citations=[],
                askedAt=datetime.now(timezone.utc),
            )
        )

    # This still does NOT auto-generate a religious ruling — per the
    # frontend README that's intentional. What it does is retrieve the
    # scholar's own ingested videos AND documents (RAG search, scoped to
    # this scholar's namespace) and have the LLM summarize whatever
    # relevant passages it finds — across possibly several videos/docs —
    # translated into the asker's preferred language, so the reply can
    # point them to specific moments/pages instead of just the latest
    # video. See _automated_disclaimer: this is always presented as an
    # automated, content-derived explanation, never as the scholar
    # personally replying.
    scholar.popularity_score += 1
    db.add(scholar)
    await db.commit()

    rag_result = await _rag_answer_for_scholar(
        scholar_id=scholar_id,
        question=payload.question,
        history=payload.history,
        preferred_language=payload.preferredLanguage,
    )

    citations: list[Citation] = []
    video_ref = None

    if rag_result and rag_result.get("answer"):
        citations = [
            Citation(
                type=c["type"],
                title=c.get("title"),
                url=c.get("url"),
                startSeconds=c.get("start_seconds"),
                endSeconds=c.get("end_seconds"),
                page=c.get("page_number"),
            )
            for c in rag_result.get("citations", [])
        ]
        # Backward compat for existing frontend consumers of `video`: the
        # first video citation, if the LLM cited one.
        video_ref = next(
            (
                VideoRef(url=c.url, startSeconds=c.startSeconds or 0, endSeconds=c.endSeconds)
                for c in citations
                if c.type == "youtube"
            ),
            None,
        )
        answer = f"{_automated_disclaimer(scholar.name)}\n\n{rag_result['answer']}"
    else:
        answer = (
            f"{_automated_disclaimer(scholar.name)} No relevant information was found "
            f"in their existing videos/documents for this question yet."
        )
        latest_video = await db.scalar(
            select(Video).where(Video.scholar_id == scholar_id).order_by(Video.created_at.desc()).limit(1)
        )
        if latest_video:
            # No RAG match (not configured, nothing ingested yet, or
            # nothing relevant found) — fall back to the old placeholder
            # reference, starting at 0s.
            video_ref = VideoRef(url=latest_video.url, startSeconds=0, endSeconds=None)

    return Envelope(
        data=AskQuestionData(
            answer=answer, video=video_ref, citations=citations, askedAt=datetime.now(timezone.utc)
        )
    )