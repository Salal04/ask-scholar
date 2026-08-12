import logging
import secrets
from datetime import datetime, timedelta, timezone
from math import ceil

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_admin
from app.models import Admin, Fiqah, Scholar, ScholarStatus, User, Video
from app.rag.gemini_manager import AllModelsExhaustedError
from app.rag.services import docs_service, embedding_service, youtube_service
from app.schemas import (
    AddDocumentUrlIn,
    AddVideoIn,
    DocumentIngestData,
    Envelope,
    InviteScholarData,
    InviteScholarIn,
    MessageData,
    Pagination,
    ScholarAdminData,
    ScholarAdminListData,
    SetScholarActiveIn,
    UserListData,
    VideoData,
)
from app.security import hash_password
from app.serializers import scholar_admin_out, user_out, video_out

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(get_current_admin)])


def _ingest_video_for_scholar(scholar_id: str, url: str) -> None:
    """
    Best-effort background ingestion: transcribes the video and stores its
    embeddings in the `scholar_id` Pinecone namespace, so
    scholars.ask_question can later retrieve a relevant, timestamped moment
    for that scholar specifically.

    Runs as a FastAPI BackgroundTask (after the response is sent) because
    transcription can take a while — see app/rag/README notes on moving
    this to a real task queue for production. Swallows all errors: if RAG
    isn't configured (no Gemini/Pinecone keys) or the video fails to
    transcribe, the video link itself was already saved successfully and
    should not be affected.
    """
    logger.info("Starting background ingestion for scholar=%s url=%s", scholar_id, url)
    try:
        result = youtube_service.ingest_youtube_video(url)
        logger.info(
            "Transcription complete for scholar=%s url=%s title=%r chunks=%d",
            scholar_id, url, result.get("title"), len(result.get("chunks", [])),
        )

        embedding_service.store_chunks(
            chunks=result["chunks"],
            source_id=result["source_id"],
            source_type="youtube",
            source_url=result["source_url"],
            title=result["title"],
            namespace=scholar_id,
        )
        logger.info(
            "Ingestion succeeded for scholar=%s url=%s source_id=%s",
            scholar_id, url, result.get("source_id"),
        )
    except AllModelsExhaustedError as e:
        logger.error(
            "Background ingestion failed (all Gemini models exhausted) for scholar=%s video=%s: %s",
            scholar_id, url, e,
        )
    except Exception:  # noqa: BLE001 - best-effort background job
        logger.exception(
            "Background ingestion failed for scholar=%s video=%s",
            scholar_id, url,
        )


INVITE_TOKEN_TTL_DAYS = 7


# ------------------------------------------------------------- Scholars ---


@router.post("/scholars", response_model=Envelope[ScholarAdminData], status_code=status.HTTP_201_CREATED)
async def create_scholar_full(
    db: AsyncSession = Depends(get_db),
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(..., min_length=6),
    fiqah: Fiqah | None = Form(default=None),
    bio: str | None = Form(default=None),
    specialization: str | None = Form(default=None),
    qualifications: str | None = Form(default=None),
    yearsOfExperience: int | None = Form(default=None),
    languages: str | None = Form(default=None),  # comma-separated, e.g. "Arabic, English"
    location: str | None = Form(default=None),
    picture: UploadFile | None = File(default=None),
):
    existing = await db.scalar(select(Scholar).where(Scholar.email == email))
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="A scholar with this email already exists.")

    picture_url = None
    if picture is not None and picture.filename:
        from app.storage import upload_scholar_picture

        picture_url = await upload_scholar_picture(picture)

    lang_list = [lang.strip() for lang in languages.split(",") if lang.strip()] if languages else []

    scholar = Scholar(
        name=name,
        email=email,
        password=hash_password(password),
        fiqah=fiqah,
        bio=bio,
        specialization=specialization,
        qualifications=qualifications,
        years_of_experience=yearsOfExperience,
        languages=lang_list,
        location=location,
        picture=picture_url,
        status=ScholarStatus.ACTIVE,
        is_active=True,
    )
    db.add(scholar)
    await db.commit()
    await db.refresh(scholar)
    return Envelope(data=ScholarAdminData(scholar=scholar_admin_out(scholar)))


@router.post("/scholars/invite", response_model=Envelope[InviteScholarData], status_code=status.HTTP_201_CREATED)
async def invite_scholar(payload: InviteScholarIn, db: AsyncSession = Depends(get_db)):
    existing = await db.scalar(select(Scholar).where(Scholar.email == payload.email))
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="A scholar with this email already exists.")

    token = secrets.token_urlsafe(32)
    expiry = datetime.now(timezone.utc) + timedelta(days=INVITE_TOKEN_TTL_DAYS)

    scholar = Scholar(
        name=payload.name,
        email=payload.email,
        fiqah=payload.fiqah,
        status=ScholarStatus.PENDING,
        invite_token=token,
        invite_token_expiry=expiry,
    )
    db.add(scholar)
    await db.commit()
    await db.refresh(scholar)

    return Envelope(data=InviteScholarData(scholar=scholar_admin_out(scholar), inviteToken=token, invitationExpiresAt=expiry))


@router.get("/scholars", response_model=Envelope[ScholarAdminListData])
async def list_scholars_admin(
    db: AsyncSession = Depends(get_db),
    search: str | None = Query(default=None),
    status_filter: ScholarStatus | None = Query(default=None, alias="status"),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=500),
):
    conditions = []
    if search:
        like = f"%{search}%"
        conditions.append(or_(Scholar.name.ilike(like), Scholar.email.ilike(like)))
    if status_filter:
        conditions.append(Scholar.status == status_filter)

    base_query = select(Scholar).where(*conditions) if conditions else select(Scholar)
    total = await db.scalar(select(func.count()).select_from(base_query.subquery()))

    base_query = base_query.order_by(Scholar.created_at.desc()).offset((page - 1) * limit).limit(limit)
    scholars = (await db.scalars(base_query)).all()

    total = total or 0
    pagination = Pagination(page=page, limit=limit, total=total, totalPages=max(1, ceil(total / limit)))
    return Envelope(data=ScholarAdminListData(scholars=[scholar_admin_out(s) for s in scholars], pagination=pagination))


@router.delete("/scholars/{scholar_id}", response_model=Envelope[MessageData])
async def delete_scholar(scholar_id: str, db: AsyncSession = Depends(get_db)):
    scholar = await db.get(Scholar, scholar_id)
    if not scholar:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scholar not found.")
    await db.delete(scholar)
    await db.commit()
    return Envelope(data=MessageData(message="Scholar deleted."))


@router.patch("/scholars/{scholar_id}/status", response_model=Envelope[ScholarAdminData])
async def set_scholar_active_status(scholar_id: str, payload: SetScholarActiveIn, db: AsyncSession = Depends(get_db)):
    scholar = await db.get(Scholar, scholar_id)
    if not scholar:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scholar not found.")
    scholar.is_active = payload.isActive
    db.add(scholar)
    await db.commit()
    await db.refresh(scholar)
    return Envelope(data=ScholarAdminData(scholar=scholar_admin_out(scholar)))


@router.post("/scholars/{scholar_id}/resend-invite", response_model=Envelope[InviteScholarData])
async def resend_scholar_invite(scholar_id: str, db: AsyncSession = Depends(get_db)):
    scholar = await db.get(Scholar, scholar_id)
    if not scholar:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scholar not found.")
    if scholar.status != ScholarStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only scholars with a pending invite can be re-invited.",
        )

    scholar.invite_token = secrets.token_urlsafe(32)
    scholar.invite_token_expiry = datetime.now(timezone.utc) + timedelta(days=INVITE_TOKEN_TTL_DAYS)
    db.add(scholar)
    await db.commit()
    await db.refresh(scholar)

    return Envelope(
        data=InviteScholarData(
            scholar=scholar_admin_out(scholar), inviteToken=scholar.invite_token, invitationExpiresAt=scholar.invite_token_expiry
        )
    )


# --------------------------------------------------------------- Videos ---


@router.post("/videos", response_model=Envelope[VideoData], status_code=status.HTTP_201_CREATED)
async def add_scholar_video(payload: AddVideoIn, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    scholar = await db.get(Scholar, payload.scholarId)
    if not scholar:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scholar not found.")

    video = Video(scholar_id=payload.scholarId, url=payload.url)
    db.add(video)
    await db.commit()
    await db.refresh(video)

    # Kick off RAG ingestion in the background (best-effort — see
    # _ingest_video_for_scholar) so this response doesn't wait on
    # transcription. Once it lands, scholars.ask_question can retrieve
    # timestamped moments from this video for this scholar's questions.
    logger.info("Queuing background ingestion for scholar=%s video=%s", payload.scholarId, payload.url)
    background_tasks.add_task(_ingest_video_for_scholar, payload.scholarId, payload.url)

    return Envelope(data=VideoData(video=video_out(video)))


# ------------------------------------------------------------- Documents ---
#
# Accepts a document (file upload or URL), runs it through the existing
# RAG document pipeline already implemented in app/rag/services/docs_service.py
# (download/extract -> page-aware chunking) and stores the resulting chunks'
# embeddings in Pinecone via app/rag/services/embedding_service.store_chunks
# — the exact same two functions app/rag/router.py's own
# /rag/ingest/document/* endpoints use.
#
# If `scholarId` is given, chunks are stored under that scholar's own
# Pinecone namespace, exactly like _ingest_video_for_scholar does for
# YouTube videos above — so scholars.ask_question (namespace=scholar_id,
# source_type=["youtube", "document"]) can retrieve this document's
# content for that scholar specifically. Without `scholarId`, the document
# lands in the shared/global namespace instead.
#
# Runs synchronously (unlike video ingestion, which is backgrounded because
# transcription is slow): docs_service's own PDF extraction is already
# batched/checkpointed page-by-page, so blocking the request is fine and
# lets the admin see exactly how many chunks were stored right away.


async def _require_scholar_if_given(db: AsyncSession, scholar_id: str | None) -> None:
    if scholar_id:
        scholar = await db.get(Scholar, scholar_id)
        if not scholar:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scholar not found.")


def _run_document_ingestion(
    *,
    scholar_id: str | None,
    url: str | None = None,
    filename: str | None = None,
    content: bytes | None = None,
) -> DocumentIngestData:
    try:
        result = docs_service.ingest_document(url=url, filename=filename, content=content)
        stored = embedding_service.store_chunks(
            chunks=result["chunks"],
            source_id=result["source_id"],
            source_type="document",
            source_url=result["source_url"],
            title=result["title"],
            namespace=scholar_id,
        )
    except AllModelsExhaustedError as e:
        logger.error(
            "Admin document ingestion failed (all Gemini models exhausted) scholar=%s: %s",
            scholar_id, e,
        )
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception("Admin document ingestion failed (scholar=%s url=%s filename=%s)", scholar_id, url, filename)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Document ingestion failed: {e}")

    logger.info(
        "Admin document ingestion complete scholar=%s source_id=%s chunks=%d",
        scholar_id, result.get("source_id"), stored,
    )
    return DocumentIngestData(
        sourceId=result["source_id"],
        sourceType="document",
        sourceUrl=result["source_url"],
        title=result["title"],
        chunksStored=stored,
        scholarId=scholar_id,
    )


@router.post("/documents/upload", response_model=Envelope[DocumentIngestData], status_code=status.HTTP_201_CREATED)
async def upload_admin_document(
    db: AsyncSession = Depends(get_db),
    file: UploadFile = File(...),
    scholarId: str | None = Form(default=None),
):
    """Accepts a direct file upload (PDF / DOCX / TXT), chunks it, and stores it in the vector DB."""
    await _require_scholar_if_given(db, scholarId)

    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A file is required.")
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    data = _run_document_ingestion(scholar_id=scholarId, filename=file.filename, content=content)
    return Envelope(data=data)


@router.post("/documents/url", response_model=Envelope[DocumentIngestData], status_code=status.HTTP_201_CREATED)
async def ingest_admin_document_url(payload: AddDocumentUrlIn, db: AsyncSession = Depends(get_db)):
    """Accepts a document URL (PDF / DOCX / TXT / Google Doc share link), chunks it, and stores it in the vector DB."""
    await _require_scholar_if_given(db, payload.scholarId)

    data = _run_document_ingestion(scholar_id=payload.scholarId, url=payload.url)
    return Envelope(data=data)


# ---------------------------------------------------------------- Users ---


@router.get("/users", response_model=Envelope[UserListData])
async def list_users_admin(
    db: AsyncSession = Depends(get_db),
    search: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=500),
):
    conditions = []
    if search:
        like = f"%{search}%"
        conditions.append(or_(User.name.ilike(like), User.email.ilike(like)))

    base_query = select(User).where(*conditions) if conditions else select(User)
    total = await db.scalar(select(func.count()).select_from(base_query.subquery()))

    base_query = base_query.order_by(User.created_at.desc()).offset((page - 1) * limit).limit(limit)
    users = (await db.scalars(base_query)).all()

    total = total or 0
    pagination = Pagination(page=page, limit=limit, total=total, totalPages=max(1, ceil(total / limit)))
    return Envelope(data=UserListData(users=[user_out(u) for u in users], pagination=pagination))


@router.delete("/users/{user_id}", response_model=Envelope[MessageData])
async def delete_user(user_id: str, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    await db.delete(user)
    await db.commit()
    return Envelope(data=MessageData(message="User deleted."))