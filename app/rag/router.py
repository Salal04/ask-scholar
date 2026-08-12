"""
Same three endpoints the standalone RAG backend exposed
(`/ingest/youtube`, `/ingest/document/url`, `/ingest/document/upload`,
`/search`), now as an APIRouter mounted at `/api/rag` in the merged app
instead of running as its own FastAPI service.

Ingestion and search are admin-only (see `app.deps.get_current_admin`) —
the standalone RAG backend had no auth of its own, which was fine as an
internal service but not once it's mounted on the public API surface.
`/health` stays open since it doesn't expose or accept anything sensitive.
"""
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.deps import get_current_admin

from .gemini_manager import AllModelsExhaustedError
from .schemas import (
    DocumentUrlIngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    YoutubeIngestRequest,
)
from .services import docs_service, embedding_service, youtube_service

router = APIRouter(prefix="/rag", tags=["rag"])


@router.get("/health")
def rag_health():
    return {"status": "ok"}


# --------------------------------------------------------------------------
# YouTube ingestion
# --------------------------------------------------------------------------
@router.post("/ingest/youtube", response_model=IngestResponse)
def ingest_youtube(req: YoutubeIngestRequest, _admin=Depends(get_current_admin)):
    try:
        print("Youtube Video is Recivied ==== " , req.url);
        result = youtube_service.ingest_youtube_video(req.url)
        stored = embedding_service.store_chunks(
            chunks=result["chunks"],
            source_id=result["source_id"],
            source_type="youtube",
            source_url=result["source_url"],
            title=result["title"],
            namespace=req.namespace,
        )
        return IngestResponse(
            source_id=result["source_id"],
            source_type="youtube",
            source_url=result["source_url"],
            chunks_stored=stored,
        )
    except AllModelsExhaustedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YouTube ingestion failed: {e}")


# --------------------------------------------------------------------------
# Document ingestion — by URL
# --------------------------------------------------------------------------
@router.post("/ingest/document/url", response_model=IngestResponse)
def ingest_document_url(req: DocumentUrlIngestRequest, _admin=Depends(get_current_admin)):
    try:
        result = docs_service.ingest_document(url=req.url)
        stored = embedding_service.store_chunks(
            chunks=result["chunks"],
            source_id=result["source_id"],
            source_type="document",
            source_url=result["source_url"],
            title=result["title"],
            namespace=req.namespace,
        )
        return IngestResponse(
            source_id=result["source_id"],
            source_type="document",
            source_url=result["source_url"],
            chunks_stored=stored,
        )
    except AllModelsExhaustedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {e}")


# --------------------------------------------------------------------------
# Document ingestion — direct file upload
# --------------------------------------------------------------------------
@router.post("/ingest/document/upload", response_model=IngestResponse)
def ingest_document_upload(
    file: UploadFile = File(...),
    namespace: Optional[str] = Form(None),
    _admin=Depends(get_current_admin),
):
    try:
        content = file.file.read()
        result = docs_service.ingest_document(filename=file.filename, content=content)
        stored = embedding_service.store_chunks(
            chunks=result["chunks"],
            source_id=result["source_id"],
            source_type="document",
            source_url=result["source_url"],
            title=result["title"],
            namespace=namespace,
        )
        return IngestResponse(
            source_id=result["source_id"],
            source_type="document",
            source_url=result["source_url"],
            chunks_stored=stored,
        )
    except AllModelsExhaustedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {e}")


# --------------------------------------------------------------------------
# Similarity search
# --------------------------------------------------------------------------
@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, _admin=Depends(get_current_admin)):
    try:
        matches = embedding_service.query_similar(
            query=req.query,
            top_k=req.top_k,
            namespace=req.namespace,
            source_type=req.source_type,
        )
        return SearchResponse(query=req.query, matches=matches)
    except AllModelsExhaustedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")
