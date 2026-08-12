from typing import List, Optional, Union

from pydantic import BaseModel, Field


class YoutubeIngestRequest(BaseModel):
    url: str = Field(..., description="Full YouTube video URL")
    namespace: Optional[str] = Field(None, description="Optional Pinecone namespace")


class DocumentUrlIngestRequest(BaseModel):
    url: str = Field(..., description="URL of a PDF / DOCX / TXT / Google Doc")
    namespace: Optional[str] = Field(None, description="Optional Pinecone namespace")


class IngestResponse(BaseModel):
    source_id: str
    source_type: str
    source_url: str
    chunks_stored: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    namespace: Optional[str] = None
    source_type: Optional[Union[str, List[str]]] = Field(
        None, description="Filter results to 'youtube' and/or 'document'"
    )


class SearchMatch(BaseModel):
    score: float
    text: str
    source_id: Optional[str] = None
    source_type: str
    source_url: str
    title: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None


class SearchResponse(BaseModel):
    query: str
    matches: List[SearchMatch]