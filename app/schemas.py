from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, EmailStr, Field

from app.models import Fiqah, ScholarStatus

T = TypeVar("T")


class Envelope(BaseModel, Generic[T]):
    success: bool = True
    data: T
    message: str | None = None


# ---------------------------------------------------------------- Auth ----


class UserRegisterIn(BaseModel):
    name: str = Field(min_length=1)
    email: EmailStr
    password: str = Field(min_length=6)


class UserLoginIn(BaseModel):
    email: EmailStr
    password: str


class AdminLoginIn(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    name: str
    email: EmailStr
    isActive: bool
    createdAt: datetime

    model_config = {"from_attributes": True}


class AdminOut(BaseModel):
    id: str
    name: str
    email: EmailStr
    createdAt: datetime

    model_config = {"from_attributes": True}


class AuthUserData(BaseModel):
    user: UserOut
    token: str


class AuthAdminData(BaseModel):
    admin: AdminOut
    token: str


# ------------------------------------------------------------ Scholars ----


class ScholarPublicOut(BaseModel):
    id: str
    name: str | None
    email: EmailStr
    fiqah: Fiqah | None
    picture: str | None
    bio: str | None
    specialization: str | None
    qualifications: str | None
    yearsOfExperience: int | None
    languages: list[str]
    location: str | None
    isVerified: bool
    popularityScore: int
    status: ScholarStatus
    isActive: bool
    createdAt: datetime

    model_config = {"from_attributes": True}


class ScholarAdminOut(ScholarPublicOut):
    inviteTokenExpiry: datetime | None = None


class Pagination(BaseModel):
    page: int
    limit: int
    total: int
    totalPages: int


class ScholarListData(BaseModel):
    scholars: list[ScholarPublicOut]
    pagination: Pagination


class ScholarAdminListData(BaseModel):
    scholars: list[ScholarAdminOut]
    pagination: Pagination


class ScholarDetailData(BaseModel):
    scholar: ScholarPublicOut


class InviteScholarIn(BaseModel):
    email: EmailStr
    name: str | None = None
    fiqah: Fiqah | None = None


class InviteScholarData(BaseModel):
    scholar: ScholarAdminOut
    inviteToken: str
    invitationExpiresAt: datetime


class SetScholarActiveIn(BaseModel):
    isActive: bool


class ScholarAdminData(BaseModel):
    scholar: ScholarAdminOut


# ------------------------------------------------------------- Videos -----


class AddVideoIn(BaseModel):
    scholarId: str
    url: str


class VideoOut(BaseModel):
    id: str
    scholarId: str
    url: str
    createdAt: datetime

    model_config = {"from_attributes": True}


class VideoData(BaseModel):
    video: VideoOut


# ------------------------------------------------------------ Documents ---


class AddDocumentUrlIn(BaseModel):
    url: str = Field(..., description="URL of a PDF / DOCX / TXT / Google Doc")
    scholarId: str | None = Field(
        default=None,
        description="If set, the document is stored in this scholar's own knowledge base "
        "(Pinecone namespace) so askQuestion can retrieve it for that scholar. "
        "If omitted, the document is stored in the shared/global namespace.",
    )


class DocumentIngestData(BaseModel):
    sourceId: str
    sourceType: str
    sourceUrl: str
    title: str
    chunksStored: int
    scholarId: str | None = None


# --------------------------------------------------------------- Chat -----


class AskQuestionIn(BaseModel):
    question: str = Field(min_length=1)
    history: str | None = None
    conversationId: str | None = None
    conversationName: str | None = None
    # Language the RAG-generated answer should be written in (e.g.
    # "Urdu", "English"). Optional — when omitted, the answer is written
    # in whatever language the question itself was asked in.
    preferredLanguage: str | None = None


class VideoRef(BaseModel):
    url: str
    startSeconds: int | None = None
    endSeconds: int | None = None


class Citation(BaseModel):
    """
    One source the RAG answer actually drew on. For a video, url +
    startSeconds/endSeconds point at the exact moment; for a document,
    the name (title) is enough, with page identifying where in it.
    """
    type: Literal["youtube", "document"]
    title: str | None = None
    url: str | None = None
    startSeconds: int | None = None
    endSeconds: int | None = None
    page: int | None = None


class AskQuestionData(BaseModel):
    answer: str
    # Kept for existing frontend consumers: the first video citation, if
    # any. New clients should prefer `citations`, which covers documents
    # too and isn't limited to a single source.
    video: VideoRef | None = None
    citations: list[Citation] = Field(default_factory=list)
    askedAt: datetime


# ---------------------------------------------------------------- Users ---


class UserAdminOut(UserOut):
    pass


class UserListData(BaseModel):
    users: list[UserAdminOut]
    pagination: Pagination


class MessageData(BaseModel):
    message: str