import enum
import uuid
from datetime import datetime

from sqlalchemy import ARRAY, Boolean, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


class Fiqah(str, enum.Enum):
    HANAFI = "HANAFI"
    SHAFI = "SHAFI"
    MALIKI = "MALIKI"
    HANBALI = "HANBALI"
    JAFARI = "JAFARI"
    Indipendent = "Indipendent"
    OTHER = "OTHER"


class ScholarStatus(str, enum.Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"


class Admin(Base):
    __tablename__ = "admins"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    password: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    password: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )


class Scholar(Base):
    __tablename__ = "scholars"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    password: Mapped[str | None] = mapped_column(String, nullable=True)

    fiqah: Mapped[Fiqah | None] = mapped_column(Enum(Fiqah, name="fiqah"), nullable=True, index=True)
    picture: Mapped[str | None] = mapped_column(String, nullable=True)
    bio: Mapped[str | None] = mapped_column(Text, nullable=True)
    specialization: Mapped[str | None] = mapped_column(String, nullable=True)
    qualifications: Mapped[str | None] = mapped_column(Text, nullable=True)
    years_of_experience: Mapped[int | None] = mapped_column(Integer, nullable=True)
    languages: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    popularity_score: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[ScholarStatus] = mapped_column(
        Enum(ScholarStatus, name="scholar_status"), default=ScholarStatus.PENDING, index=True
    )
    invite_token: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    invite_token_expiry: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_by_admin_id: Mapped[str | None] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    videos: Mapped[list["Video"]] = relationship(back_populates="scholar", cascade="all, delete-orphan")


# NOT part of the original schema.prisma you supplied — the frontend's
# admin "Add video" tab and the chat panel's YouTube timestamp link both
# need somewhere to persist/read a scholar's linked YouTube videos, so this
# table was added to support that part of the UI.
class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    scholar_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("scholars.id", ondelete="CASCADE"), nullable=False, index=True
    )
    url: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    scholar: Mapped["Scholar"] = relationship(back_populates="videos")
