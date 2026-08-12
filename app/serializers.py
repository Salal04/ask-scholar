from app.models import Admin, Scholar, User, Video
from app.schemas import AdminOut, ScholarAdminOut, ScholarPublicOut, UserOut, VideoOut


def user_out(u: User) -> UserOut:
    return UserOut(id=u.id, name=u.name, email=u.email, isActive=u.is_active, createdAt=u.created_at)


def admin_out(a: Admin) -> AdminOut:
    return AdminOut(id=a.id, name=a.name, email=a.email, createdAt=a.created_at)


def scholar_public_out(s: Scholar) -> ScholarPublicOut:
    return ScholarPublicOut(
        id=s.id,
        name=s.name,
        email=s.email,
        fiqah=s.fiqah,
        picture=s.picture,
        bio=s.bio,
        specialization=s.specialization,
        qualifications=s.qualifications,
        yearsOfExperience=s.years_of_experience,
        languages=s.languages or [],
        location=s.location,
        isVerified=s.is_verified,
        popularityScore=s.popularity_score,
        status=s.status,
        isActive=s.is_active,
        createdAt=s.created_at,
    )


def scholar_admin_out(s: Scholar) -> ScholarAdminOut:
    base = scholar_public_out(s)
    return ScholarAdminOut(**base.model_dump(), inviteTokenExpiry=s.invite_token_expiry)


def video_out(v: Video) -> VideoOut:
    return VideoOut(id=v.id, scholarId=v.scholar_id, url=v.url, createdAt=v.created_at)
