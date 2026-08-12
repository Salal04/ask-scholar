from sqlalchemy import select

from app.config import settings
from app.database import AsyncSessionLocal
from app.models import Admin
from app.security import hash_password


async def seed_admin() -> None:
    """One admin account is auto-seeded on first server start (see schema.prisma)."""
    async with AsyncSessionLocal() as db:
        existing = await db.scalar(select(Admin))
        if existing:
            return

        admin = Admin(
            name=settings.admin_name,
            email=settings.admin_email,
            password=hash_password(settings.admin_password),
        )
        db.add(admin)
        await db.commit()
