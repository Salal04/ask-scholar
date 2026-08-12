from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Admin, User
from app.schemas import (
    AdminLoginIn,
    AuthAdminData,
    AuthUserData,
    Envelope,
    UserLoginIn,
    UserRegisterIn,
)
from app.security import create_access_token, hash_password, verify_password
from app.serializers import admin_out, user_out

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/user/register", response_model=Envelope[AuthUserData], status_code=status.HTTP_201_CREATED)
async def register_user(payload: UserRegisterIn, db: AsyncSession = Depends(get_db)):
    existing = await db.scalar(select(User).where(User.email == payload.email))
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists.")

    user = User(name=payload.name, email=payload.email, password=hash_password(payload.password))
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_access_token(subject=user.id, role="USER")
    return Envelope(data=AuthUserData(user=user_out(user), token=token))


@router.post("/user/login", response_model=Envelope[AuthUserData])
async def login_user(payload: UserLoginIn, db: AsyncSession = Depends(get_db)):
    user = await db.scalar(select(User).where(User.email == payload.email))
    if not user or not verify_password(payload.password, user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This account has been disabled.")

    token = create_access_token(subject=user.id, role="USER")
    return Envelope(data=AuthUserData(user=user_out(user), token=token))


@router.post("/admin/login", response_model=Envelope[AuthAdminData])
async def login_admin(payload: AdminLoginIn, db: AsyncSession = Depends(get_db)):
    admin = await db.scalar(select(Admin).where(Admin.email == payload.email))
    if not admin or not verify_password(payload.password, admin.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")

    token = create_access_token(subject=admin.id, role="ADMIN")
    return Envelope(data=AuthAdminData(admin=admin_out(admin), token=token))
