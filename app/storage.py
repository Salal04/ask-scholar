import uuid

import httpx
from fastapi import HTTPException, UploadFile, status

from app.config import settings

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 5 * 1024 * 1024  # 5MB


async def upload_scholar_picture(file: UploadFile) -> str:
    """
    Uploads a scholar's profile picture to Supabase Storage and returns its
    public URL. Requires SUPABASE_URL + SUPABASE_SERVICE_KEY to be set, and
    a public bucket named settings.supabase_storage_bucket to already exist
    (create it once from the Supabase dashboard: Storage -> New bucket ->
    "scholar-pictures", toggle Public).
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Picture must be a JPEG, PNG, or WEBP image.",
        )

    contents = await file.read()
    if len(contents) > MAX_BYTES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Picture must be under 5MB.")

    if not settings.supabase_url or not settings.supabase_service_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Image storage is not configured (missing SUPABASE_URL / SUPABASE_SERVICE_KEY).",
        )

    ext = (file.filename or "").rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else "jpg"
    object_path = f"{uuid.uuid4()}.{ext}"

    upload_url = (
        f"{settings.supabase_url}/storage/v1/object/"
        f"{settings.supabase_storage_bucket}/{object_path}"
    )

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            upload_url,
            content=contents,
            headers={
                "Authorization": f"Bearer {settings.supabase_service_key}",
                "apikey": settings.supabase_service_key,
                "Content-Type": file.content_type,
                "x-upsert": "true",
            },
        )

    if resp.status_code not in (200, 201):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to upload picture to storage: {resp.text}",
        )

    return (
        f"{settings.supabase_url}/storage/v1/object/public/"
        f"{settings.supabase_storage_bucket}/{object_path}"
    )
