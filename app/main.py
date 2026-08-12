from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import Base, engine
from app.rag.router import router as rag_router
from app.routers import admin, auth, scholars
from app.seed import seed_admin
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Creates any tables that don't exist yet. For a production setup you'd
    # normally swap this for Alembic migrations, but this keeps the schema
    # in sync with app/models.py with zero extra tooling to run.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await seed_admin()
    yield


app = FastAPI(title="Ask Scholar API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# The frontend's axios client reads `err.response?.data?.message`, so every
# error response (whatever raised it) is normalized to a `message` field.
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"success": False, "message": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    first_error = exc.errors()[0] if exc.errors() else None
    field = ".".join(str(loc) for loc in first_error["loc"] if loc != "body") if first_error else None
    message = f"{field}: {first_error['msg']}" if first_error and field else "Invalid request data."
    return JSONResponse(status_code=422, content={"success": False, "message": message})


@app.get("/api/health")
async def health():
    return {"success": True, "data": {"status": "ok"}}


app.include_router(auth.router, prefix="/api")
app.include_router(scholars.router, prefix="/api")
app.include_router(admin.router, prefix="/api")
# Ingestion (/api/rag/ingest/*) + similarity search (/api/rag/search), merged
# in from the standalone RAG backend. See app/rag/ for the implementation.
app.include_router(rag_router, prefix="/api")
