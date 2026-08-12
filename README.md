# Ask Scholar — Backend (FastAPI + Supabase Postgres + RAG)

A single FastAPI service combining:

- **The Ask Scholar API** — auth, scholar browsing, admin CRUD, the
  `askQuestion` chat endpoint — backed by Supabase Postgres.
- **The RAG pipeline** — YouTube/document ingestion, Gemini embeddings,
  and Pinecone similarity search — mounted under `/api/rag/*` and now also
  wired directly into the scholar flows (see **Integration** below).

Previously these were two separate backends; this merges them into one
codebase and one running process.

## 1. Create your Supabase project

1. Go to [supabase.com](https://supabase.com) → New project.
2. **Database connection string**: Project Settings → Database →
   "Connection string" → URI. Use the **Transaction pooler** (port `6543`)
   string for normal use. Copy it into `DATABASE_URL` in your `.env`, but:
   - change the scheme from `postgresql://` to `postgresql+asyncpg://`
   - fill in your real database password
3. **Storage (for scholar profile pictures)**: Storage → New bucket →
   name it `scholar-pictures` → toggle **Public**. Then copy the
   **Project URL** and **service_role key** into `SUPABASE_URL` /
   `SUPABASE_SERVICE_KEY`.

If you don't need picture uploads yet, leave the Supabase Storage vars
unset — everything else works fine.

## 2. (Optional) Set up RAG

RAG (video/document ingestion + search) is optional — the API starts and
runs fine without it, it just no-ops. To enable it:

1. **System dependency**: install `ffmpeg` (required by `yt-dlp`):
   ```bash
   sudo apt-get install -y ffmpeg   # Debian/Ubuntu
   brew install ffmpeg              # macOS
   ```
2. Fill in `GEMINI_API_KEY_1/2/3` (up to 3 keys for rotation) and
   `PINECONE_API_KEY` / `PINECONE_INDEX_NAME` in `.env`. The Pinecone index
   is auto-created on first use if it doesn't exist.

## 3. Install & run

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env — see sections 1 and 2 above

uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

On first startup the app creates all tables (via SQLAlchemy
`metadata.create_all`) and auto-seeds one admin account from
`ADMIN_NAME`/`ADMIN_EMAIL`/`ADMIN_PASSWORD`.

The API is served under `/api`, e.g. `http://localhost:5000/api/scholars`,
matching the frontend's `VITE_API_BASE_URL=http://localhost:5000/api`
default. Interactive docs: `http://localhost:5000/docs`.

## 4. Point the frontend at it

In `frontend/.env`:
```
VITE_API_BASE_URL=http://localhost:5000/api
```
And set `FRONTEND_ORIGIN=http://localhost:5173` in the backend's `.env` so
CORS allows it.

## Project layout

```
app/
  main.py          FastAPI app, CORS, error envelope, startup (create tables + seed admin)
  config.py        Ask Scholar settings loaded from .env (pydantic-settings)
  database.py      Async SQLAlchemy engine/session (asyncpg, Supabase-pooler friendly)
  models.py        Admin, User, Scholar, Video (SQLAlchemy)
  schemas.py       Pydantic request/response models (camelCase, matches frontend)
  serializers.py   ORM -> Pydantic conversion helpers
  security.py      bcrypt hashing + JWT create/decode
  deps.py          get_current_user / get_current_admin auth dependencies
  storage.py       Uploads scholar pictures to Supabase Storage
  seed.py          Auto-seeds the admin account
  routers/
    auth.py        POST /auth/user/register, /auth/user/login, /auth/admin/login
    scholars.py    GET /scholars, GET /scholars/{id}, POST /scholars/askQuestion/{id}
    admin.py       Scholar CRUD/invite, video linking, user management (all admin-only)
  rag/             The former standalone RAG backend, now a subpackage:
    config.py        env vars, model fallback chains, chunking/pinecone settings
    gemini_manager.py  GeminiKeyManager (key pool + per-task model fallback), lazily built
    schemas.py         pydantic request/response models for /api/rag/*
    router.py          POST /api/rag/ingest/youtube, /ingest/document/*, /search
    services/
      youtube_service.py   download audio -> Gemini transcript -> cleanup
      docs_service.py      fetch/extract doc text -> cleanup
      chunking.py          plain-text and timestamped-transcript chunking
      embedding_service.py Gemini embeddings + Pinecone upsert/query, lazily connected
```

## Integration

The two pieces aren't just mounted side by side — `askQuestion` and video
linking now use RAG directly:

- **`POST /api/admin/videos`** (linking a video to a scholar) now also
  kicks off a **background task** that transcribes the video and stores its
  embeddings in Pinecone, namespaced by `scholar_id`. This doesn't block
  the response (transcription can take a while); if it fails or RAG isn't
  configured, the video link itself is unaffected.
- **`POST /api/scholars/askQuestion/{id}`** still does **not** auto-generate
  a religious ruling (per the original design — a human always reviews the
  real answer). It now additionally does a RAG search scoped to that
  scholar's namespace for a passage relevant to the question, and if found,
  points the asker to that specific timestamp in that specific video
  instead of just "the latest video at 0s". Falls back to the original
  placeholder behavior if nothing matches or RAG isn't configured.
- **`/api/rag/*`** (ingest by URL/upload, generic `/search`) is still
  exposed directly too, for ingesting standalone reference documents/videos
  that aren't tied to a specific scholar (pass your own `namespace`).
  Ingestion and search require an admin JWT (`Authorization: Bearer
  <token>` from `/api/auth/admin/login`) — only `/api/rag/health` is open.

## Notes / decisions carried over from both backends

- **`Video` model**: not in the original `schema.prisma`, added because the
  frontend's admin "Add video" tab and the chat panel's YouTube-timestamp
  link both need somewhere to persist a scholar's linked videos.
- **Public vs admin scholar visibility**: `GET /scholars` and
  `GET /scholars/{id}` only return `status=ACTIVE` + `isActive=true`
  scholars; `PENDING`/`SUSPENDED` scholars stay visible via
  `GET /admin/scholars`.
- **Auth**: JWTs carry `{ sub: <id>, role: "USER"|"ADMIN" }`, single
  `Authorization: Bearer <token>` header. Passwords hashed with bcrypt.
- **Scholar self-registration via invite token**: schema fields exist
  (`inviteToken`/`inviteTokenExpiry`), but there's no frontend page to
  redeem one yet — `POST /scholars/complete-invite` is the natural next
  endpoint if you add that page.
- **Tables are created with `create_all` on startup**, not Alembic
  migrations — swap in Alembic if you need versioned migrations later.
- **RAG async work**: ingestion (via `/api/rag/ingest/*` or the video-link
  background task) can take a while — for real production traffic, move it
  to a proper task queue (Celery/RQ) instead of `BackgroundTasks`.
- **File size / duration limits**: add guards on audio length and document
  size to avoid huge Gemini uploads.
