"""
Vercel serverless entrypoint.

Vercel's Python runtime looks inside /api for files that expose an ASGI/WSGI
"app" object. This file just re-exports the existing FastAPI app from
app/main.py so nothing in the rest of the codebase has to change.
"""

import sys
from pathlib import Path

# Make sure the project root (one level up from /api) is importable as
# "app.*", since Vercel executes this file directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.main import app  # noqa: E402

# Vercel's Python runtime detects this variable name automatically.
app = app
