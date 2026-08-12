"""
Strict JSON parsing helpers shared by every Gemini call that needs
structured output (transcription, answer generation, ...).

Why this exists: even with response_mime_type="application/json", Gemini
can still occasionally hand back something that isn't valid JSON on its
own — wrapped in markdown fences, with a trailing comma, or (most common
on long transcription responses) cut off mid-object because the response
hit a length/stop boundary. This module makes "the model didn't return
valid JSON" a recoverable situation instead of a hard failure:

1. `build_json_config(schema)` asks Gemini for constrained/structured
   output (response_schema) whenever the installed google-genai version
   supports it, in addition to response_mime_type="application/json".
   This is strictly stronger than the mime type alone — the model is
   constrained to the schema's shape while generating, not just asked to
   *format* as JSON after the fact.
2. `extract_parsed(response)` prefers the SDK's own `response.parsed`
   (built from response_schema) when present and valid.
3. `loads_relaxed(raw)` is the fallback: strips ```json fences, then
   tries increasingly forgiving repairs (trailing commas, and — for the
   common "truncated mid-array" case — dropping the last, incomplete
   element and closing the array/object) before giving up.

Callers should treat a raised JSONRepairError as a normal "ask again"
failure: it bubbles up through GeminiKeyManager.call, which already
retries on a different key and eventually rotates to the next model in
the fallback chain, so a single bad JSON response never wastes the whole
ingestion job.
"""
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class JSONRepairError(ValueError):
    """Raised when a Gemini response could not be coerced into valid JSON
    by any of our repair strategies."""


def build_json_config(schema: Optional[object] = None) -> dict:
    """
    Config dict for client.models.generate_content() that strictly
    enforces JSON output. Always sets response_mime_type; additionally
    sets response_schema when a schema is given, which makes Gemini
    constrain generation to that exact shape (list[PydanticModel] or a
    single PydanticModel) instead of merely being told to "output JSON".
    """
    cfg = {"response_mime_type": "application/json"}
    if schema is not None:
        cfg["response_schema"] = schema
    return cfg


def extract_parsed(response):
    """
    Returns response.parsed if the SDK populated it (only happens when a
    response_schema was set and the model's output matched it exactly).
    Returns None otherwise so the caller falls back to loads_relaxed().
    """
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return parsed
    return None


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()
    return raw


def _drop_trailing_commas(raw: str) -> str:
    return re.sub(r",\s*([\]}])", r"\1", raw)


def _salvage_truncated_array(raw: str) -> Optional[str]:
    """
    Handles the most common truncation shape: a JSON array of objects
    where generation stopped partway through the last object, e.g.
    '[{"a": 1}, {"a": 2}, {"a": 3' (no closing braces at all).

    Strategy: walk the string tracking bracket/brace depth and string
    state; remember the offset right after the last position where we
    were back down to depth 1 inside the top-level array (i.e. right
    after a complete top-level element). Truncate there and close the
    array. Returns None if the raw text isn't array-shaped or no
    complete element was found.
    """
    s = raw.strip()
    if not s.startswith("["):
        return None

    depth = 0
    in_string = False
    escape = False
    last_safe_cut = None

    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 1 and ch == "}":
                # just closed a top-level element (an object directly
                # inside the outer array) -> safe place to cut+close
                last_safe_cut = i + 1

    if last_safe_cut is None:
        return None
    return s[:last_safe_cut] + "]"


def loads_relaxed(raw: str):
    """
    Best-effort JSON parse with progressively more aggressive repairs.
    Raises JSONRepairError (never a bare json.JSONDecodeError) if every
    strategy fails, so callers only need to catch one exception type.
    """
    original = raw
    raw = _strip_fences(raw)

    attempts = [
        raw,
        _drop_trailing_commas(raw),
    ]

    for candidate in attempts:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    salvaged = _salvage_truncated_array(raw)
    if salvaged is not None:
        try:
            result = json.loads(salvaged)
            logger.warning(
                "Recovered valid JSON from a truncated response by dropping the "
                "incomplete trailing element (kept %d of the raw %d chars).",
                len(salvaged), len(raw),
            )
            return result
        except json.JSONDecodeError:
            pass

    logger.error("Could not parse or repair JSON response. Raw (first 500 chars): %r", original[:500])
    raise JSONRepairError("Gemini did not return valid/repairable JSON.")
