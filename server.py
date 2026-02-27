import os
import json
import base64
import logging
import uuid
import re
from typing import List, Optional, Literal, Dict, Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
os.environ["AWS_REGION"] = AWS_REGION
os.environ.setdefault("AWS_DEFAULT_REGION", AWS_REGION)

# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------
MODEL_ID_TEXT   = os.getenv("MODEL_ID",        "us.amazon.nova-lite-v2:0")
MODEL_ID_VISION = os.getenv("VISION_MODEL_ID", "us.amazon.nova-lite-v2:0")
MODEL_ID_EMBED  = os.getenv("EMBED_MODEL_ID",  "amazon.titan-embed-text-v2:0")
MODEL_ID_CANVAS = os.getenv("CANVAS_MODEL_ID", "amazon.nova-canvas-v1:0")
MODEL_ID_SONIC  = os.getenv("SONIC_MODEL_ID",  "us.amazon.nova-sonic-v1:0")

NOVA_ACT_ENDPOINT = os.getenv("NOVA_ACT_ENDPOINT", "")
NOVA_ACT_API_KEY  = os.getenv("NOVA_ACT_API_KEY",  "")

# ---------------------------------------------------------------------------
# Feature flags / limits
# ---------------------------------------------------------------------------
VOICE_INTERNAL_LANGUAGE = os.getenv("VOICE_INTERNAL_LANGUAGE", "en")
TRANSLATE_ENABLED       = os.getenv("TRANSLATE_ENABLED", "1") == "1"

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(8  * 1024 * 1024)))   # 8 MB
MAX_PDF_BYTES   = int(os.getenv("MAX_PDF_BYTES",   str(12 * 1024 * 1024)))   # 12 MB

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# ---------------------------------------------------------------------------
# Bedrock client
# ---------------------------------------------------------------------------
BEDROCK_CONFIG = Config(
    connect_timeout=int(os.getenv("BEDROCK_CONNECT_TIMEOUT", "60")),
    read_timeout   =int(os.getenv("BEDROCK_READ_TIMEOUT",    "300")),
    retries        ={"max_attempts": int(os.getenv("BEDROCK_MAX_ATTEMPTS", "2"))},
)

bedrock    = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=BEDROCK_CONFIG)
_translate = boto3.client("translate",       region_name=AWS_REGION, config=BEDROCK_CONFIG)

# =============================================================================
# IN-MEMORY STORE (embeddings / sessions)
# =============================================================================
_embed_store: Dict[str, Dict[str, Any]] = {}

# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(title="PulseNova Server (Amazon Nova via Bedrock)")

cors_raw     = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
cors_origins = [x.strip() for x in cors_raw.split(",") if x.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins    =cors_origins if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods    =["*"],
    allow_headers    =["*"],
)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================
Role = Literal["user", "assistant"]


class ChatTurn(BaseModel):
    role: Role
    text: str


# ---------- /api/triage ----------
class TriageRequest(BaseModel):
    message:      str                 = Field(...)
    history:      List[ChatTurn]      = Field(default_factory=list)
    image_base64: Optional[str]       = Field(default=None)
    language:     Optional[str]       = Field(default="en-US")
    temperature:  float               = 0.3
    max_tokens:   int                 = 700

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, v):
        if v is None:
            raise ValueError("Message is required.")
        return v.strip()


class TextResponse(BaseModel):
    text: str


# ---------- /api/vision ----------
class VisionRequest(BaseModel):
    image_base64: Optional[str] = Field(default=None)
    pdf_base64:   Optional[str] = Field(default=None)
    prompt:       str           = Field(...)
    language:     Optional[str] = Field(default="en-US")
    temperature:  float         = 0.2
    max_tokens:   int           = 900


# ---------- /api/first-aid-guide ----------
class FirstAidRequest(BaseModel):
    injury_description: str           = Field(...)
    language:           Optional[str] = Field(default="en-US")


class FirstAidStep(BaseModel):
    step_text:    str
    image_base64: str   # base64-encoded PNG from Nova Canvas (may be "" on failure)


class FirstAidResponse(BaseModel):
    steps: List[FirstAidStep]


# ---------- /api/voice-turn ----------
class VoiceTurnRequest(BaseModel):
    transcript:           str             = Field(...)
    detected_lang:        Optional[str]   = Field(default=None)
    history:              List[ChatTurn]  = Field(default_factory=list)
    include_english_reply: bool           = Field(default=True)
    max_tokens:           int             = 350
    temperature:          float           = 0.3


class VoiceTurnResponse(BaseModel):
    session_id:          str
    source_language:     str
    source_language_name: str
    transcript_original: str
    transcript_en:       str
    reply_local:         str
    reply_en:            Optional[str] = None
    internal_language:   str          = "en"


# =============================================================================
# HELPERS
# =============================================================================

def _strip_data_url(b64: str) -> str:
    """Remove 'data:<mime>;base64,' prefix if present."""
    if not b64:
        return b64
    if b64.startswith("data:"):
        return b64.split(",", 1)[1]
    return b64


def _b64_to_bytes(b64: str, max_bytes: int) -> bytes:
    b64_clean = _strip_data_url(b64).strip()
    try:
        raw = base64.b64decode(b64_clean, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 payload.")
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Payload too large (max {max_bytes // (1024*1024)} MB).",
        )
    return raw


def _guess_image_format(raw: bytes) -> str:
    """Return Nova-compatible image format string."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if raw.startswith(b"RIFF") and b"WEBP" in raw[8:16]:
        return "webp"
    if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
        return "gif"
    raise HTTPException(
        status_code=400,
        detail="Unsupported image format. Please upload PNG, JPEG, WEBP, or GIF.",
    )


def _sanitize_history_for_nova(history: List[ChatTurn]) -> List[dict]:
    """
    Convert ChatTurn list → Nova messages-v1 format.
    Drops empty turns, ensures the list starts with a 'user' turn,
    and caps at 20 turns to keep context manageable.
    """
    cleaned = []
    for turn in history[-20:]:
        text = (turn.text or "").strip()
        if not text:
            continue
        # Strip any leftover gatekeeper tokens from history
        text = re.sub(r"\[TRIGGER_\w+\]", "", text).strip()
        if not text:
            continue
        cleaned.append({"role": turn.role, "content": [{"text": text}]})

    # Nova requires the first turn to be from "user"
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)

    return cleaned


def _normalize_lang_code(code: Optional[str]) -> str:
    """Return lowercase ISO-639-1 code, e.g. 'hi', 'en', 'fr'."""
    if not code:
        return ""
    code = code.strip().lower().replace("_", "-")
    if "-" in code:
        code = code.split("-")[0]
    return re.sub(r"[^a-z]", "", code)[:5]


def _language_name(code: str) -> str:
    return {
        "en": "English",
        "hi": "Hindi",
        "mr": "Marathi",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ar": "Arabic",
        "pt": "Portuguese",
        "it": "Italian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ru": "Russian",
        "nl": "Dutch",
        "pl": "Polish",
        "tr": "Turkish",
        "vi": "Vietnamese",
        "th": "Thai",
    }.get(code, code or "Unknown")


def _detect_language(text: str, browser_hint: Optional[str] = None) -> str:
    """
    Simple language detection:
    1. Trust the browser hint if provided.
    2. Unicode-range heuristic for Devanagari (Hindi/Marathi).
    3. Fall back to English.
    """
    hint = _normalize_lang_code(browser_hint)
    if hint:
        return hint
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    return "en"


def _append_language_reminder(messages: List[dict], ui_lang: Optional[str]) -> None:
    """
    Appends a language enforcement reminder to the last user message so the
    model sees the instruction as close to the generation point as possible.
    """
    lang_name = _language_name(_normalize_lang_code(ui_lang))
    if not messages:
        return
    last_msg = messages[-1]
    if last_msg.get("role") == "user":
        reminder = f"\n\n(CRITICAL REMINDER: You MUST reply entirely in {lang_name.upper()}. Do NOT switch to English.)"
        # Append to the last text block in that message's content
        for block in reversed(last_msg["content"]):
            if "text" in block:
                block["text"] += reminder
                break


def _translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate `text` using Amazon Nova (LLM-based) rather than the AWS Translate
    service, keeping all translation within the Bedrock stack.
    Falls back to the original text on any failure.
    """
    if not TRANSLATE_ENABLED or not text or source_lang == target_lang:
        return text
    try:
        source_name = _language_name(source_lang)
        target_name = _language_name(target_lang)
        prompt = (
            f"Translate the following text from {source_name} to {target_name}. "
            f"Output ONLY the translated text, nothing else, no preamble.\n\nText: {text}"
        )
        request_body = {
            "schemaVersion": "messages-v1",
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": 500, "temperature": 0.1},
        }
        return _nova_invoke(MODEL_ID_TEXT, request_body).strip()
    except Exception as e:
        logger.warning(f"Nova translation failed ({source_lang}→{target_lang}): {e}")
        return text


def _triage_system_prompt(ui_lang: Optional[str] = "en-US") -> str:
    """
    Builds the full triage system prompt that is injected into every
    conversation endpoint (triage, vision, voice-turn).
    """
    lang_name = _language_name(_normalize_lang_code(ui_lang))
    return (
        f"CRITICAL SYSTEM INSTRUCTION: YOU MUST COMMUNICATE ENTIRELY AND EXCLUSIVELY IN {lang_name.upper()}.\n"
        f"DO NOT USE ANY OTHER LANGUAGE UNDER ANY CIRCUMSTANCES. "
        f"Even if the user types in English, you must reply in {lang_name.upper()}.\n\n"

        "You are PulseNova, an AI-powered medical triage assistant.\n"
        "Your goal is to help users understand their symptoms and decide the most appropriate next step.\n\n"

        "CONVERSATION STYLE:\n"
        "- Be warm, calm, and reassuring.\n"
        "- Be concise and practical — avoid walls of text.\n"
        "- Ask ONE clarifying question at a time.\n"
        "- Use plain, simple language.\n\n"

        "TRIAGE RULES:\n"
        "1) NEVER diagnose. Never say 'you have X'.\n"
        "2) Do NOT list multiple possible conditions at once.\n"
        "3) EMERGENCY: If symptoms suggest a life-threatening emergency (e.g., chest pain, stroke signs, "
        "   severe allergic reaction, unconsciousness), instruct the user to call emergency services IMMEDIATELY.\n"
        "4) HOME CARE: If the concern seems minor, say so clearly and reassuringly.\n\n"

        "GATEKEEPER TOKENS (append at the very end of the reply, NEVER in the middle):\n"
        "5) FIRST AID GATEKEEPER: If the user describes a minor, treatable physical injury "
        "   (e.g., small cut, scrape, minor burn, bruise, splinter), append the EXACT string "
        "   '[TRIGGER_FIRST_AID]' as the very last characters of your response.\n"
        "6) CARE FINDER GATEKEEPER: If the user explicitly asks to find a hospital, doctor, clinic, "
        "   emergency room, urgent care, or pharmacy near them, append the EXACT string "
        "   '[TRIGGER_CARE_FINDER]' as the very last characters of your response so the UI can redirect them.\n\n"

        "DISCLAIMER RULE:\n"
        "Only add 'Not a diagnosis' at the end of a response when actively discussing symptoms.\n"
    )


def _extract_text_from_invoke(data: Dict[str, Any]) -> str:
    """Parse the text content out of a Nova messages-v1 response envelope."""
    # Primary path: messages-v1 format
    content = data.get("output", {}).get("message", {}).get("content", [])
    if isinstance(content, list) and content:
        t0 = content[0].get("text", "")
        if t0:
            return t0
    # Legacy / alternate formats
    if "results" in data and data["results"]:
        return data["results"][0].get("outputText", "") or ""
    if "completion" in data and isinstance(data["completion"], str):
        return data["completion"]
    if "text" in data and isinstance(data["text"], str):
        return data["text"]
    return ""


def _nova_invoke(model_id: str, request_body: dict) -> str:
    """Single Bedrock invoke call, returns the assistant text or raises HTTPException."""
    try:
        resp = bedrock.invoke_model(
            modelId     =model_id,
            contentType ="application/json",
            accept      ="application/json",
            body        =json.dumps(request_body),
        )
        data = json.loads(resp["body"].read())
        result = _extract_text_from_invoke(data).strip()
        return result or "No response from model."
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg  = e.response["Error"]["Message"]
        logger.error(f"Bedrock ClientError [{code}]: {msg}")
        raise HTTPException(status_code=502, detail=f"Bedrock error: {code} – {msg}")
    except (NoCredentialsError, EndpointConnectionError) as e:
        logger.error(f"Bedrock connectivity error: {e}")
        raise HTTPException(status_code=503, detail="Cannot reach Bedrock. Check AWS credentials/region.")
    except Exception as e:
        logger.error(f"Unexpected Bedrock error: {e}")
        raise HTTPException(status_code=500, detail="Model invocation failed.")


def _nova_canvas_invoke(prompt: str) -> str:
    """
    Generate a first-aid illustration via Nova Canvas.
    Returns a base64-encoded PNG string, or "" on failure (non-fatal).
    The prompt is enhanced for a clean medical-infographic look.
    """
    if not prompt:
        return ""

    enhanced_prompt = (
        "Clean, minimalist vector-style illustration for a medical first-aid instructional guide. "
        "Flat colors, white background, simple lines, no text, easy to understand at a glance. "
        f"{prompt.strip()}"
    )

    body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": enhanced_prompt[:1024],   # Nova Canvas prompt limit
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height":         1024,
            "width":          1024,
            "cfgScale":       8.0,
            "quality":        "standard",
        },
    }

    try:
        resp = bedrock.invoke_model(
            modelId     =MODEL_ID_CANVAS,
            contentType ="application/json",
            accept      ="application/json",
            body        =json.dumps(body),
        )
        data = json.loads(resp["body"].read())
        images = data.get("images", [])
        return images[0] if images else ""
    except Exception as e:
        logger.error(f"Nova Canvas invocation failed: {e}")
        return ""   # Non-fatal: the step is still returned without an image


def _pdf_first_page_to_png_b64(pdf_bytes: bytes) -> str:
    """
    Convert the first page of a PDF to a PNG base64 string using PyMuPDF.
    Raises HTTPException if PyMuPDF is not installed or conversion fails.
    """
    try:
        import fitz  # PyMuPDF
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(0)
        pix  = page.get_pixmap(dpi=200)
        return base64.b64encode(pix.tobytes("png")).decode("utf-8")
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="PDF support requires PyMuPDF. Install with: pip install pymupdf",
        )
    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        raise HTTPException(status_code=400, detail="Could not read PDF. Try uploading a screenshot instead.")


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/")
def index():
    """Serve the frontend single-page application."""
    return FileResponse("index.html")


@app.get("/health")
def health():
    """Simple liveness probe used by load-balancers / CI."""
    return {"ok": True, "region": AWS_REGION, "model_text": MODEL_ID_TEXT, "model_vision": MODEL_ID_VISION}


@app.get("/config")
def public_config():
    """
    Exposes non-secret public configuration values to the frontend.
    The frontend (index.html) calls this to get the Google Maps API key
    so it can lazy-load the Maps SDK at runtime.
    """
    return {"google_maps_api_key": GOOGLE_MAPS_API_KEY}


# ---------------------------------------------------------------------------
# /api/triage  — main text + optional image chat endpoint
# ---------------------------------------------------------------------------
@app.post("/api/triage", response_model=TextResponse)
def triage(req: TriageRequest):
    """
    Primary symptom-triage endpoint.

    Request fields (from index.html callNovaTriage):
        message       – user's text message (required)
        history       – previous ChatTurn list
        image_base64  – optional image (data-URL or raw base64)
        language      – BCP-47 locale string, e.g. "en-US", "hi-IN"
        temperature   – model temperature (default 0.3)
        max_tokens    – max output tokens (default 700)

    Response: { "text": "<assistant reply, may contain [TRIGGER_*] tokens>" }

    The frontend strips [TRIGGER_FIRST_AID] / [TRIGGER_CARE_FINDER] from the
    displayed text and uses them to trigger UI actions.
    """
    if not req.message and not req.image_base64:
        raise HTTPException(status_code=400, detail="Provide a message or an image.")

    system_list = [{"text": _triage_system_prompt(req.language)}]
    messages    = _sanitize_history_for_nova(req.history)

    # Build the new user turn
    user_content: List[dict] = []
    if req.message:
        user_content.append({"text": req.message})

    if req.image_base64:
        raw = _b64_to_bytes(req.image_base64, MAX_IMAGE_BYTES)
        fmt = _guess_image_format(raw)
        user_content.append({
            "image": {
                "format": fmt,
                "source": {"bytes": _strip_data_url(req.image_base64)},
            }
        })
        user_content.append({
            "text": "Describe what you can observe in the image and advise next steps if relevant."
        })

    messages.append({"role": "user", "content": user_content})
    _append_language_reminder(messages, req.language)

    request_body = {
        "schemaVersion":  "messages-v1",
        "system":         system_list,
        "messages":       messages,
        "inferenceConfig": {
            "maxTokens":  int(req.max_tokens),
            "temperature": float(req.temperature),
            "topP":        0.9,
        },
    }

    return TextResponse(text=_nova_invoke(MODEL_ID_TEXT, request_body))


# ---------------------------------------------------------------------------
# /api/first-aid-guide  — visual step-by-step first aid generator
# ---------------------------------------------------------------------------
@app.post("/api/first-aid-guide", response_model=FirstAidResponse)
def first_aid_guide(req: FirstAidRequest):
    """
    Generates 3–4 illustrated first-aid steps for a described minor injury.

    Request fields (from index.html callFirstAidGuide):
        injury_description – free-text description of the injury
        language           – BCP-47 locale string

    Response: { "steps": [ { "step_text": "...", "image_base64": "..." }, ... ] }

    The model returns a JSON array. Each element has:
        step_text    – the instruction in the requested language
        image_prompt – English description used to generate the illustration

    Nova Canvas is called per-step to generate images. Canvas failures are
    non-fatal; the step is still returned with image_base64 = "".
    """
    target_lang = _language_name(_normalize_lang_code(req.language))

    system_prompt = (
        "You are a certified first-aid expert producing home-care instructions.\n"
        "The user will describe a minor physical injury.\n"
        "Provide exactly 3 to 4 clear, practical, safe home-care steps.\n\n"
        "OUTPUT FORMAT — CRITICAL:\n"
        "Return ONLY a raw JSON array (no markdown fences, no preamble, no trailing text).\n"
        "Each element must have exactly two string keys:\n"
        f"  'step_text'    — the instruction written entirely in {target_lang.upper()}\n"
        "  'image_prompt' — a SHORT English description of the action for an illustration\n\n"
        "Example element:\n"
        '  {"step_text": "Rinse the wound under cool running water for 2 minutes.", '
        '"image_prompt": "Hands rinsing a small cut under flowing tap water"}'
    )

    request_body = {
        "schemaVersion":  "messages-v1",
        "system":         [{"text": system_prompt}],
        "messages":       [{"role": "user", "content": [{"text": req.injury_description}]}],
        "inferenceConfig": {"maxTokens": 1200, "temperature": 0.2},
    }

    raw_text   = _nova_invoke(MODEL_ID_TEXT, request_body)
    clean_json = re.sub(r"```(?:json)?|```", "", raw_text).strip()

    try:
        steps_data = json.loads(clean_json)
        if not isinstance(steps_data, list):
            raise ValueError("Expected a JSON array.")
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"first-aid-guide JSON parse error: {e}\nRaw: {raw_text[:500]}")
        raise HTTPException(
            status_code=500,
            detail="Failed to parse first-aid steps from the model. Please try again.",
        )

    results: List[FirstAidStep] = []
    for step in steps_data[:4]:   # cap at 4 steps
        step_text    = (step.get("step_text", "") or "").strip()
        image_prompt = (step.get("image_prompt", "") or "").strip()

        if not step_text:
            continue    # skip malformed steps

        image_b64 = _nova_canvas_invoke(image_prompt) if image_prompt else ""
        results.append(FirstAidStep(step_text=step_text, image_base64=image_b64))

    if not results:
        raise HTTPException(
            status_code=500,
            detail="Model returned no usable first-aid steps.",
        )

    return FirstAidResponse(steps=results)


# ---------------------------------------------------------------------------
# /api/vision  — X-Ray and Lab report analysis
# ---------------------------------------------------------------------------
@app.post("/api/vision", response_model=TextResponse)
def vision(req: VisionRequest):
    """
    Analyzes a medical image (X-Ray, MRI, CT, lab report) and returns a
    structured plain-language report.

    Request fields (from index.html callNovaVision):
        image_base64 – image as data-URL or raw base64
        pdf_base64   – optional PDF (first page is extracted as PNG)
        prompt       – the structured report prompt built by the frontend
        language     – BCP-47 locale string
        temperature  – model temperature (default 0.2)
        max_tokens   – max output tokens (default 900)

    Response: { "text": "<markdown-formatted report>" }

    The frontend renders the markdown with formatMarkdown() and offers a
    print button that uses @media print CSS rules.
    """
    if not req.image_base64 and not req.pdf_base64:
        raise HTTPException(status_code=400, detail="Provide image_base64 or pdf_base64.")

    system_list = [{"text": _triage_system_prompt(req.language)}]

    # Resolve image source: PDF takes precedence over raw image if both provided
    if req.pdf_base64 and not req.image_base64:
        pdf_bytes        = _b64_to_bytes(req.pdf_base64, MAX_PDF_BYTES)
        image_b64_model  = _pdf_first_page_to_png_b64(pdf_bytes)
        image_fmt        = "png"
    else:
        raw             = _b64_to_bytes(req.image_base64, MAX_IMAGE_BYTES)
        image_fmt       = _guess_image_format(raw)
        image_b64_model = _strip_data_url(req.image_base64)

    messages = [
        {
            "role": "user",
            "content": [
                {"text": req.prompt.strip()},
                {
                    "image": {
                        "format": image_fmt,
                        "source": {"bytes": image_b64_model},
                    }
                },
            ],
        }
    ]
    _append_language_reminder(messages, req.language)

    request_body = {
        "schemaVersion":  "messages-v1",
        "system":         system_list,
        "messages":       messages,
        "inferenceConfig": {
            "maxTokens":  int(req.max_tokens),
            "temperature": float(req.temperature),
        },
    }

    return TextResponse(text=_nova_invoke(MODEL_ID_VISION, request_body))


# ---------------------------------------------------------------------------
# /api/voice-turn  — voice conversation endpoint
# ---------------------------------------------------------------------------
@app.post("/api/voice-turn", response_model=VoiceTurnResponse)
def voice_turn(req: VoiceTurnRequest):
    """
    Handles one turn of a voice conversation.

    The frontend (index.html _voiceSendAndReply / callVoiceTurn) sends:
        transcript           – the speech-to-text result (browser Web Speech API)
        detected_lang        – BCP-47 code from the language selector, e.g. "hi-IN"
        history              – previous ChatTurn list (same shape as /api/triage)
        include_english_reply – whether to also return an English translation
        max_tokens           – default 350 (shorter for voice)
        temperature          – default 0.3

    Response fields consumed by the frontend:
        reply_local          – assistant reply in the user's language (displayed + spoken)
        source_language      – ISO-639-1 code used for language display
        source_language_name – Human-readable language name
        transcript_original  – the raw transcript (displayed in voice overlay)
        transcript_en        – English translation of the transcript (for the translation panel)
        reply_en             – English translation of reply (optional, for translation panel)

    The reply may contain [TRIGGER_FIRST_AID] or [TRIGGER_CARE_FINDER] tokens
    which the frontend intercepts for UI actions.
    """
    session_id          = str(uuid.uuid4())
    transcript_original = req.transcript.strip()

    if not transcript_original:
        raise HTTPException(status_code=400, detail="transcript must not be empty.")

    # Detect / confirm the language from the browser hint
    source_lang = _detect_language(transcript_original, req.detected_lang)

    # Build conversation messages
    messages = _sanitize_history_for_nova(req.history)
    messages.append({"role": "user", "content": [{"text": transcript_original}]})
    _append_language_reminder(messages, req.detected_lang)

    voice_system_prompt = (
        _triage_system_prompt(req.detected_lang)
        + "\n\nVOICE OUTPUT RULES:\n"
        "- Keep responses smooth and conversational — they will be read aloud via browser TTS.\n"
        "- Use short sentences. Avoid bullet points and markdown.\n"
        "- Keep replies under 3 sentences unless the situation is urgent.\n"
        "- Gatekeeper tokens ([TRIGGER_FIRST_AID], [TRIGGER_CARE_FINDER]) must still be appended when appropriate.\n"
    )

    request_body = {
        "schemaVersion":  "messages-v1",
        "system":         [{"text": voice_system_prompt}],
        "messages":       messages,
        "inferenceConfig": {
            "maxTokens":  int(req.max_tokens),
            "temperature": float(req.temperature),
            "topP":        0.9,
        },
    }

    reply_local = _nova_invoke(MODEL_ID_TEXT, request_body)

    # Translate the user's transcript to English for the UI translation panel
    transcript_en = (
        _translate_text(transcript_original, source_lang, "en")
        if source_lang != "en"
        else transcript_original
    )

    # Optionally provide an English translation of the reply
    reply_en: Optional[str] = None
    if req.include_english_reply and source_lang != "en":
        # Strip gatekeeper tokens before translating
        clean_reply = re.sub(r"\[TRIGGER_\w+\]", "", reply_local).strip()
        reply_en    = _translate_text(clean_reply, source_lang, "en")

    return VoiceTurnResponse(
        session_id          =session_id,
        source_language     =source_lang,
        source_language_name=_language_name(source_lang),
        transcript_original =transcript_original,
        transcript_en       =transcript_en,
        reply_local         =reply_local,
        reply_en            =reply_en,
        internal_language   =source_lang,
    )
