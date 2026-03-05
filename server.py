import os
import json
import base64
import logging
import uuid
import re
import time
from typing import List, Optional, Literal, Dict, Any, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import boto3
from botocore.config import Config

from fastapi import FastAPI, HTTPException, Depends, Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from sqlalchemy.orm import Session
from database import (
    init_db as db_init_db, get_db,
    User, TriageSession, VitalReading, MedicalDocument, Prescription
)

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

MODEL_ID_TEXT   = os.getenv("MODEL_ID",        "us.amazon.nova-lite-v2:0")
MODEL_ID_VISION = os.getenv("VISION_MODEL_ID", "us.amazon.nova-lite-v2:0")
MODEL_ID_EMBED  = os.getenv("EMBED_MODEL_ID",  "amazon.titan-embed-text-v2:0")
MODEL_ID_CANVAS = os.getenv("CANVAS_MODEL_ID", "amazon.nova-canvas-v1:0")
MODEL_ID_SONIC  = os.getenv("SONIC_MODEL_ID",  "us.amazon.nova-sonic-v1:0")

NOVA_ACT_ENDPOINT = os.getenv("NOVA_ACT_ENDPOINT", "")
NOVA_ACT_API_KEY  = os.getenv("NOVA_ACT_API_KEY",  "")

VOICE_INTERNAL_LANGUAGE = os.getenv("VOICE_INTERNAL_LANGUAGE", "en")
TRANSLATE_ENABLED       = os.getenv("TRANSLATE_ENABLED", "1") == "1"

MAX_IMAGE_BYTES     = int(os.getenv("MAX_IMAGE_BYTES", str(8  * 1024 * 1024)))
MAX_PDF_BYTES       = int(os.getenv("MAX_PDF_BYTES",   str(12 * 1024 * 1024)))
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# Your Alexa Skill ID — rejects requests from other skills
ALEXA_SKILL_ID = os.getenv(
    "ALEXA_SKILL_ID",
    "amzn1.ask.skill.dee8af9a-0210-4b14-b90d-0ef50921882e"
)

# ---------------------------------------------------------------------------
# Cognito OAuth
# ---------------------------------------------------------------------------
COGNITO_DOMAIN              = (os.getenv("COGNITO_DOMAIN") or "").rstrip("/")
COGNITO_CLIENT_ID           = os.getenv("COGNITO_CLIENT_ID",           "")
COGNITO_CLIENT_SECRET       = os.getenv("COGNITO_CLIENT_SECRET",       "")
COGNITO_REDIRECT_URI        = os.getenv("COGNITO_REDIRECT_URI",        "")
COGNITO_LOGOUT_REDIRECT_URI = os.getenv("COGNITO_LOGOUT_REDIRECT_URI", "")
COGNITO_SCOPES              = os.getenv("COGNITO_SCOPES",    "openid email profile")
COGNITO_IDP                 = os.getenv("COGNITO_IDP",       "")
COGNITO_USER_POOL_ID        = os.getenv("COGNITO_USER_POOL_ID", "")
COGNITO_ISSUER              = os.getenv("COGNITO_ISSUER",       "")
SESSION_COOKIE              = os.getenv("SESSION_COOKIE", "pulsenova_session")

BEDROCK_CONFIG = Config(
    connect_timeout=int(os.getenv("BEDROCK_CONNECT_TIMEOUT", "60")),
    read_timeout=int(os.getenv("BEDROCK_READ_TIMEOUT",       "300")),
    retries={"max_attempts": int(os.getenv("BEDROCK_MAX_ATTEMPTS", "2"))},
)

bedrock     = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=BEDROCK_CONFIG)
_translate  = boto3.client("translate",       region_name=AWS_REGION, config=BEDROCK_CONFIG)
_comprehend = boto3.client("comprehend",      region_name=AWS_REGION, config=BEDROCK_CONFIG)

# =============================================================================
# IN-MEMORY STORES
# =============================================================================
_embed_store: Dict[str, Dict[str, Any]] = {}
_sessions:    Dict[str, Dict[str, Any]] = {}

# =============================================================================
# FASTAPI
# =============================================================================
app = FastAPI(title="PulseNova Server (Amazon Nova via Bedrock)")

cors_raw = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000",
)
cors_origins = [x.strip() for x in cors_raw.split(",") if x.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=".", html=False), name="static")


@app.on_event("startup")
def on_startup():
    try:
        db_init_db()
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")


# =============================================================================
# PYDANTIC MODELS
# =============================================================================
Role = Literal["user", "assistant"]


class ChatTurn(BaseModel):
    role: Role
    text: str


class TriageRequest(BaseModel):
    message:      Optional[str]  = Field(default=None)
    history:      List[ChatTurn] = Field(default_factory=list)
    image_base64: Optional[str]  = Field(default=None)
    language:     Optional[str]  = Field(default="en-US")
    chat_id:      Optional[str]  = Field(default=None)
    temperature:  float          = 0.3
    max_tokens:   int            = 700

    @field_validator("message")
    @classmethod
    def normalize_message(cls, v):
        if v is None:
            return None
        v = v.strip()
        return v if v else None


class VisionRequest(BaseModel):
    image_base64: Optional[str] = Field(default=None)
    pdf_base64:   Optional[str] = Field(default=None)
    prompt:       str           = Field(...)
    language:     Optional[str] = Field(default="en-US")
    doc_type:     str           = Field(default="xray")
    save_to_db:   bool          = Field(default=True)
    temperature:  float         = 0.2
    max_tokens:   int           = 900


class TextResponse(BaseModel):
    text: str


class FirstAidRequest(BaseModel):
    injury_description: str           = Field(...)
    language:           Optional[str] = Field(default="en-US")


class FirstAidStep(BaseModel):
    step_text:    str
    image_base64: str


class FirstAidResponse(BaseModel):
    steps: List[FirstAidStep]


class VoiceTurnRequest(BaseModel):
    transcript:            str            = Field(...)
    detected_lang:         Optional[str]  = Field(default=None)
    history:               List[ChatTurn] = Field(default_factory=list)
    include_english_reply: bool           = Field(default=True)
    max_tokens:            int            = 350
    temperature:           float          = 0.3


class VoiceTurnResponse(BaseModel):
    session_id:           str
    source_language:      str
    source_language_name: str
    transcript_original:  str
    transcript_en:        str
    reply_local:          str
    reply_en:             Optional[str] = None
    internal_language:    str           = "en"


class EmbedRequest(BaseModel):
    text:         str
    image_base64: Optional[str] = None
    metadata:     dict          = Field(default_factory=dict)
    doc_id:       Optional[str] = None


class EmbedResponse(BaseModel):
    doc_id:        str
    embedding_dim: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    doc_id:   str
    score:    float
    text:     str
    metadata: dict


class SearchResponse(BaseModel):
    hits: List[SearchHit]


# ---------------------------------------------------------------------------
# Prescription Pydantic models  ← NEW
# ---------------------------------------------------------------------------
class PrescriptionIn(BaseModel):
    name:   str           = Field(...)
    dose:   Optional[str] = ""
    times:  List[str]     = Field(default_factory=list)
    repeat: Optional[str] = "DAILY"
    days:   List[str]     = Field(default_factory=list)
    notes:  Optional[str] = ""


# =============================================================================
# PYDANTIC RESPONSE SCHEMAS FOR HISTORY
# =============================================================================
class TriageSessionOut(BaseModel):
    id:         Optional[str] = None
    title:      Optional[str] = None
    messages:   Any           = None
    created_at: Any           = None

    class Config:
        from_attributes = True


class MedicalDocumentOut(BaseModel):
    id:          Optional[str] = None
    doc_type:    Optional[str] = None
    image_b64:   Optional[str] = None
    report_html: Optional[str] = None
    created_at:  Any           = None

    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    triage:    List[TriageSessionOut]    = []
    documents: List[MedicalDocumentOut] = []


# =============================================================================
# HELPERS
# =============================================================================
def _strip_data_url(b64: str) -> str:
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
        raise HTTPException(status_code=413, detail=f"Payload too large (>{max_bytes} bytes).")
    return raw


def _guess_image_format(raw: bytes) -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if raw.startswith(b"RIFF") and b"WEBP" in raw[8:16]:
        return "webp"
    if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
        return "gif"
    raise ValueError("Unsupported file format.")


def _sanitize_history_for_nova(history: List[ChatTurn]) -> List[dict]:
    cleaned = []
    for turn in history[-20:]:
        text = (turn.text or "").strip()
        if not text:
            continue
        cleaned.append({"role": turn.role, "content": [{"text": text}]})
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    return cleaned


def _normalize_lang_code(code: Optional[str]) -> str:
    if not code:
        return ""
    code = code.strip().lower().replace("_", "-")
    if "-" in code:
        code = code.split("-")[0]
    return re.sub(r"[^a-z]", "", code)[:5]


def _language_name(code: str) -> str:
    return {
        "en": "English", "hi": "Hindi", "mr": "Marathi",
        "es": "Spanish", "fr": "French", "de": "German", "ar": "Arabic",
    }.get(code, code or "Unknown")


def _detect_language(text: str, browser_hint: Optional[str] = None) -> str:
    hint = _normalize_lang_code(browser_hint)
    if hint:
        return hint
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    return "en"


def _translate_text(text: str, source_lang: str, target_lang: str) -> str:
    if not TRANSLATE_ENABLED or not text or source_lang == target_lang:
        return text
    try:
        resp = _translate.translate_text(
            Text=text, SourceLanguageCode=source_lang, TargetLanguageCode=target_lang
        )
        return resp.get("TranslatedText", text)
    except Exception as e:
        logger.warning(f"Translate API skipped/failed: {e}")
        return text


def _triage_system_prompt(ui_lang: Optional[str] = "en-US") -> str:
    c         = _normalize_lang_code(ui_lang)
    lang_name = _language_name(c)
    return (
        f"CRITICAL SYSTEM INSTRUCTION: YOU MUST COMMUNICATE ENTIRELY AND EXCLUSIVELY IN {lang_name.upper()}.\n"
        f"DO NOT USE ENGLISH UNDER ANY CIRCUMSTANCES unless citing a specific medical term. "
        f"Even if the user types in English, you must reply in {lang_name.upper()}.\n\n"
        "You are PulseNova, an AI-powered medical triage assistant.\n"
        "Your goal is to help users understand their symptoms and decide the most appropriate next step: "
        "Emergency, Urgent Care, or Home Care.\n\n"
        "CONVERSATION STYLE:\n"
        "- Be warm, calm, and reassuring.\n"
        "- For medical topics, be concise and practical.\n"
        "- Ask ONE clarifying question at a time.\n\n"
        "TRIAGE RULES:\n"
        "1) Never diagnose.\n"
        "2) Do not list multiple possible conditions at once.\n"
        "3) EMERGENCY: If symptoms are life-threatening, tell them to call emergency services immediately.\n"
        "4) HOME CARE: If the concern seems minor, say so clearly.\n"
        "5) FIRST AID GATEKEEPER: If the user describes a minor, treatable physical injury (e.g., a cut, scrape, minor burn), "
        "append the exact string '[TRIGGER_FIRST_AID]' at the very end of your response so the system knows to generate a visual guide.\n\n"
        "DISCLAIMER RULE:\n"
    )


def _extract_text_from_invoke(data: Dict[str, Any]) -> str:
    text = data.get("output", {}).get("message", {}).get("content", [{}])
    if isinstance(text, list) and text:
        t0 = text[0].get("text", "")
        if t0:
            return t0
    if "results" in data and data["results"]:
        return data["results"][0].get("outputText", "") or ""
    if "completion" in data and isinstance(data["completion"], str):
        return data["completion"]
    if "text" in data and isinstance(data["text"], str):
        return data["text"]
    return ""


def _nova_invoke(model_id: str, request_body: dict) -> str:
    try:
        resp = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body),
        )
        data = json.loads(resp["body"].read())
        return _extract_text_from_invoke(data).strip() or "No response from model."
    except Exception as e:
        logger.error(f"Bedrock Error: {e}")
        raise HTTPException(status_code=500, detail="Model invocation failed.")


def _nova_canvas_invoke(prompt: str) -> str:
    enhanced_prompt = (
        "Clean, minimalist vector illustrations, medical instructional infographic style, flat colors, white background. "
        + (prompt or "")
    )
    body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": enhanced_prompt[:1000]},
        "imageGenerationConfig": {
            "numberOfImages": 1, "height": 1024, "width": 1024, "cfgScale": 8.0,
        },
    }
    try:
        resp = bedrock.invoke_model(
            modelId=MODEL_ID_CANVAS, contentType="application/json",
            accept="application/json", body=json.dumps(body),
        )
        data = json.loads(resp["body"].read())
        return data["images"][0] if "images" in data and data["images"] else ""
    except Exception as e:
        logger.error(f"Canvas invocation failed: {e}")
        return ""


def _pdf_first_page_to_png_b64(pdf_bytes: bytes) -> str:
    try:
        import fitz
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(0)
        return base64.b64encode(page.get_pixmap(dpi=200).tobytes("png")).decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read PDF. Try a screenshot.")


def _build_patient_context_from_user(user: Optional[User]) -> str:
    if not user:
        return ""
    parts = []
    if getattr(user, "age",             None): parts.append(f"Age: {user.age}")
    if getattr(user, "gender",          None): parts.append(f"Gender: {user.gender}")
    hw = []
    if getattr(user, "height", None): hw.append(str(user.height))
    if getattr(user, "weight", None): hw.append(str(user.weight))
    if hw: parts.append(f"Height/Weight: {' / '.join(hw)}")
    if getattr(user, "allergies",       None): parts.append(f"Allergies: {user.allergies}")
    if getattr(user, "medical_history", None): parts.append(f"Medical history: {user.medical_history}")
    if not parts:
        return ""
    return "PATIENT CONTEXT (use this for safety and personalization):\n- " + "\n- ".join(parts)


def _allergy_safety_rule_block() -> str:
    return (
        "ALLERGY SAFETY RULE (IMPORTANT):\n"
        "- You must consider the patient's listed allergies in every response.\n"
        "- If the user asks whether they can eat/take/use something that matches a listed allergy "
        "(including plurals and common variations), you must clearly say NO and explain it is "
        "because they have that allergy.\n"
        "- If the user reports possible allergic reaction symptoms (hives, swelling, wheezing, "
        "trouble breathing, fainting, vomiting after exposure), treat it as urgent/emergency and "
        "tell them to seek emergency care immediately.\n"
        "- Do not diagnose. Keep guidance practical and safe.\n"
    )


# =============================================================================
# OAuth helpers
# =============================================================================
def _require_oauth_config():
    if not (COGNITO_DOMAIN and COGNITO_CLIENT_ID and COGNITO_REDIRECT_URI):
        raise HTTPException(status_code=500, detail="Missing Cognito OAuth configuration.")


def _http_post_form(
    url: str, data: Dict[str, str], headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    encoded = urlencode(data).encode("utf-8")
    req = Request(url, data=encoded, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urlopen(req, timeout=25) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        try:
            body = e.read().decode("utf-8")
            return {"error": "http_error", "status": e.code, "body": body}
        except Exception:
            return {"error": "http_error", "status": e.code}
    except URLError as e:
        return {"error": "url_error", "detail": str(e)}


def _get_session(session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not session_id:
        return None
    return _sessions.get(session_id)


def _decode_jwt_no_verify(token: str) -> Dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload_b64  = parts[1] + "==="
        payload_json = base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8")
        return json.loads(payload_json)
    except Exception:
        return {}


def _create_session(tokens: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    session_id = str(uuid.uuid4())
    profile    = {}
    id_token   = tokens.get("id_token")
    if id_token:
        profile = _decode_jwt_no_verify(id_token) or {}
    _sessions[session_id] = {
        "tokens": tokens, "profile": profile, "created_at": int(time.time()),
    }
    return session_id, profile


_jwks_cache: Dict[str, Any] = {"keys": None, "fetched_at": 0}


def _cognito_issuer() -> str:
    if COGNITO_ISSUER:
        return COGNITO_ISSUER.rstrip("/")
    if COGNITO_USER_POOL_ID:
        return f"https://cognito-idp.{AWS_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}"
    return ""


def _fetch_jwks() -> Dict[str, Any]:
    issuer = _cognito_issuer()
    if not issuer:
        raise HTTPException(status_code=500, detail="Missing COGNITO_USER_POOL_ID or COGNITO_ISSUER.")
    now = int(time.time())
    if _jwks_cache["keys"] and (now - _jwks_cache["fetched_at"] < 3600):
        return _jwks_cache["keys"]
    jwks_url = f"{issuer.rstrip('/')}/.well-known/jwks.json"
    try:
        with urlopen(jwks_url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            _jwks_cache["keys"]       = data
            _jwks_cache["fetched_at"] = now
            return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch JWKS: {e}")


def _get_bearer_token(req: FastAPIRequest) -> str:
    auth = req.headers.get("authorization") or ""
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    return auth.split(" ", 1)[1].strip()


# =============================================================================
# Embeddings
# =============================================================================
def _get_text_embedding(text: str) -> List[float]:
    resp = bedrock.invoke_model(
        modelId=MODEL_ID_EMBED, contentType="application/json",
        accept="application/json", body=json.dumps({"inputText": text}),
    )
    return json.loads(resp["body"].read())["embedding"]


def _get_multimodal_embedding(text: str, image_b64: Optional[str]) -> List[float]:
    body: dict = {"inputText": text[:2048]}
    if image_b64:
        body["inputImage"] = _strip_data_url(image_b64)
    resp = bedrock.invoke_model(
        modelId=MODEL_ID_EMBED, contentType="application/json",
        accept="application/json", body=json.dumps(body),
    )
    return json.loads(resp["body"].read())["embedding"]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


# =============================================================================
# ALEXA REMINDERS HELPERS  ← NEW
# =============================================================================
def _get_alexa_timezone(api_endpoint: str, device_id: str, api_access_token: str) -> str:
    """Fetch the Alexa device's configured timezone. Falls back to UTC."""
    url = f"{api_endpoint}/v2/devices/{device_id}/settings/System.timeZone"
    req = Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {api_access_token}")
    req.add_header("Accept", "application/json")
    try:
        with urlopen(req, timeout=8) as resp:
            return resp.read().decode("utf-8").strip().strip('"') or "UTC"
    except Exception as e:
        logger.warning(f"Could not fetch Alexa timezone: {e}")
        return "UTC"


def _create_alexa_reminder(
    api_endpoint: str,
    api_access_token: str,
    med_name: str,
    med_dose: str,
    trigger_time: str,   # "HH:MM" 24-hour
    repeat: str,         # "DAILY" | "WEEKDAYS" | "CUSTOM_DAYS"
    days: List[str],     # ["MO","TU",...] for CUSTOM_DAYS
    timezone: str,
) -> bool:
    """
    Create a single recurring Alexa reminder via the Reminders API.
    Returns True on success, False on failure.
    """
    # Build RRULE
    if repeat == "DAILY":
        rrule = "FREQ=DAILY"
    elif repeat == "WEEKDAYS":
        rrule = "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"
    elif repeat == "CUSTOM_DAYS" and days:
        rrule = f"FREQ=WEEKLY;BYDAY={','.join(days)}"
    else:
        rrule = "FREQ=DAILY"

    # Parse HH:MM
    try:
        hour, minute = [int(x) for x in trigger_time.split(":")]
    except Exception:
        hour, minute = 8, 0

    scheduled_time = f"2000-01-01T{hour:02d}:{minute:02d}:00"
    dose_str       = f" {med_dose}" if med_dose else ""
    speech_text    = f"Time to take your {med_name}{dose_str}."

    payload = {
        "requestTime": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "trigger": {
            "type":          "SCHEDULED_ABSOLUTE",
            "scheduledTime": scheduled_time,
            "timeZoneId":    timezone,
            "recurrence": {
                "startDateTime":  scheduled_time,
                "endDateTime":    "2099-12-31T23:59:59",
                "recurrenceRules": [rrule],
            },
        },
        "alertInfo": {
            "spokenInfo": {
                "content": [{"locale": "en-US", "text": speech_text}]
            }
        },
        "pushNotification": {"status": "ENABLED"},
    }

    url  = f"{api_endpoint}/v1/alerts/reminders"
    body = json.dumps(payload).encode("utf-8")
    req  = Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {api_access_token}")
    req.add_header("Content-Type",  "application/json")
    req.add_header("Accept",        "application/json")

    try:
        with urlopen(req, timeout=10) as resp:
            logger.info(f"Reminder created: {med_name} @ {trigger_time} — HTTP {resp.status}")
            return True
    except HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8")
        except Exception:
            pass
        logger.error(f"Reminder API {e.code} for {med_name}: {body_text}")
        return False
    except Exception as e:
        logger.error(f"Reminder creation error for {med_name}: {e}")
        return False


def _check_reminder_permission(body: dict) -> bool:
    """Returns True if the user has granted the Reminders permission."""
    try:
        scopes = (
            body.get("context", {})
                .get("System", {})
                .get("user", {})
                .get("permissions", {})
                .get("scopes", {})
        )
        status = scopes.get("alexa::alerts:reminders:skill:readwrite", {}).get("status")
        return status == "GRANTED"
    except Exception:
        return False


def _alexa_permissions_card_response(attributes: dict) -> JSONResponse:
    """Send a permissions consent card so the user can grant Reminders in the Alexa app."""
    return JSONResponse({
        "version": "1.0",
        "sessionAttributes": attributes,
        "response": {
            "outputSpeech": {
                "type": "PlainText",
                "text": (
                    "To set medication reminders, I need permission to create reminders on your device. "
                    "Please check your Alexa app to grant this permission, "
                    "then say 'set my medication reminders' again."
                ),
            },
            "card": {
                "type": "AskForPermissionsConsent",
                "permissions": ["alexa::alerts:reminders:skill:readwrite"],
            },
            "shouldEndSession": True,
        },
    })


# =============================================================================
# PAGES
# =============================================================================
@app.get("/")
def page_index():
    return FileResponse("index.html")


@app.get("/login")
def page_login():
    return FileResponse("login.html")


@app.get("/account")
def page_account():
    return FileResponse("accounts.html")


@app.get("/privacy")
def page_privacy():
    return FileResponse("privacy.html")


@app.get("/health")
def health():
    return {"ok": True, "region": AWS_REGION}


@app.get("/config")
def public_config():
    return {"google_maps_api_key": GOOGLE_MAPS_API_KEY}


# =============================================================================
# AUTH
# =============================================================================
@app.get("/auth/login")
def auth_login():
    _require_oauth_config()
    params = {
        "client_id": COGNITO_CLIENT_ID, "response_type": "code",
        "scope": COGNITO_SCOPES, "redirect_uri": COGNITO_REDIRECT_URI,
        "state": str(uuid.uuid4()),
    }
    if COGNITO_IDP:
        params["identity_provider"] = COGNITO_IDP
    return RedirectResponse(url=f"{COGNITO_DOMAIN}/oauth2/authorize?{urlencode(params)}", status_code=302)


@app.get("/auth/callback")
def auth_callback(
    code:  Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    db:    Session        = Depends(get_db),
):
    if error:
        logger.error(f"Cognito returned error: {error}")
        return {"error": error}
    _require_oauth_config()
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code.")

    token_url = f"{COGNITO_DOMAIN}/oauth2/token"
    form      = {
        "grant_type": "authorization_code", "client_id": COGNITO_CLIENT_ID,
        "code": code, "redirect_uri": COGNITO_REDIRECT_URI,
    }
    headers = {}
    if COGNITO_CLIENT_SECRET:
        basic = base64.b64encode(
            f"{COGNITO_CLIENT_ID}:{COGNITO_CLIENT_SECRET}".encode("utf-8")
        ).decode("utf-8")
        headers["Authorization"] = f"Basic {basic}"

    token_resp = _http_post_form(token_url, form, headers=headers)
    if token_resp.get("error"):
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {token_resp}")

    session_id, profile = _create_session(token_resp)
    sub   = (profile or {}).get("sub") or (profile or {}).get("username")
    email = (profile or {}).get("email")

    if sub:
        try:
            user = db.query(User).filter(User.sub == sub).first()
            if not user:
                db.add(User(sub=sub, email=email))
                db.commit()
        except Exception as e:
            logger.error(f"Failed to upsert user on login: {e}")

    resp = RedirectResponse(url="/account", status_code=302)
    secure_cookie = os.getenv("SECURE_COOKIES", "0") == "1"
    resp.set_cookie(
        key=SESSION_COOKIE, value=session_id, httponly=True,
        secure=secure_cookie, samesite="lax", max_age=7 * 24 * 3600, path="/",
    )
    return resp


@app.post("/auth/logout")
def auth_logout(req: FastAPIRequest):
    session_id = req.cookies.get(SESSION_COOKIE)
    if session_id and session_id in _sessions:
        del _sessions[session_id]
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(key=SESSION_COOKIE, path="/")
    return resp


@app.get("/me")
def me(req: FastAPIRequest, db: Session = Depends(get_db)):
    session_id = req.cookies.get(SESSION_COOKIE)
    s          = _get_session(session_id)
    if not s:
        return {"authenticated": False}

    profile = s.get("profile") or {}
    sub     = profile.get("sub") or profile.get("username") or ""
    user    = db.query(User).filter(User.sub == sub).first()

    prefs = (
        {
            "consent_store_history": user.consent_store_history,
            "data_retention_days":   user.data_retention_days,
            "age": user.age, "gender": user.gender,
            "height": user.height, "weight": user.weight,
            "allergies": user.allergies, "medical_history": user.medical_history,
        }
        if user else {"consent_store_history": False, "data_retention_days": 30}
    )

    return {
        "authenticated": True,
        "profile": {
            "sub":   sub,
            "email": profile.get("email"),
            "name":  profile.get("name") or profile.get("given_name"),
        },
        "prefs": prefs,
    }


@app.post("/prefs")
async def set_prefs(req: FastAPIRequest, db: Session = Depends(get_db)):
    body       = await req.json()
    session_id = req.cookies.get(SESSION_COOKIE)
    s          = _get_session(session_id)
    if not s:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    sub = (s.get("profile") or {}).get("sub") or ""
    if not sub:
        raise HTTPException(status_code=400, detail="Missing user identifier.")

    user = db.query(User).filter(User.sub == sub).first()
    if not user:
        user = User(sub=sub, email=(s.get("profile") or {}).get("email"))
        db.add(user)

    if "consent_store_history" in body: user.consent_store_history = bool(body["consent_store_history"])
    if "data_retention_days"   in body: user.data_retention_days   = max(1, min(int(body["data_retention_days"]), 3650))
    if "age"             in body: user.age             = body["age"]
    if "gender"          in body: user.gender          = body["gender"]
    if "height"          in body: user.height          = body["height"]
    if "weight"          in body: user.weight          = body["weight"]
    if "allergies"       in body: user.allergies       = body["allergies"]
    if "medical_history" in body: user.medical_history = body["medical_history"]

    db.commit()
    db.refresh(user)
    return {"ok": True, "message": "Profile saved successfully."}


# =============================================================================
# PRESCRIPTIONS  ← NEW
# =============================================================================
@app.get("/api/prescriptions")
def get_prescriptions(req: FastAPIRequest, db: Session = Depends(get_db)):
    """Return all saved prescriptions for the authenticated user."""
    session_id = req.cookies.get(SESSION_COOKIE)
    s          = _get_session(session_id)
    if not s:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    sub = (s.get("profile") or {}).get("sub") or ""
    if not sub:
        raise HTTPException(status_code=401, detail="Missing user identifier.")

    rows = (
        db.query(Prescription)
          .filter(Prescription.user_sub == sub)
          .order_by(Prescription.created_at.asc())
          .all()
    )
    return [
        {
            "id":      r.id,
            "name":    r.name,
            "dose":    r.dose   or "",
            "times":   r.times  or [],
            "repeat":  r.repeat or "DAILY",
            "days":    r.days   or [],
            "notes":   r.notes  or "",
            "savedAt": r.created_at.strftime("%Y-%m-%d") if r.created_at else "",
        }
        for r in rows
    ]


@app.post("/api/prescriptions")
async def save_prescriptions(req: FastAPIRequest, db: Session = Depends(get_db)):
    """
    Replace ALL prescriptions for the authenticated user with the new list.
    Body: { "prescriptions": [{name, dose, times, repeat, days, notes}, ...] }
    """
    session_id = req.cookies.get(SESSION_COOKIE)
    s          = _get_session(session_id)
    if not s:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    sub = (s.get("profile") or {}).get("sub") or ""
    if not sub:
        raise HTTPException(status_code=401, detail="Missing user identifier.")

    user = db.query(User).filter(User.sub == sub).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    body = await req.json()
    meds = body.get("prescriptions", [])

    # Delete existing and re-insert — simple and correct
    db.query(Prescription).filter(Prescription.user_sub == sub).delete()
    for med in meds:
        db.add(Prescription(
            user_sub = sub,
            name     = (med.get("name") or "").strip(),
            dose     = (med.get("dose") or "").strip(),
            times    = med.get("times")  or [],
            repeat   = med.get("repeat") or "DAILY",
            days     = med.get("days")   or [],
            notes    = (med.get("notes") or "").strip(),
        ))
    db.commit()
    return {"ok": True, "saved": len(meds)}


@app.delete("/api/prescriptions/{prescription_id}")
def delete_prescription(
    prescription_id: str, req: FastAPIRequest, db: Session = Depends(get_db)
):
    """Delete one prescription by ID."""
    session_id = req.cookies.get(SESSION_COOKIE)
    s          = _get_session(session_id)
    if not s:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    sub = (s.get("profile") or {}).get("sub") or ""
    row = db.query(Prescription).filter(
        Prescription.id == prescription_id,
        Prescription.user_sub == sub,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Prescription not found.")

    db.delete(row)
    db.commit()
    return {"ok": True}


# =============================================================================
# TRIAGE
# =============================================================================
@app.post("/api/triage", response_model=TextResponse)
def triage(req: TriageRequest, req_fastapi: FastAPIRequest, db: Session = Depends(get_db)):
    if not req.message and not req.image_base64:
        raise HTTPException(status_code=400, detail="Provide a message or an image.")

    session_id = req_fastapi.cookies.get(SESSION_COOKIE)
    s          = _get_session(session_id)
    u_sub      = (s.get("profile") or {}).get("sub") if s else None

    patient_ctx = ""
    if u_sub:
        user_row    = db.query(User).filter(User.sub == u_sub).first()
        patient_ctx = _build_patient_context_from_user(user_row)

    system_text = _triage_system_prompt(req.language)
    if patient_ctx:
        system_text = patient_ctx + "\n\n" + _allergy_safety_rule_block() + "\n\n" + system_text
    else:
        system_text = "If the user has allergies, ask them to share them before suggesting foods/medications.\n\n" + system_text

    messages     = _sanitize_history_for_nova(req.history)
    user_content = []
    if req.message:
        user_content.append({"text": req.message})
    if req.image_base64:
        fmt = _guess_image_format(_b64_to_bytes(req.image_base64, MAX_IMAGE_BYTES))
        user_content.append({"image": {"format": fmt, "source": {"bytes": _strip_data_url(req.image_base64)}}})
        user_content.append({"text": "If the image matters, mention what you can observe and suggest next steps."})
    messages.append({"role": "user", "content": user_content})

    ai_text = _nova_invoke(MODEL_ID_TEXT, {
        "schemaVersion":   "messages-v1",
        "system":          [{"text": system_text}],
        "messages":        messages,
        "inferenceConfig": {"maxTokens": int(req.max_tokens), "temperature": float(req.temperature), "topP": 0.9},
    })

    if u_sub:
        try:
            user_row = db.query(User).filter(User.sub == u_sub).first()
            if user_row and user_row.consent_store_history:
                history_list = [
                    h.model_dump() if hasattr(h, "model_dump") else h.dict()
                    for h in req.history
                ]
                history_list.append({"role": "user",      "text": req.message or "[Image Uploaded]"})
                history_list.append({"role": "assistant", "text": ai_text})
                title          = ((req.message or "Triage")[:47] + "...") if req.message and len(req.message) > 50 else (req.message or "Image Analysis")
                stable_chat_id = req.chat_id or str(uuid.uuid4())
                existing       = db.query(TriageSession).filter(TriageSession.id == stable_chat_id, TriageSession.user_sub == u_sub).first()
                if existing:
                    existing.messages = history_list; existing.title = title
                else:
                    db.add(TriageSession(id=stable_chat_id, user_sub=u_sub, title=title, messages=history_list))
                db.commit()
        except Exception as e:
            logger.error(f"Failed to save triage history: {e}")

    return TextResponse(text=ai_text)


@app.post("/api/first-aid-guide", response_model=FirstAidResponse)
def first_aid_guide(req: FirstAidRequest):
    target_lang   = _language_name(_normalize_lang_code(req.language))
    system_prompt = (
        "You are a medical first aid expert. Provide 3 to 4 step-by-step home care instructions. "
        f"Output ONLY a raw JSON array. Each object: 'step_text' in {target_lang.upper()}, 'image_prompt' in English."
    )
    clean_json = re.sub(r"```json|```", "", _nova_invoke(MODEL_ID_TEXT, {
        "schemaVersion": "messages-v1",
        "system":        [{"text": system_prompt}],
        "messages":      [{"role": "user", "content": [{"text": req.injury_description}]}],
        "inferenceConfig": {"maxTokens": 1000, "temperature": 0.2},
    })).strip()
    try:
        steps_data = json.loads(clean_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse steps from model.")
    return FirstAidResponse(steps=[
        FirstAidStep(
            step_text    = s.get("step_text", ""),
            image_base64 = _nova_canvas_invoke(s.get("image_prompt", "")) if s.get("image_prompt") else "",
        )
        for s in steps_data
    ])


# =============================================================================
# VISION
# =============================================================================
@app.post("/api/vision", response_model=TextResponse)
def vision(req: VisionRequest, req_fastapi: FastAPIRequest, db: Session = Depends(get_db)):
    system_list = [{"text": (
        "You are an expert medical image analysis assistant. "
        "Provide clear, structured, plain-language findings. "
        "Never diagnose. Always recommend professional medical evaluation."
    )}]

    if req.pdf_base64 and not req.image_base64:
        image_b64_for_model = _pdf_first_page_to_png_b64(_b64_to_bytes(req.pdf_base64, MAX_PDF_BYTES))
        image_fmt = "png"; raw_payload_b64 = req.pdf_base64; doc_type = req.doc_type
    else:
        if not req.image_base64:
            raise HTTPException(status_code=400, detail="Provide image_base64 or pdf_base64.")
        image_fmt           = _guess_image_format(_b64_to_bytes(req.image_base64, MAX_IMAGE_BYTES))
        image_b64_for_model = _strip_data_url(req.image_base64)
        raw_payload_b64     = req.image_base64; doc_type = req.doc_type

    ai_analysis = _nova_invoke(MODEL_ID_VISION, {
        "schemaVersion":   "messages-v1",
        "system":          system_list,
        "messages":        [{"role": "user", "content": [
            {"text": req.prompt.strip()},
            {"image": {"format": image_fmt, "source": {"bytes": image_b64_for_model}}},
        ]}],
        "inferenceConfig": {"maxTokens": int(req.max_tokens), "temperature": float(req.temperature)},
    })

    try:
        session_id = req_fastapi.cookies.get(SESSION_COOKIE)
        s          = _get_session(session_id)
        u_sub      = (s.get("profile") or {}).get("sub") if s else None
        if u_sub:
            user_row = db.query(User).filter(User.sub == u_sub).first()
            if user_row and user_row.consent_store_history:
                db.add(MedicalDocument(user_sub=u_sub, doc_type=doc_type, image_b64=raw_payload_b64 or "", report_html=ai_analysis))
                db.commit()
    except Exception as e:
        logger.error(f"Failed to save document: {e}")

    return TextResponse(text=ai_analysis)


# =============================================================================
# HISTORY
# =============================================================================
@app.get("/api/history", response_model=HistoryResponse)
def get_all_history(req: FastAPIRequest, db: Session = Depends(get_db)):
    session_id = req.cookies.get(SESSION_COOKIE)
    s          = _get_session(session_id)
    if not s:
        raise HTTPException(status_code=401, detail="Not authenticated")
    profile = s.get("profile") or {}
    u_sub   = profile.get("sub") or profile.get("username")
    if not u_sub:
        raise HTTPException(status_code=401, detail="Missing user identifier in session.")
    return HistoryResponse(
        triage    = db.query(TriageSession).filter(TriageSession.user_sub == u_sub).all(),
        documents = db.query(MedicalDocument).filter(MedicalDocument.user_sub == u_sub).all(),
    )


@app.post("/api/history/triage")
async def save_triage_history(req: FastAPIRequest, db: Session = Depends(get_db)):
    session_id = req.cookies.get(SESSION_COOKIE)
    s          = _get_session(session_id)
    if not s:
        raise HTTPException(status_code=401, detail="Not authenticated")
    u_sub = (s.get("profile") or {}).get("sub")
    if not u_sub:
        raise HTTPException(status_code=401, detail="Missing user identifier.")
    user_row = db.query(User).filter(User.sub == u_sub).first()
    if not (user_row and user_row.consent_store_history):
        return {"ok": True, "saved": False, "reason": "consent_off"}
    body     = await req.json()
    chat_id  = body.get("id")
    if not chat_id:
        raise HTTPException(status_code=400, detail="Missing chat id.")
    existing = db.query(TriageSession).filter(TriageSession.id == str(chat_id), TriageSession.user_sub == u_sub).first()
    if existing:
        existing.messages = body.get("messages", []); existing.title = body.get("title", "Triage Session")
    else:
        db.add(TriageSession(id=str(chat_id), user_sub=u_sub, title=body.get("title", "Triage Session"), messages=body.get("messages", [])))
    db.commit()
    return {"ok": True, "saved": True}


# =============================================================================
# VOICE
# =============================================================================
@app.post("/api/voice-turn", response_model=VoiceTurnResponse)
def voice_turn(req: VoiceTurnRequest):
    session_id          = str(uuid.uuid4())
    transcript_original = req.transcript.strip()
    source_lang         = _detect_language(transcript_original, req.detected_lang)
    messages            = _sanitize_history_for_nova(req.history)
    messages.append({"role": "user", "content": [{"text": transcript_original}]})

    reply_local = _nova_invoke(MODEL_ID_TEXT, {
        "schemaVersion":   "messages-v1",
        "system":          [{"text": _triage_system_prompt(req.detected_lang) + "\n\nVOICE OUTPUT RULES:\n- Short sentences.\n- Conversational.\n- Avoid heavy formatting."}],
        "messages":        messages,
        "inferenceConfig": {"maxTokens": int(req.max_tokens), "temperature": float(req.temperature), "topP": 0.9},
    })

    transcript_en = _translate_text(transcript_original, source_lang, "en") if source_lang != "en" else transcript_original
    reply_en      = _translate_text(reply_local, source_lang, "en") if req.include_english_reply and source_lang != "en" else None

    return VoiceTurnResponse(
        session_id=session_id, source_language=source_lang,
        source_language_name=_language_name(source_lang),
        transcript_original=transcript_original, transcript_en=transcript_en,
        reply_local=reply_local, reply_en=reply_en, internal_language=source_lang,
    )


# =============================================================================
# EMBEDDINGS
# =============================================================================
@app.post("/api/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    doc_id    = req.doc_id or str(uuid.uuid4())
    embedding = _get_multimodal_embedding(req.text, req.image_base64) if req.image_base64 else _get_text_embedding(req.text)
    _embed_store[doc_id] = {"text": req.text, "image_b64": req.image_base64, "embedding": embedding, "metadata": req.metadata}
    return EmbedResponse(doc_id=doc_id, embedding_dim=len(embedding))


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not _embed_store:
        return SearchResponse(hits=[])
    query_embedding = _get_text_embedding(req.query)
    scored = sorted(
        [(d_id, _cosine_similarity(query_embedding, doc["embedding"])) for d_id, doc in _embed_store.items()],
        key=lambda x: x[1], reverse=True
    )
    return SearchResponse(hits=[
        SearchHit(doc_id=d_id, score=round(s, 6), text=_embed_store[d_id]["text"], metadata=_embed_store[d_id]["metadata"])
        for d_id, s in scored[: req.top_k]
    ])


# =============================================================================
# COGNITO USERINFO
# =============================================================================
def _cognito_userinfo(access_token: str) -> Dict[str, Any]:
    if not COGNITO_DOMAIN:
        raise HTTPException(status_code=500, detail="Missing COGNITO_DOMAIN env var.")
    url = f"{COGNITO_DOMAIN.rstrip('/')}/oauth2/userInfo"
    req = Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {access_token}")
    try:
        with urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as e:
        logger.error(f"UserInfo HTTPError {e.code}")
        raise HTTPException(status_code=401, detail="Invalid/expired access token.")
    except URLError as e:
        logger.error(f"UserInfo URLError: {e}")
        raise HTTPException(status_code=502, detail="Could not reach Cognito userInfo endpoint.")


# =============================================================================
# ALEXA WEBHOOK  ← FULL REMINDERS IMPLEMENTATION
# =============================================================================
def _alexa_response(
    text: str,
    end_session: bool = False,
    link_account: bool = False,
    attributes: Optional[dict] = None,
) -> JSONResponse:
    body = {
        "version": "1.0",
        "sessionAttributes": attributes or {},
        "response": {
            "outputSpeech": {"type": "PlainText", "text": text},
            "shouldEndSession": bool(end_session),
        },
    }
    if link_account:
        body["response"]["card"]             = {"type": "LinkAccount"}
        body["response"]["shouldEndSession"] = True
    return JSONResponse(body)


def _alexa_get_access_token(body: dict) -> Optional[str]:
    return body.get("session", {}).get("user", {}).get("accessToken")


def _alexa_user_from_token(access_token: str) -> Dict[str, Any]:
    info = _cognito_userinfo(access_token)
    if not info or not info.get("sub"):
        raise HTTPException(status_code=401, detail="userInfo did not include sub.")
    return info


@app.post("/alexa/triage-turn")
async def alexa_webhook(req: FastAPIRequest, db: Session = Depends(get_db)):
    try:
        body = await req.json()

        # Verify skill ID
        application_id = (
            body.get("session", {}).get("application", {}).get("applicationId") or
            body.get("context", {}).get("System", {}).get("application", {}).get("applicationId")
        )
        if ALEXA_SKILL_ID and application_id and application_id != ALEXA_SKILL_ID:
            logger.warning(f"Rejected request from unknown skill: {application_id}")
            return _alexa_response("Unauthorized.", end_session=True)

        req_data      = body.get("request", {}) or {}
        req_type      = req_data.get("type")
        session_attrs = body.get("session", {}).get("attributes") or {}
        history       = session_attrs.get("history", [])
        if not isinstance(history, list):
            history = []

        # Alexa API context — needed for Reminders API and timezone
        system_ctx       = body.get("context", {}).get("System", {})
        api_endpoint     = system_ctx.get("apiEndpoint", "https://api.amazonalexa.com")
        api_access_token = system_ctx.get("apiAccessToken", "")
        device_id        = system_ctx.get("device", {}).get("deviceId", "")

        # ---- Built-in intents (before auth check) ----
        if req_type == "IntentRequest":
            intent_name = (req_data.get("intent") or {}).get("name")
            if intent_name in ("AMAZON.StopIntent", "AMAZON.CancelIntent", "AMAZON.NavigateHomeIntent"):
                return _alexa_response("Okay. Take care. Goodbye.", end_session=True, attributes={"history": []})
            if intent_name == "AMAZON.HelpIntent":
                return _alexa_response(
                    "You can describe your symptoms, or say 'set my medication reminders' "
                    "to schedule your medications. Say 'stop' anytime to exit.",
                    end_session=False, attributes={"history": history[-10:]},
                )

        # ---- Require account linking ----
        access_token = _alexa_get_access_token(body)
        if not access_token:
            return _alexa_response(
                "Please link your PulseNova account in the Alexa app to continue.",
                link_account=True,
            )

        try:
            user_info = _alexa_user_from_token(access_token)
            user_sub  = user_info.get("sub")
        except HTTPException as e:
            if e.status_code == 401:
                return _alexa_response(
                    "Your PulseNova login needs to be refreshed. "
                    "Please relink your account in the Alexa app.",
                    link_account=True,
                )
            raise

        # ---- LaunchRequest ----
        if req_type == "LaunchRequest":
            return _alexa_response(
                "Welcome to PulseNova. You can describe your symptoms, "
                "or say 'set my medication reminders' to schedule your medications.",
                end_session=False, attributes={"history": []},
            )

        # ---- IntentRequest ----
        if req_type == "IntentRequest":
            intent      = (req_data.get("intent") or {})
            intent_name = intent.get("name")
            slots       = intent.get("slots") or {}

            # ---- SetRemindersIntent ---- NEW
            if intent_name == "SetRemindersIntent":

                # 1. Check permission
                if not _check_reminder_permission(body):
                    return _alexa_permissions_card_response({"history": history[-10:]})

                # 2. Load prescriptions from DB
                prescriptions = (
                    db.query(Prescription)
                      .filter(Prescription.user_sub == user_sub)
                      .all()
                )

                if not prescriptions:
                    return _alexa_response(
                        "I don't see any saved medications in your PulseNova account. "
                        "Please add your prescriptions in the PulseNova app first, "
                        "then say 'set my medication reminders' again.",
                        end_session=False, attributes={"history": history[-10:]},
                    )

                # 3. Get device timezone
                tz = _get_alexa_timezone(api_endpoint, device_id, api_access_token)
                logger.info(f"Creating reminders for {user_sub}, tz={tz}, meds={len(prescriptions)}")

                # 4. Create reminders
                created   = 0
                failed    = 0
                med_names = []

                for rx in prescriptions:
                    times = rx.times or ["08:00"]
                    for t in times:
                        ok = _create_alexa_reminder(
                            api_endpoint     = api_endpoint,
                            api_access_token = api_access_token,
                            med_name         = rx.name,
                            med_dose         = rx.dose or "",
                            trigger_time     = t,
                            repeat           = rx.repeat or "DAILY",
                            days             = rx.days   or [],
                            timezone         = tz,
                        )
                        if ok: created += 1
                        else:  failed  += 1
                    if rx.name not in med_names:
                        med_names.append(rx.name)

                # 5. Respond
                if created > 0 and failed == 0:
                    med_list = ", ".join(med_names)
                    reply = (
                        f"Done! I've set {created} reminder{'s' if created > 1 else ''} "
                        f"for {med_list}. You'll be reminded at your scheduled times. "
                        f"Is there anything else I can help you with?"
                    )
                elif created > 0:
                    reply = (
                        f"I set {created} reminder{'s' if created > 1 else ''} successfully, "
                        f"but {failed} couldn't be created. Please try again for those."
                    )
                else:
                    reply = (
                        "I wasn't able to create the reminders. "
                        "Please make sure reminders are enabled in the Alexa app and try again."
                    )

                return _alexa_response(reply, end_session=False, attributes={"history": history[-10:]})

            # ---- FallbackIntent ----
            if intent_name == "AMAZON.FallbackIntent":
                return _alexa_response(
                    "Sorry, I didn't catch that. Describe your symptoms "
                    "or say 'set my medication reminders'.",
                    end_session=False, attributes={"history": history[-10:]},
                )

            # ---- Triage / CatchAll / GetMedicalData ----
            user_text = (
                slots.get("symptoms", {}).get("value") or
                slots.get("catchall", {}).get("value") or
                slots.get("query",    {}).get("value")
            )

            if not user_text:
                return _alexa_response(
                    "I didn't quite catch that. Could you repeat?",
                    end_session=False, attributes={"history": history[-10:]},
                )

            nova_history = history if (history and isinstance(history[0], dict) and "role" in history[0]) else []
            nova_history.append({"role": "user", "content": [{"text": user_text}]})

            user_row    = db.query(User).filter(User.sub == user_sub).first()
            patient_ctx = _build_patient_context_from_user(user_row)
            system_prompt = "You are PulseNova on Alexa. Short, conversational voice responses. ONE clarifying question at a time. Advise emergency services for life-threatening symptoms.\n\n"
            if patient_ctx:
                system_prompt = patient_ctx + "\n\n" + _allergy_safety_rule_block() + "\n\n" + system_prompt
            else:
                system_prompt = "If the user has allergies, ask them to share them before suggesting foods/medications.\n\n" + system_prompt

            try:
                reply = _nova_invoke(MODEL_ID_TEXT, {
                    "schemaVersion": "messages-v1",
                    "system":        [{"text": system_prompt}],
                    "messages":      nova_history[-10:],
                    "inferenceConfig": {"maxTokens": 200, "temperature": 0.3},
                }).strip()
            except Exception as e:
                logger.error(f"Bedrock Alexa Error: {e}")
                return _alexa_response(
                    "I'm having trouble reaching the medical model right now. Please try again.",
                    end_session=False, attributes={"history": nova_history[-10:]},
                )

            nova_history.append({"role": "assistant", "content": [{"text": reply}]})
            return _alexa_response(reply, end_session=False, attributes={"history": nova_history[-10:]})

        return _alexa_response(
            "Tell me your symptoms, or say 'set my medication reminders' to get started.",
            end_session=False, attributes={"history": history[-10:]},
        )

    except Exception as e:
        logger.error(f"Fatal Alexa Webhook Error: {e}")
        return _alexa_response(
            "Sorry, PulseNova encountered an internal error. Please try again.",
            end_session=False, attributes={"history": []},
        )
