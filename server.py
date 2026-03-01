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

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from database import init_db, get_db, User, TriageSession, VitalReading, MedicalDocument
from models import Base

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
MODEL_ID_TEXT   = os.getenv("MODEL_ID", "us.amazon.nova-lite-v2:0")
MODEL_ID_VISION = os.getenv("VISION_MODEL_ID", "us.amazon.nova-lite-v2:0")
MODEL_ID_EMBED  = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
MODEL_ID_CANVAS = os.getenv("CANVAS_MODEL_ID", "amazon.nova-canvas-v1:0")
MODEL_ID_SONIC  = os.getenv("SONIC_MODEL_ID",  "us.amazon.nova-sonic-v1:0")

NOVA_ACT_ENDPOINT = os.getenv("NOVA_ACT_ENDPOINT", "")
NOVA_ACT_API_KEY  = os.getenv("NOVA_ACT_API_KEY",  "")

VOICE_INTERNAL_LANGUAGE = os.getenv("VOICE_INTERNAL_LANGUAGE", "en")
TRANSLATE_ENABLED = os.getenv("TRANSLATE_ENABLED", "1") == "1"

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))
MAX_PDF_BYTES   = int(os.getenv("MAX_PDF_BYTES",   str(12 * 1024 * 1024)))
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# ---------------------------------------------------------------------------
# Cognito OAuth (Hosted UI) + Login with Amazon federation
# ---------------------------------------------------------------------------
COGNITO_DOMAIN = (os.getenv("COGNITO_DOMAIN") or "").rstrip("/")
COGNITO_CLIENT_ID = os.getenv("COGNITO_CLIENT_ID", "")
COGNITO_CLIENT_SECRET = os.getenv("COGNITO_CLIENT_SECRET", "")
COGNITO_REDIRECT_URI = os.getenv("COGNITO_REDIRECT_URI", "")
COGNITO_LOGOUT_REDIRECT_URI = os.getenv("COGNITO_LOGOUT_REDIRECT_URI", "")
COGNITO_SCOPES = os.getenv("COGNITO_SCOPES", "openid email profile")
COGNITO_IDP = os.getenv("COGNITO_IDP", "")  # e.g. LoginWithAmazon

# These are needed for proper JWT verification (recommended).
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID", "")
COGNITO_ISSUER = os.getenv("COGNITO_ISSUER", "")

# Session cookie name
SESSION_COOKIE = os.getenv("SESSION_COOKIE", "pulsenova_session")

BEDROCK_CONFIG = Config(
    connect_timeout=int(os.getenv("BEDROCK_CONNECT_TIMEOUT", "60")),
    read_timeout=int(os.getenv("BEDROCK_READ_TIMEOUT", "300")),
    retries={"max_attempts": int(os.getenv("BEDROCK_MAX_ATTEMPTS", "2"))},
)

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=BEDROCK_CONFIG)
_translate = boto3.client("translate", region_name=AWS_REGION, config=BEDROCK_CONFIG)
_comprehend = boto3.client("comprehend", region_name=AWS_REGION, config=BEDROCK_CONFIG)

# =============================================================================
# IN-MEMORY STORES
# =============================================================================
_embed_store: Dict[str, Dict[str, Any]] = {}

# Sessions: session_id -> {tokens, profile, created_at}
_sessions: Dict[str, Dict[str, Any]] = {}

# Database initialization function
def init_db():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if DATABASE_URL:
        engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(engine)
        print("Database tables initialized successfully.")
    else:
        print("DATABASE_URL not found. Skipping table initialization.")



@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("Database tables initialized successfully.")

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

# Serve /static/app.js and /static/styles.css from project root
app.mount("/static", StaticFiles(directory=".", html=False), name="static")



# =============================================================================
# PYDANTIC MODELS
# =============================================================================
Role = Literal["user", "assistant"]

class ChatTurn(BaseModel):
    role: Role
    text: str

class TriageRequest(BaseModel):
    message: str = Field(..., description="User message (symptoms, questions, etc.)")
    history: List[ChatTurn] = Field(default_factory=list, description="Prior chat turns")
    image_base64: Optional[str] = Field(default=None, description="Optional base64 image.")
    language: Optional[str] = Field(default="en-US", description="UI language hint.")
    temperature: float = 0.3
    max_tokens: int = 700

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, v):
        if v is None:
            raise ValueError("Message is required.")
        return v.strip()

class VisionRequest(BaseModel):
    image_base64: Optional[str] = Field(default=None, description="Base64 image.")
    pdf_base64: Optional[str] = Field(default=None, description="Base64 PDF.")
    prompt: str = Field(..., description="Instruction for the model.")
    language: Optional[str] = Field(default="en-US", description="UI language hint.")
    temperature: float = 0.2
    max_tokens: int = 900

class TextResponse(BaseModel):
    text: str

class FirstAidRequest(BaseModel):
    injury_description: str = Field(..., description="Description of the physical injury")
    language: Optional[str] = Field(default="en-US", description="Target language for steps")

class FirstAidStep(BaseModel):
    step_text: str
    image_base64: str

class FirstAidResponse(BaseModel):
    steps: List[FirstAidStep]

class VoiceTurnRequest(BaseModel):
    transcript: str = Field(...)
    detected_lang: Optional[str] = Field(default=None)
    history: List[ChatTurn] = Field(default_factory=list)
    include_english_reply: bool = Field(default=True)
    max_tokens: int = 350
    temperature: float = 0.3

class VoiceTurnResponse(BaseModel):
    session_id: str
    source_language: str
    source_language_name: str
    transcript_original: str
    transcript_en: str
    reply_local: str
    reply_en: Optional[str] = None
    internal_language: str = "en"

class EmbedRequest(BaseModel):
    text: str
    image_base64: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    doc_id: Optional[str] = None

class EmbedResponse(BaseModel):
    doc_id: str
    embedding_dim: int

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)

class SearchHit(BaseModel):
    doc_id: str
    score: float
    text: str
    metadata: dict

class SearchResponse(BaseModel):
    hits: List[SearchHit]

class AlexaTurnRequest(BaseModel):
    transcript: str
    locale: Optional[str] = "en-US"
    session: Optional[dict] = Field(default_factory=dict)

class AlexaTurnResponse(BaseModel):
    speech: str
    cardText: Optional[str] = None
    shouldEndSession: bool = False

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
    raise ValueError("Unsupported file format. Please upload PNG, JPEG, WEBP, or GIF.")

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
        "en": "English", "hi": "Hindi", "mr": "Marathi", "es": "Spanish",
        "fr": "French", "de": "German", "ar": "Arabic"
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
        resp = _translate.translate_text(Text=text, SourceLanguageCode=source_lang, TargetLanguageCode=target_lang)
        return resp.get("TranslatedText", text)
    except Exception as e:
        logger.warning(f"Translate API skipped/failed: {e}")
        return text

def _triage_system_prompt(ui_lang: Optional[str] = "en-US") -> str:
    c = _normalize_lang_code(ui_lang)
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
        "imageGenerationConfig": {"numberOfImages": 1, "height": 1024, "width": 1024, "cfgScale": 8.0},
    }
    try:
        resp = bedrock.invoke_model(
            modelId=MODEL_ID_CANVAS,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        data = json.loads(resp["body"].read())
        return data["images"][0] if "images" in data and data["images"] else ""
    except Exception as e:
        logger.error(f"Canvas invocation failed: {e}")
        return ""

def _pdf_first_page_to_png_b64(pdf_bytes: bytes) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(0)
        return base64.b64encode(page.get_pixmap(dpi=200).tobytes("png")).decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read PDF. Try a screenshot.")

# =============================================================================
# OAuth helpers (Cognito Hosted UI)
# =============================================================================
def _require_oauth_config():
    if not (COGNITO_DOMAIN and COGNITO_CLIENT_ID and COGNITO_REDIRECT_URI):
        raise HTTPException(
            status_code=500,
            detail="Missing Cognito OAuth configuration. Set COGNITO_DOMAIN, COGNITO_CLIENT_ID, COGNITO_REDIRECT_URI in .env",
        )

def _http_post_form(url: str, data: Dict[str, str], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    encoded = urlencode(data).encode("utf-8")
    req = Request(url, data=encoded, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urlopen(req, timeout=25) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
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
    s = _sessions.get(session_id)
    return s

def _create_session(tokens: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    session_id = str(uuid.uuid4())
    profile = {}
    id_token = tokens.get("id_token")
    if id_token:
        profile = _decode_jwt_no_verify(id_token) or {}
    _sessions[session_id] = {
        "tokens": tokens,
        "profile": profile,
        "created_at": int(time.time()),
    }
    return session_id, profile

def _decode_jwt_no_verify(token: str) -> Dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1] + "==="
        payload_json = base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8")
        return json.loads(payload_json)
    except Exception:
        return {}

# -----------------------------------------------------------------------------
# JWT verification for Cognito
# -----------------------------------------------------------------------------
_jwks_cache: Dict[str, Any] = {"keys": None, "fetched_at": 0}

def _cognito_issuer() -> str:
    if COGNITO_ISSUER:
        return os.getenv("COGNITO_ISSUER").rstrip('/')
    if COGNITO_USER_POOL_ID:
        return f"https://cognito-idp.{AWS_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}"
    return ""

def _fetch_jwks() -> Dict[str, Any]:
    issuer = _cognito_issuer()
    if not issuer:
        raise HTTPException(status_code=500, detail="Missing COGNITO_USER_POOL_ID or COGNITO_ISSUER for JWT verification.")
    now = int(time.time())
    if _jwks_cache["keys"] and (now - _jwks_cache["fetched_at"] < 3600):
        return _jwks_cache["keys"]
    jwks_url = f"{issuer.rstrip('/')}/.well-known/jwks.json"
    try:
        with urlopen(jwks_url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            _jwks_cache["keys"] = data
            _jwks_cache["fetched_at"] = now
            return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch JWKS: {e}")

def _verify_cognito_jwt(token: str, audience: Optional[str] = None) -> Dict[str, Any]:
    try:
        import jwt  # PyJWT
        from jwt.algorithms import RSAAlgorithm
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="PyJWT not installed. Add 'PyJWT' to requirements.txt to verify Cognito tokens securely.",
        )

    jwks = _fetch_jwks()
    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")
    if not kid:
        raise HTTPException(status_code=401, detail="Invalid token header (no kid).")

    key = None
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            key = k
            break
    if not key:
        raise HTTPException(status_code=401, detail="Signing key not found.")

    public_key = RSAAlgorithm.from_jwk(json.dumps(key))
    issuer = os.getenv("COGNITO_ISSUER")
    aud = audience or COGNITO_CLIENT_ID

    try:
        claims = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=aud,
            issuer=issuer,
            options={"require": ["exp", "iat"]},
        )
        return claims
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {e}")

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
        modelId=MODEL_ID_EMBED,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text}),
    )
    return json.loads(resp["body"].read())["embedding"]

def _get_multimodal_embedding(text: str, image_b64: Optional[str]) -> List[float]:
    body: dict = {"inputText": text[:2048]}
    if image_b64:
        body["inputImage"] = _strip_data_url(image_b64)
    resp = bedrock.invoke_model(
        modelId=MODEL_ID_EMBED,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    return json.loads(resp["body"].read())["embedding"]

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a, mag_b = sum(x * x for x in a) ** 0.5, sum(x * x for x in b) ** 0.5
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

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
# AUTH (Cognito Hosted UI)
# =============================================================================
@app.get("/auth/login")
def auth_login():
    _require_oauth_config()
    params = {
        "client_id": COGNITO_CLIENT_ID,
        "response_type": "code",
        "scope": COGNITO_SCOPES,
        "redirect_uri": COGNITO_REDIRECT_URI,
    }
    if COGNITO_IDP:
        params["identity_provider"] = COGNITO_IDP
    params["state"] = str(uuid.uuid4())

    url = f"{COGNITO_DOMAIN}/oauth2/authorize?{urlencode(params)}"
    return RedirectResponse(url=url, status_code=302)

@app.get("/auth/callback")
def auth_callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    if error:
        logger.error(f"Cognito returned error: {error}")
        return {"error": error, "message": "Check server logs for details"}

    _require_oauth_config()
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code.")

    token_url = f"{COGNITO_DOMAIN}/oauth2/token"
    form = {
        "grant_type": "authorization_code",
        "client_id": COGNITO_CLIENT_ID,
        "code": code,
        "redirect_uri": COGNITO_REDIRECT_URI,
    }

    headers = {}
    if COGNITO_CLIENT_SECRET:
        basic = base64.b64encode(f"{COGNITO_CLIENT_ID}:{COGNITO_CLIENT_SECRET}".encode("utf-8")).decode("utf-8")
        headers["Authorization"] = f"Basic {basic}"

    token_resp = _http_post_form(token_url, form, headers=headers)
    if token_resp.get("error"):
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {token_resp}")

    session_id, profile = _create_session(token_resp)

    resp = RedirectResponse(url="/account", status_code=302)
    secure_cookie = os.getenv("SECURE_COOKIES", "0") == "1"
    resp.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        max_age=7 * 24 * 3600,
        path="/",
    )
    return resp

@app.post("/auth/logout")
def auth_logout(req: FastAPIRequest):
    session_id = req.cookies.get(SESSION_COOKIE)
    if session_id and session_id in _sessions:
        del _sessions[session_id]

    resp = JSONResponse({"ok": True})
    secure_cookie = os.getenv("SECURE_COOKIES", "0") == "1"
    resp.delete_cookie(key=SESSION_COOKIE, path="/")
    return resp

@app.get("/me")
def me(req: FastAPIRequest, db: Session = Depends(get_db)):
    session_id = req.cookies.get(SESSION_COOKIE)
    s = _get_session(session_id)
    if not s:
        return {"authenticated": False}

    profile = s.get("profile") or {}
    sub = profile.get("sub") or profile.get("username") or ""
    
    # Fetch user from the PostgreSQL database
    user = db.query(User).filter(User.sub == sub).first()
    
    if user:
        prefs = {
            "consent_store_history": user.consent_store_history, 
            "data_retention_days": user.data_retention_days,
            "age": user.age,
            "gender": user.gender,
            "height": user.height,
            "weight": user.weight,
            "allergies": user.allergies,
            "medical_history": user.medical_history
        }
    else:
        # Default preferences if user hasn't saved any yet
        prefs = {"consent_store_history": False, "data_retention_days": 30}

    return {
        "authenticated": True,
        "profile": {
            "sub": sub,
            "email": profile.get("email"),
            "name": profile.get("name") or profile.get("given_name"),
        },
        "prefs": prefs,
    }

@app.post("/prefs")
async def set_prefs(req: FastAPIRequest, db: Session = Depends(get_db)):
    body = await req.json()
    session_id = req.cookies.get(SESSION_COOKIE)
    s = _get_session(session_id)
    if not s:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    profile = s.get("profile") or {}
    sub = profile.get("sub") or ""
    if not sub:
        raise HTTPException(status_code=400, detail="Missing user identifier in token.")

    # 1. Look up the user in the database
    user = db.query(User).filter(User.sub == sub).first()
    
    # 2. If they don't exist, create a new record
    if not user:
        user = User(sub=sub, email=profile.get("email"))
        db.add(user)

    # 3. Update their fields based on the incoming request
    if "consent_store_history" in body:
        user.consent_store_history = bool(body["consent_store_history"])
    if "data_retention_days" in body:
        user.data_retention_days = max(1, min(int(body["data_retention_days"]), 3650))
    
    # Update Medical Profile fields
    if "age" in body: user.age = body["age"]
    if "gender" in body: user.gender = body["gender"]
    if "height" in body: user.height = body["height"]
    if "weight" in body: user.weight = body["weight"]
    if "allergies" in body: user.allergies = body["allergies"]
    if "medical_history" in body: user.medical_history = body["medical_history"]

    # 4. Save to PostgreSQL
    db.commit()
    db.refresh(user)

    return {"ok": True, "message": "Profile saved successfully."}

# =============================================================================
# TRIAGE / VISION / VOICE
# =============================================================================
@app.post("/api/triage", response_model=TextResponse)
def triage(req: TriageRequest):
    if not req.message and not req.image_base64:
        raise HTTPException(status_code=400, detail="Provide a message or an image.")

    system_list = [{"text": _triage_system_prompt(req.language)}]
    messages = _sanitize_history_for_nova(req.history)

    user_content = []
    if req.message:
        user_content.append({"text": req.message})

    if req.image_base64:
        fmt = _guess_image_format(_b64_to_bytes(req.image_base64, MAX_IMAGE_BYTES))
        user_content.append({"image": {"format": fmt, "source": {"bytes": _strip_data_url(req.image_base64)}}})
        user_content.append({"text": "If the image matters, mention what you can observe and suggest next steps."})

    messages.append({"role": "user", "content": user_content})

    request_body = {
        "schemaVersion": "messages-v1",
        "system": system_list,
        "messages": messages,
        "inferenceConfig": {"maxTokens": int(req.max_tokens), "temperature": float(req.temperature), "topP": 0.9},
    }

    return TextResponse(text=_nova_invoke(MODEL_ID_TEXT, request_body))

@app.post("/api/first-aid-guide", response_model=FirstAidResponse)
def first_aid_guide(req: FirstAidRequest):
    target_lang = _language_name(_normalize_lang_code(req.language))
    system_prompt = (
        "You are a medical first aid expert. The user will describe a minor physical injury. "
        "You must provide 3 to 4 safe, practical, step-by-step home care instructions. "
        "CRITICAL: Output ONLY a raw, valid JSON array. Each object must have two keys: "
        f"'step_text' (instruction written entirely in {target_lang.upper()}) and "
        "'image_prompt' (a brief English description of the action to generate an illustration)."
    )

    request_body = {
        "schemaVersion": "messages-v1",
        "system": [{"text": system_prompt}],
        "messages": [{"role": "user", "content": [{"text": req.injury_description}]}],
        "inferenceConfig": {"maxTokens": 1000, "temperature": 0.2},
    }

    clean_json = re.sub(r"```json|```", "", _nova_invoke(MODEL_ID_TEXT, request_body)).strip()
    try:
        steps_data = json.loads(clean_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse instructional steps from the model.")

    results = []
    for step in steps_data:
        results.append(
            FirstAidStep(
                step_text=step.get("step_text", ""),
                image_base64=_nova_canvas_invoke(step.get("image_prompt", "")) if step.get("image_prompt") else "",
            )
        )
    return FirstAidResponse(steps=results)

@app.post("/api/vision", response_model=TextResponse)
def vision(req: VisionRequest):
    system_list = [{"text": _triage_system_prompt(req.language)}]

    if req.pdf_base64 and not req.image_base64:
        image_b64_for_model = _pdf_first_page_to_png_b64(_b64_to_bytes(req.pdf_base64, MAX_PDF_BYTES))
        image_fmt = "png"
    else:
        if not req.image_base64:
            raise HTTPException(status_code=400, detail="Provide image_base64 or pdf_base64.")
        image_fmt = _guess_image_format(_b64_to_bytes(req.image_base64, MAX_IMAGE_BYTES))
        image_b64_for_model = _strip_data_url(req.image_base64)

    messages = [
        {
            "role": "user",
            "content": [
                {"text": req.prompt.strip()},
                {"image": {"format": image_fmt, "source": {"bytes": image_b64_for_model}}},
            ],
        }
    ]
    request_body = {
        "schemaVersion": "messages-v1",
        "system": system_list,
        "messages": messages,
        "inferenceConfig": {"maxTokens": int(req.max_tokens), "temperature": float(req.temperature)},
    }
    return TextResponse(text=_nova_invoke(MODEL_ID_VISION, request_body))

@app.post("/api/voice-turn", response_model=VoiceTurnResponse)
def voice_turn(req: VoiceTurnRequest):
    session_id = str(uuid.uuid4())
    transcript_original = req.transcript.strip()
    source_lang = _detect_language(transcript_original, req.detected_lang)

    messages = _sanitize_history_for_nova(req.history)
    messages.append({"role": "user", "content": [{"text": transcript_original}]})

    voice_system_prompt = (
        _triage_system_prompt(req.detected_lang)
        + "\n\nVOICE OUTPUT RULES:\n- Keep responses smooth and conversational for speech.\n- Use short sentences.\n- Avoid heavy formatting."
    )

    request_body = {
        "schemaVersion": "messages-v1",
        "system": [{"text": voice_system_prompt}],
        "messages": messages,
        "inferenceConfig": {"maxTokens": int(req.max_tokens), "temperature": float(req.temperature), "topP": 0.9},
    }

    reply_local = _nova_invoke(MODEL_ID_TEXT, request_body)

    transcript_en = _translate_text(transcript_original, source_lang, "en") if source_lang != "en" else transcript_original
    reply_en = (
        _translate_text(reply_local, source_lang, "en")
        if req.include_english_reply and source_lang != "en"
        else None
    )

    return VoiceTurnResponse(
        session_id=session_id,
        source_language=source_lang,
        source_language_name=_language_name(source_lang),
        transcript_original=transcript_original,
        transcript_en=transcript_en,
        reply_local=reply_local,
        reply_en=reply_en,
        internal_language=source_lang,
    )

# =============================================================================
# EMBEDDINGS 
# =============================================================================
@app.post("/api/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    doc_id = req.doc_id or str(uuid.uuid4())
    embedding = _get_multimodal_embedding(req.text, req.image_base64) if req.image_base64 else _get_text_embedding(req.text)
    _embed_store[doc_id] = {"text": req.text, "image_b64": req.image_base64, "embedding": embedding, "metadata": req.metadata}
    return EmbedResponse(doc_id=doc_id, embedding_dim=len(embedding))

@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not _embed_store:
        return SearchResponse(hits=[])
    query_embedding = _get_text_embedding(req.query)
    scored = [(doc_id, _cosine_similarity(query_embedding, doc["embedding"])) for doc_id, doc in _embed_store.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    hits = [
        SearchHit(
            doc_id=d_id,
            score=round(s, 6),
            text=_embed_store[d_id]["text"],
            metadata=_embed_store[d_id]["metadata"],
        )
        for d_id, s in scored[: req.top_k]
    ]
    return SearchResponse(hits=hits)

# =============================================================================
# ALEXA (Native Webhook Handler)
# =============================================================================
def _alexa_response(text: str, end_session: bool = False, link_account: bool = False):
    """Formats the JSON exactly how Amazon Alexa expects it."""
    response_body = {
        "version": "1.0",
        "response": {
            "outputSpeech": {
                "type": "PlainText",
                "text": text
            },
            "shouldEndSession": end_session
        }
    }
    if link_account:
        # This triggers a card in the Alexa app prompting them to log in
        response_body["response"]["card"] = {"type": "LinkAccount"}
        response_body["response"]["shouldEndSession"] = True
        
    return JSONResponse(response_body)

@app.post("/alexa/triage-turn")
async def alexa_webhook(req: FastAPIRequest, db: Session = Depends(get_db)):
    body = await req.json()
    
    # 1. Extract the Cognito Access Token from Alexa's payload
    access_token = body.get("session", {}).get("user", {}).get("accessToken")
    
    if not access_token:
        return _alexa_response(
            "Welcome to Pulse Nova. Please link your account in the Alexa app to continue.", 
            link_account=True
        )

    # 2. Verify the User with Cognito
    try:
        claims = _verify_cognito_jwt(access_token)
        user_sub = claims.get("sub")
    except Exception as e:
        logger.error(f"Alexa Token Error: {e}")
        return _alexa_response(
            "Your session has expired. Please relink your account in the Alexa app.", 
            link_account=True
        )

    # 3. Parse the Intent (What the user actually asked)
    req_data = body.get("request", {})
    req_type = req_data.get("type")

    # If they just say "Alexa, open Pulse Nova"
    if req_type == "LaunchRequest":
        return _alexa_response("Welcome to Pulse Nova. You can describe your symptoms, or ask me to read your last lab report.")

    # If they use one of the specific commands we set up
    if req_type == "IntentRequest":
        intent_name = req_data.get("intent", {}).get("name")
        slots = req_data.get("intent", {}).get("slots", {})

        # --- COMMAND: "I have a fever and sore throat" ---
        if intent_name == "TriageIntent":
            symptoms = slots.get("symptoms", {}).get("value", "")
            if not symptoms:
                return _alexa_response("I didn't quite catch your symptoms. Could you repeat that?", end_session=False)

            system_prompt = (
                "You are PulseNova on Alexa. Keep responses short and conversational. "
                "Ask ONE clarifying question at a time. If it's a life-threatening emergency, "
                "advise them to call emergency services immediately."
            )
            bedrock_req = {
                "schemaVersion": "messages-v1",
                "system": [{"text": system_prompt}],
                "messages": [{"role": "user", "content": [{"text": symptoms}]}],
                "inferenceConfig": {"maxTokens": 200, "temperature": 0.3}
            }
            reply = _nova_invoke(MODEL_ID_TEXT, bedrock_req).strip()
            return _alexa_response(reply, end_session=False)

        # --- COMMAND: "What did my last lab report say?" ---
        elif intent_name == "GetMedicalDataIntent":
            query = slots.get("query", {}).get("value", "").lower()
            
            if "lab" in query:
                # Fetch their most recent lab from PostgreSQL
                doc = db.query(MedicalDocument).filter(
                    MedicalDocument.user_sub == user_sub, 
                    MedicalDocument.doc_type == 'lab'
                ).order_by(MedicalDocument.created_at.desc()).first()

                if doc:
                    # Ask Nova to summarize the HTML report into a spoken sentence
                    summary_req = {
                        "schemaVersion": "messages-v1",
                        "system": [{"text": "Summarize this medical lab report in one or two short sentences for a patient listening via a smart speaker. Be reassuring."}],
                        "messages": [{"role": "user", "content": [{"text": doc.report_html}]}],
                        "inferenceConfig": {"maxTokens": 150, "temperature": 0.2}
                    }
                    spoken_summary = _nova_invoke(MODEL_ID_TEXT, summary_req).strip()
                    return _alexa_response(f"Here is the summary of your last lab: {spoken_summary}", end_session=True)
                else:
                    return _alexa_response("I couldn't find any recent lab reports saved in your account. You can upload them using the Pulse Nova web app.", end_session=True)

    return _alexa_response("I'm not sure how to help with that yet.", end_session=False)
