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
MODEL_ID_TEXT   = os.getenv("MODEL_ID", "us.amazon.nova-lite-v2:0")
MODEL_ID_VISION = os.getenv("VISION_MODEL_ID", "us.amazon.nova-lite-v2:0")
MODEL_ID_EMBED  = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
MODEL_ID_CANVAS = os.getenv("CANVAS_MODEL_ID", "amazon.nova-canvas-v1:0")

MODEL_ID_SONIC  = os.getenv("SONIC_MODEL_ID",  "us.amazon.nova-sonic-v1:0")

NOVA_ACT_ENDPOINT = os.getenv("NOVA_ACT_ENDPOINT", "")
NOVA_ACT_API_KEY  = os.getenv("NOVA_ACT_API_KEY",  "")

# Multilingual layer
VOICE_INTERNAL_LANGUAGE = os.getenv("VOICE_INTERNAL_LANGUAGE", "en")
TRANSLATE_ENABLED = os.getenv("TRANSLATE_ENABLED", "1") == "1"

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))
MAX_PDF_BYTES   = int(os.getenv("MAX_PDF_BYTES",   str(12 * 1024 * 1024)))
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

BEDROCK_CONFIG = Config(
    connect_timeout=int(os.getenv("BEDROCK_CONNECT_TIMEOUT", "60")),
    read_timeout=int(os.getenv("BEDROCK_READ_TIMEOUT", "300")),
    retries={"max_attempts": int(os.getenv("BEDROCK_MAX_ATTEMPTS", "2"))},
)

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=BEDROCK_CONFIG)
_translate = boto3.client("translate", region_name=AWS_REGION, config=BEDROCK_CONFIG)
_comprehend = boto3.client("comprehend", region_name=AWS_REGION, config=BEDROCK_CONFIG)

# =============================================================================
# IN-MEMORY EMBEDDING STORE
# =============================================================================
_embed_store: Dict[str, Dict[str, Any]] = {}

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
        if v is None: raise ValueError("Message is required.")
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

class VoiceRequest(BaseModel):
    audio_base64: Optional[str] = Field(default=None)
    text_fallback: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)

class VoiceResponse(BaseModel):
    session_id: str
    transcript: Optional[str] = None
    response_text: str
    audio_base64: Optional[str] = None

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

class ActRequest(BaseModel):
    start_url: str
    task: str
    session_id: Optional[str] = None

class ActResponse(BaseModel):
    session_id: str
    status: str
    result: Optional[str] = None
    screenshot_base64: Optional[str] = None

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

# =============================================================================
# HELPERS
# =============================================================================
def _strip_data_url(b64: str) -> str:
    if not b64: return b64
    if b64.startswith("data:"): return b64.split(",", 1)[1]
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
    if raw.startswith(b"\x89PNG\r\n\x1a\n"): return "png"
    if raw.startswith(b"\xff\xd8\xff"): return "jpeg"
    if raw.startswith(b"RIFF") and b"WEBP" in raw[8:16]: return "webp"
    if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"): return "gif"
    raise ValueError("Unsupported file format. Please upload PNG, JPEG, WEBP, or GIF.")

def _aws_env_status_log():
    logger.info(
        "AWS env status: "
        f"AWS_ACCESS_KEY_ID={'SET' if os.getenv('AWS_ACCESS_KEY_ID') else 'MISSING'} "
        f"AWS_SECRET_ACCESS_KEY={'SET' if os.getenv('AWS_SECRET_ACCESS_KEY') else 'MISSING'} "
        f"AWS_REGION={'SET' if os.getenv('AWS_REGION') else 'MISSING'}"
    )

def _sanitize_history_for_nova(history: List[ChatTurn]) -> List[dict]:
    cleaned = []
    for turn in history[-20:]:
        text = (turn.text or "").strip()
        if not text: continue
        cleaned.append({"role": turn.role, "content": [{"text": text}]})
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    return cleaned

def _normalize_lang_code(code: Optional[str]) -> str:
    if not code: return ""
    code = code.strip().lower().replace("_", "-")
    if "-" in code: code = code.split("-")[0]
    return re.sub(r"[^a-z]", "", code)[:5]

def _language_name(code: str) -> str:
    return {
        "en": "English", "hi": "Hindi", "mr": "Marathi", "es": "Spanish",
        "fr": "French", "de": "German", "ar": "Arabic"
    }.get(code, code or "Unknown")

def _detect_language(text: str, browser_hint: Optional[str] = None) -> str:
    hint = _normalize_lang_code(browser_hint)
    if hint: return hint
    if re.search(r"[\u0900-\u097F]", text): return "hi"
    return "en"

def _translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Best-effort translation strictly for the UI English preview panel."""
    if not TRANSLATE_ENABLED or not text or source_lang == target_lang: return text
    try:
        resp = _translate.translate_text(Text=text, SourceLanguageCode=source_lang, TargetLanguageCode=target_lang)
        return resp.get("TranslatedText", text)
    except Exception as e:
        logger.warning(f"Translate API skipped/failed: {e}")
        return text

def _triage_system_prompt(ui_lang: Optional[str] = "en-US") -> str:
    c = _normalize_lang_code(ui_lang)
    lang_name = _language_name(c)
    
    # FIX 1: Language instruction moved to the VERY TOP and made extremely strict
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
        if t0: return t0
    if "results" in data and data["results"]: return data["results"][0].get("outputText", "") or ""
    if "completion" in data and isinstance(data["completion"], str): return data["completion"]
    if "text" in data and isinstance(data["text"], str): return data["text"]
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
    enhanced_prompt = f"Clean, minimalist vector illustrations, medical instructional infographic style, flat colors, white background. {prompt}"
    body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": enhanced_prompt[:1000]},
        "imageGenerationConfig": {"numberOfImages": 1, "height": 1024, "width": 1024, "cfgScale": 8.0}
    }
    try:
        resp = bedrock.invoke_model(modelId=MODEL_ID_CANVAS, contentType="application/json", accept="application/json", body=json.dumps(body))
        data = json.loads(resp["body"].read())
        return data["images"][0] if "images" in data and data["images"] else ""
    except Exception as e:
        logger.error(f"Canvas invocation failed: {e}")
        return ""

def _pdf_first_page_to_png_b64(pdf_bytes: bytes) -> str:
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(0)
        return base64.b64encode(page.get_pixmap(dpi=200).tobytes("png")).decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read PDF. Try a screenshot.")

# =============================================================================
# Embeddings
# =============================================================================
def _get_text_embedding(text: str) -> List[float]:
    resp = bedrock.invoke_model(modelId=MODEL_ID_EMBED, contentType="application/json", accept="application/json", body=json.dumps({"inputText": text}))
    return json.loads(resp["body"].read())["embedding"]

def _get_multimodal_embedding(text: str, image_b64: Optional[str]) -> List[float]:
    body: dict = {"inputText": text[:2048]}
    if image_b64: body["inputImage"] = _strip_data_url(image_b64)
    resp = bedrock.invoke_model(modelId=MODEL_ID_EMBED, contentType="application/json", accept="application/json", body=json.dumps(body))
    return json.loads(resp["body"].read())["embedding"]

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a, mag_b = sum(x * x for x in a) ** 0.5, sum(x * x for x in b) ** 0.5
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

# =============================================================================
# ROUTES
# =============================================================================
@app.get("/")
def index(): return FileResponse("index.html")

@app.get("/health")
def health(): return {"ok": True, "region": AWS_REGION}

@app.get("/config")
def public_config(): return {"google_maps_api_key": GOOGLE_MAPS_API_KEY}

# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------
@app.post("/api/triage", response_model=TextResponse)
def triage(req: TriageRequest):
    if not req.message and not req.image_base64:
        raise HTTPException(status_code=400, detail="Provide a message or an image.")

    system_list = [{"text": _triage_system_prompt(req.language)}]
    messages = _sanitize_history_for_nova(req.history)

    user_content = []
    if req.message: user_content.append({"text": req.message})

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

# ---------------------------------------------------------------------------
# First Aid (Visual Workflow)
# ---------------------------------------------------------------------------
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
        results.append(FirstAidStep(
            step_text=step.get("step_text", ""), 
            image_base64=_nova_canvas_invoke(step.get("image_prompt", "")) if step.get("image_prompt") else ""
        ))
    return FirstAidResponse(steps=results)

# ---------------------------------------------------------------------------
# Vision (X-Ray & Labs)
# ---------------------------------------------------------------------------
@app.post("/api/vision", response_model=TextResponse)
def vision(req: VisionRequest):
    system_list = [{"text": _triage_system_prompt(req.language)}]
    
    if req.pdf_base64 and not req.image_base64:
        image_b64_for_model = _pdf_first_page_to_png_b64(_b64_to_bytes(req.pdf_base64, MAX_PDF_BYTES))
        image_fmt = "png"
    else:
        image_fmt = _guess_image_format(_b64_to_bytes(req.image_base64, MAX_IMAGE_BYTES))
        image_b64_for_model = _strip_data_url(req.image_base64)

    messages = [{"role": "user", "content": [{"text": req.prompt.strip()}, {"image": {"format": image_fmt, "source": {"bytes": image_b64_for_model}}}]}]
    request_body = {
        "schemaVersion": "messages-v1",
        "system": system_list,
        "messages": messages,
        "inferenceConfig": {"maxTokens": int(req.max_tokens), "temperature": float(req.temperature)},
    }
    return TextResponse(text=_nova_invoke(MODEL_ID_VISION, request_body))

# ---------------------------------------------------------------------------
# Voice Concierge (Native Language)
# ---------------------------------------------------------------------------
@app.post("/api/voice-turn", response_model=VoiceTurnResponse)
def voice_turn(req: VoiceTurnRequest):
    session_id = str(uuid.uuid4())
    transcript_original = req.transcript.strip()
    source_lang = _detect_language(transcript_original, req.detected_lang)

    messages = _sanitize_history_for_nova(req.history)
    messages.append({"role": "user", "content": [{"text": transcript_original}]})

    # FIX 2: Pass the local language DIRECTLY into the system prompt. Nova translates natively!
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

    # Nova will now output Marathi/Hindi naturally.
    reply_local = _nova_invoke(MODEL_ID_TEXT, request_body)

    # Optional: Best effort to show English preview in UI
    transcript_en = _translate_text(transcript_original, source_lang, "en") if source_lang != "en" else transcript_original
    reply_en = _translate_text(reply_local, source_lang, "en") if req.include_english_reply and source_lang != "en" else None

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

# ---------------------------------------------------------------------------
# Embeddings (Unchanged)
# ---------------------------------------------------------------------------
@app.post("/api/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    doc_id = req.doc_id or str(uuid.uuid4())
    embedding = _get_multimodal_embedding(req.text, req.image_base64) if req.image_base64 else _get_text_embedding(req.text)
    _embed_store[doc_id] = {"text": req.text, "image_b64": req.image_base64, "embedding": embedding, "metadata": req.metadata}
    return EmbedResponse(doc_id=doc_id, embedding_dim=len(embedding))

@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not _embed_store: return SearchResponse(hits=[])
    query_embedding = _get_text_embedding(req.query)
    scored = [(doc_id, _cosine_similarity(query_embedding, doc["embedding"])) for doc_id, doc in _embed_store.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    hits = [SearchHit(doc_id=d_id, score=round(s, 6), text=_embed_store[d_id]["text"], metadata=_embed_store[d_id]["metadata"]) for d_id, s in scored[: req.top_k]]
    return SearchResponse(hits=hits)
