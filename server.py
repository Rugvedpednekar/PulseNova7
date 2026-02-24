import os
import json
import base64
import logging
import uuid
from typing import List, Optional, Literal

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
# Nova 2 Lite is the new default text/reasoning model (multimodal-capable).
# Nova 2 Lite supports image input, so we use it as the vision model too.
# Override via env vars if needed.
# ---------------------------------------------------------------------------
MODEL_ID_TEXT   = os.getenv("MODEL_ID", "us.amazon.nova-lite-v2:0")
MODEL_ID_VISION = os.getenv("VISION_MODEL_ID", "us.amazon.nova-lite-v2:0")
MODEL_ID_EMBED  = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")

# Nova 2 Sonic (voice) — stub; set env var when you have access
MODEL_ID_SONIC  = os.getenv("SONIC_MODEL_ID",  "us.amazon.nova-sonic-v1:0")

# Nova Act — separate service; set endpoint + key via env
NOVA_ACT_ENDPOINT = os.getenv("NOVA_ACT_ENDPOINT", "")
NOVA_ACT_API_KEY  = os.getenv("NOVA_ACT_API_KEY",  "")

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))  # 8 MB

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

BEDROCK_CONFIG = Config(
    connect_timeout=int(os.getenv("BEDROCK_CONNECT_TIMEOUT", "60")),
    read_timeout=int(os.getenv("BEDROCK_READ_TIMEOUT", "300")),
    retries={"max_attempts": int(os.getenv("BEDROCK_MAX_ATTEMPTS", "2"))},
)

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=BEDROCK_CONFIG)


# =============================================================================
# IN-MEMORY EMBEDDING STORE  (MVP — replace with a real vector DB for prod)
# =============================================================================
# Structure: { id: { "text": str, "image_b64": str|None, "embedding": List[float], "metadata": dict } }
_embed_store: dict[str, dict] = {}


# =============================================================================
# FASTAPI
# =============================================================================
app = FastAPI(title="PulseNova Server (Amazon Nova 2 via Bedrock)")

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
    image_base64: Optional[str] = Field(default=None, description="Optional base64 image (no data: prefix).")
    temperature: float = 0.3
    max_tokens: int = 700

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, v):
        if v is None:
            raise ValueError("Message is required.")
        return v.strip()


class VisionRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 image (no data: prefix).")
    prompt: str = Field(..., description="Instruction for the model.")
    temperature: float = 0.2
    max_tokens: int = 900


class TextResponse(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Voice (Nova Sonic)
# ---------------------------------------------------------------------------
class VoiceRequest(BaseModel):
    """
    For a real Nova Sonic integration, audio_base64 would be a base64-encoded
    audio blob (WAV/PCM) from the browser. text_fallback is used when Sonic
    is not configured so we can still test the pipeline with the text model.
    """
    audio_base64: Optional[str] = Field(default=None, description="Base64 audio input (WAV/PCM).")
    text_fallback: Optional[str] = Field(default=None, description="Text input when audio is unavailable.")
    session_id: Optional[str] = Field(default=None, description="Conversation session ID.")


class VoiceResponse(BaseModel):
    session_id: str
    transcript: Optional[str] = None   # ASR output
    response_text: str                  # Model reply text
    audio_base64: Optional[str] = None  # TTS output (base64); None when Sonic not configured


# ---------------------------------------------------------------------------
# Nova Act (UI automation)
# ---------------------------------------------------------------------------
class ActRequest(BaseModel):
    """
    Nova Act performs browser/UI automation tasks described in natural language.
    start_url: the URL the agent should open first.
    task: natural-language description of what to do.
    """
    start_url: str = Field(..., description="URL to start automation from.")
    task: str = Field(..., description="Natural-language task for Nova Act.")
    session_id: Optional[str] = None


class ActResponse(BaseModel):
    session_id: str
    status: str           # "completed" | "running" | "failed" | "stubbed"
    result: Optional[str] = None
    screenshot_base64: Optional[str] = None


# ---------------------------------------------------------------------------
# Embeddings & Search
# ---------------------------------------------------------------------------
class EmbedRequest(BaseModel):
    """
    Embed a text (and optionally an image) and store it for later search.
    """
    text: str = Field(..., description="Text to embed.")
    image_base64: Optional[str] = Field(default=None, description="Optional image to embed alongside text.")
    metadata: dict = Field(default_factory=dict, description="Arbitrary metadata stored with the embedding.")
    doc_id: Optional[str] = Field(default=None, description="Optional stable ID; auto-generated if omitted.")


class EmbedResponse(BaseModel):
    doc_id: str
    embedding_dim: int


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language query to search the embedding store.")
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
    if not b64:
        return b64
    if b64.startswith("data:"):
        return b64.split(",", 1)[1]
    return b64


def _b64_to_bytes(b64: str) -> bytes:
    b64_clean = _strip_data_url(b64).strip()
    try:
        raw = base64.b64decode(b64_clean, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"Image too large (>{MAX_IMAGE_BYTES} bytes).")
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


def _aws_env_status_log():
    logger.info(
        "AWS env status: "
        f"AWS_ACCESS_KEY_ID={'SET' if os.getenv('AWS_ACCESS_KEY_ID') else 'MISSING'} "
        f"AWS_SECRET_ACCESS_KEY={'SET' if os.getenv('AWS_SECRET_ACCESS_KEY') else 'MISSING'} "
        f"AWS_SESSION_TOKEN={'SET' if os.getenv('AWS_SESSION_TOKEN') else 'MISSING'} "
        f"AWS_REGION={'SET' if os.getenv('AWS_REGION') else 'MISSING'}"
    )


def _sanitize_history_for_nova(history: List[ChatTurn]) -> List[dict]:
    """
    Nova expects the first message in 'messages' to be role='user'.
    Remove empty turns and trim leading assistant turns.
    """
    cleaned = []
    for turn in history[-20:]:
        text = (turn.text or "").strip()
        if not text:
            continue
        cleaned.append({"role": turn.role, "content": [{"text": text}]})

    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)

    return cleaned


def _nova_invoke(model_id: str, request_body: dict) -> str:
    """Invoke a Nova converse model and return the response text."""
    try:
        logger.info(f"Invoking Bedrock model: {model_id}")
        _aws_env_status_log()

        resp = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body),
        )
        data = json.loads(resp["body"].read())

        return (
            data.get("output", {})
            .get("message", {})
            .get("content", [{}])[0]
            .get("text", "")
        ) or "No response from model."

    except (NoCredentialsError, EndpointConnectionError) as e:
        logger.error(f"AWS Connection Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=(
                "AWS Bedrock credentials/connection error. "
                "Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION."
            ),
        )
    except ClientError as e:
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        logger.error(f"Bedrock ClientError: {error_msg}")
        raise HTTPException(status_code=500, detail=f"AWS Bedrock Error: {error_msg}")
    except Exception as e:
        logger.error(f"Unexpected model invocation error: {e}")
        raise HTTPException(status_code=500, detail="Model invocation failed due to an unexpected server error.")


def _get_text_embedding(text: str) -> List[float]:
    """
    Fetch a text embedding from Amazon Titan Embed Text v2 (or any compatible model).
    Titan Embed Text v2 also accepts an 'inputImage' field for multimodal embeddings,
    but image embedding requires the amazon.titan-embed-image-v1 model.

    To add image embedding:
      1. Change MODEL_ID_EMBED to "amazon.titan-embed-image-v1"
      2. Add "inputImage": <base64_string>  alongside "inputText" in the body below.
      3. The model will return a joint image+text embedding vector.
    """
    try:
        resp = bedrock.invoke_model(
            modelId=MODEL_ID_EMBED,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text}),
        )
        data = json.loads(resp["body"].read())
        return data["embedding"]
    except (NoCredentialsError, EndpointConnectionError) as e:
        logger.error(f"Embedding AWS error: {e}")
        raise HTTPException(status_code=500, detail="AWS credentials/connection error during embedding.")
    except ClientError as e:
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        logger.error(f"Embedding ClientError: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Embedding error: {error_msg}")
    except Exception as e:
        logger.error(f"Embedding unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Embedding failed.")


def _get_multimodal_embedding(text: str, image_b64: Optional[str]) -> List[float]:
    """
    Multimodal embedding using amazon.titan-embed-image-v1.
    Falls back to text-only if image is not provided.

    NOTE: To use this, set EMBED_MODEL_ID=amazon.titan-embed-image-v1 in your .env.
    The titan-embed-image-v1 model accepts both inputText and inputImage (base64).
    Max image size: 5 MB (base64 decoded). Supported formats: PNG, JPEG, GIF, BMP.
    """
    body: dict = {"inputText": text[:2048]}  # model cap
    if image_b64:
        # Strip data URL prefix if present
        body["inputImage"] = _strip_data_url(image_b64)

    try:
        resp = bedrock.invoke_model(
            modelId=MODEL_ID_EMBED,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        data = json.loads(resp["body"].read())
        return data["embedding"]
    except (NoCredentialsError, EndpointConnectionError) as e:
        raise HTTPException(status_code=500, detail="AWS credentials/connection error during embedding.")
    except ClientError as e:
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        raise HTTPException(status_code=500, detail=f"Embedding error: {error_msg}")
    except Exception as e:
        logger.error(f"Multimodal embedding error: {e}")
        raise HTTPException(status_code=500, detail="Multimodal embedding failed.")


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity (no numpy needed for MVP)."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# =============================================================================
# ROUTES — Core
# =============================================================================
@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/favicon.ico")
def favicon():
    raise HTTPException(status_code=204, detail="No favicon")


@app.get("/health")
def health():
    return {
        "ok": True,
        "region": AWS_REGION,
        "text_model": MODEL_ID_TEXT,
        "vision_model": MODEL_ID_VISION,
        "embed_model": MODEL_ID_EMBED,
        "sonic_model": MODEL_ID_SONIC,
        "nova_act_configured": bool(NOVA_ACT_ENDPOINT and NOVA_ACT_API_KEY),
        "embed_store_size": len(_embed_store),
    }


@app.get("/debug-env")
def debug_env():
    return {
        "AWS_ACCESS_KEY_ID": "SET" if os.getenv("AWS_ACCESS_KEY_ID") else "MISSING",
        "AWS_SECRET_ACCESS_KEY": "SET" if os.getenv("AWS_SECRET_ACCESS_KEY") else "MISSING",
        "AWS_SESSION_TOKEN": "SET" if os.getenv("AWS_SESSION_TOKEN") else "MISSING",
        "AWS_REGION": os.getenv("AWS_REGION"),
        "MODEL_ID": MODEL_ID_TEXT,
        "VISION_MODEL_ID": MODEL_ID_VISION,
        "EMBED_MODEL_ID": MODEL_ID_EMBED,
        "SONIC_MODEL_ID": MODEL_ID_SONIC,
        "NOVA_ACT_ENDPOINT": "SET" if NOVA_ACT_ENDPOINT else "MISSING",
        "NOVA_ACT_API_KEY": "SET" if NOVA_ACT_API_KEY else "MISSING",
        "GOOGLE_MAPS_API_KEY": "SET" if GOOGLE_MAPS_API_KEY else "MISSING",
    }


@app.get("/config")
def public_config():
    return {"google_maps_api_key": GOOGLE_MAPS_API_KEY}


# =============================================================================
# ROUTES — Triage (text + optional image)
# =============================================================================
@app.post("/api/triage", response_model=TextResponse)
def triage(req: TriageRequest):
    if not req.message and not req.image_base64:
        raise HTTPException(status_code=400, detail="Provide a message or an image.")

    system_list = [
        {
            "text": (
                "You are PulseNova, an AI-powered medical triage assistant.\n"
                "Your goal is to help users understand their symptoms and decide the most appropriate next step: "
                "Emergency (call 911), Urgent Care, or Home Care.\n\n"

                "CONVERSATION STYLE:\n"
                "- Be warm, calm, and reassuring. You are like a knowledgeable friend, not a clinical report.\n"
                "- For greetings, small talk, or non-medical questions, respond naturally and conversationally. No disclaimers.\n"
                "- For medical topics, be concise and practical. Avoid overwhelming the user with too much at once.\n"
                "- Ask ONE clarifying question at a time when you need more information.\n\n"

                "TRIAGE RULES:\n"
                "1) Never diagnose. Use language like 'This could be consistent with...' or "
                "'This sounds like it might be...' — but do not state a diagnosis as fact.\n"
                "2) Do not list multiple possible conditions at once. Offer the single most likely "
                "possibility with a brief caveat if needed.\n"
                "3) EMERGENCY: If the user describes chest pain, severe difficulty breathing, stroke symptoms "
                "(face drooping, arm weakness, speech difficulty), fainting, severe bleeding, or any life-threatening "
                "signs — immediately tell them to call 911 or go to the nearest emergency room. Do not ask follow-up questions first.\n"
                "4) HOME CARE: If the concern seems minor and manageable at home, say so clearly and confidently. "
                "Do not over-escalate. Reassure the user when reassurance is genuinely appropriate.\n"
                "5) SAFETY NET: Whenever you recommend home care, always include one specific warning sign "
                "that should prompt the user to seek further care.\n\n"

                "DISCLAIMER RULE:\n"
                "Only add 'Not a diagnosis' at the end of a response when you are actively discussing symptoms, "
                "conditions, medications, or health assessments. "
                "Do not add it to greetings, small talk, clarifying questions, or general conversation.\n\n"

                "IMPORTANT:\n"
                "You are not a replacement for a doctor. If a user's situation is unclear or potentially serious, "
                "always err on the side of recommending they speak to a healthcare professional."
            )
        }
    ]

    messages = _sanitize_history_for_nova(req.history)

    user_content = []
    if req.message:
        user_content.append({"text": req.message})

    # Nova 2 Lite supports image input natively — no model switch needed.
    if req.image_base64:
        try:
            raw = _b64_to_bytes(req.image_base64)
            fmt = _guess_image_format(raw)
            user_content.append(
                {
                    "image": {
                        "format": fmt,
                        "source": {"bytes": _strip_data_url(req.image_base64)},
                    }
                }
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

    if not user_content:
        raise HTTPException(status_code=400, detail="Empty request.")

    messages.append({"role": "user", "content": user_content})

    request_body = {
        "schemaVersion": "messages-v1",
        "system": system_list,
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": int(req.max_tokens),
            "temperature": float(req.temperature),
            "topP": 0.9,
        },
    }

    # Always use MODEL_ID_TEXT — Nova 2 Lite handles both text and image.
    text = _nova_invoke(MODEL_ID_TEXT, request_body)
    return TextResponse(text=text)


# =============================================================================
# ROUTES — Vision (dedicated image analysis endpoint)
# =============================================================================
@app.post("/api/vision", response_model=TextResponse)
def vision(req: VisionRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        raw = _b64_to_bytes(req.image_base64)
        fmt = _guess_image_format(raw)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    system_list = [
        {
            "text": (
                "You are PulseNova, an AI-powered medical triage assistant.\n"
                "Your goal is to help users understand their symptoms and decide the most appropriate next step: "
                "Emergency (call 911), Urgent Care, or Home Care.\n\n"
    
                "CONVERSATION STYLE:\n"
                "- Be warm, calm, and reassuring. You are like a knowledgeable friend, not a clinical report.\n"
                "- For greetings, small talk, or non-medical questions, respond naturally and conversationally. No disclaimers.\n"
                "- For medical topics, be concise and practical. Avoid overwhelming the user with too much at once.\n"
                "- Ask ONE clarifying question at a time when you need more information.\n\n"
    
                "TRIAGE RULES:\n"
                "1) Never diagnose. Use language like 'This could be consistent with...' or "
                "'This sounds like it might be...' — but do not state a diagnosis as fact.\n"
                "2) Do not list multiple possible conditions at once. Offer the single most likely "
                "possibility with a brief caveat if needed.\n"
                "3) EMERGENCY: If the user describes chest pain, severe difficulty breathing, stroke symptoms "
                "(face drooping, arm weakness, speech difficulty), fainting, severe bleeding, or any life-threatening "
                "signs — immediately tell them to call 911 or go to the nearest emergency room. Do not ask follow-up questions first.\n"
                "4) HOME CARE: If the concern seems minor and manageable at home, say so clearly and confidently. "
                "Do not over-escalate. Reassure the user when reassurance is genuinely appropriate.\n"
                "5) SAFETY NET: Whenever you recommend home care, always include one specific warning sign "
                "that should prompt the user to seek further care. "
                "Example: 'Rest and stay hydrated — but if your fever rises above 103°F or you develop "
                "difficulty breathing, seek care right away.'\n\n"
    
                "DISCLAIMER RULE:\n"
                "Only add 'Not a diagnosis' at the end of a response when you are actively discussing symptoms, "
                "conditions, medications, or health assessments. "
                "Do not add it to greetings, small talk, clarifying questions, or general conversation.\n\n"
    
                "IMPORTANT:\n"
                "You are not a replacement for a doctor. If a user's situation is unclear or potentially serious, "
                "always err on the side of recommending they speak to a healthcare professional."
            )
        }
    ]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": fmt,
                        "source": {"bytes": _strip_data_url(req.image_base64)},
                    }
                },
                {"text": req.prompt.strip()},
            ],
        }
    ]

    request_body = {
        "schemaVersion": "messages-v1",
        "system": system_list,
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": int(req.max_tokens),
            "temperature": float(req.temperature),
            "topP": 0.9,
        },
    }

    text = _nova_invoke(MODEL_ID_VISION, request_body)
    return TextResponse(text=text)


# =============================================================================
# ROUTES — Voice AI (Nova Sonic stub)
# =============================================================================
@app.post("/api/voice", response_model=VoiceResponse)
def voice(req: VoiceRequest):
    """
    Voice AI endpoint powered by Amazon Nova Sonic.

    STATUS: Stub implementation.

    Real integration steps:
    1. Nova Sonic uses a bidirectional streaming WebSocket, not a simple REST call.
       Use the Bedrock Runtime streaming API (invoke_model_with_response_stream or
       the newer bidirectional streaming API when generally available).
    2. Send audio chunks as PCM/WAV base64 and receive audio+text back.
    3. Set SONIC_MODEL_ID=us.amazon.nova-sonic-v1:0 in your .env.

    For now, if a text_fallback is provided this endpoint proxies to the text
    model so the rest of the pipeline (frontend, session tracking) can be tested
    end-to-end without Sonic access.
    """
    session_id = req.session_id or str(uuid.uuid4())

    # --- Real Sonic path (streaming) would go here ---
    # Example skeleton (not yet callable as a simple REST endpoint):
    #
    # if req.audio_base64:
    #     audio_bytes = base64.b64decode(_strip_data_url(req.audio_base64))
    #     # 1. Open bidirectional stream to Sonic
    #     # 2. Send audio_bytes in chunks
    #     # 3. Receive transcript + synthesized audio
    #     # 4. Return VoiceResponse(session_id=session_id, transcript=...,
    #     #                         response_text=..., audio_base64=...)
    #     pass

    if req.audio_base64 and not req.text_fallback:
        # Sonic not yet wired up — return 501 with helpful message
        raise HTTPException(
            status_code=501,
            detail=(
                "Nova Sonic audio streaming is not yet configured. "
                "Set SONIC_MODEL_ID and implement the bidirectional streaming client. "
                "Pass text_fallback to test the text pipeline in the meantime."
            ),
        )

    # Fallback: run through text model so frontend can be tested
    fallback_text = (req.text_fallback or "").strip()
    if not fallback_text:
        raise HTTPException(status_code=400, detail="Provide audio_base64 or text_fallback.")

    logger.info(f"[Voice] Session {session_id} — using text-model fallback.")
    request_body = {
        "schemaVersion": "messages-v1",
        "system": [{"text": "You are PulseNova, a medical triage assistant. Respond briefly and clearly."}],
        "messages": [{"role": "user", "content": [{"text": fallback_text}]}],
        "inferenceConfig": {"maxTokens": 300, "temperature": 0.3, "topP": 0.9},
    }
    response_text = _nova_invoke(MODEL_ID_TEXT, request_body)

    return VoiceResponse(
        session_id=session_id,
        transcript=fallback_text,   # echo back as "transcription"
        response_text=response_text,
        audio_base64=None,           # TTS output would go here when Sonic is live
    )


# =============================================================================
# ROUTES — Nova Act (UI automation stub)
# =============================================================================
@app.post("/api/act", response_model=ActResponse)
def act(req: ActRequest):
    """
    UI automation endpoint powered by Amazon Nova Act.

    STATUS: Stub implementation.

    Real integration steps:
    1. Install the Nova Act SDK: pip install nova-act
    2. Set NOVA_ACT_ENDPOINT and NOVA_ACT_API_KEY in your .env.
    3. Use the SDK to spin up a browser session, navigate to req.start_url,
       and execute req.task.

    Example (once SDK is available):

        from nova_act import NovaAct

        with NovaAct(
            starting_page=req.start_url,
            nova_act_api_key=NOVA_ACT_API_KEY,
        ) as agent:
            result = agent.act(req.task)
            screenshot_b64 = agent.screenshot()   # if supported

        return ActResponse(
            session_id=session_id,
            status="completed",
            result=str(result),
            screenshot_base64=screenshot_b64,
        )
    """
    session_id = req.session_id or str(uuid.uuid4())

    if not NOVA_ACT_ENDPOINT or not NOVA_ACT_API_KEY:
        logger.warning(f"[Act] Session {session_id} — Nova Act not configured.")
        return ActResponse(
            session_id=session_id,
            status="stubbed",
            result=(
                f"Nova Act is not yet configured. "
                f"Set NOVA_ACT_ENDPOINT and NOVA_ACT_API_KEY, then integrate the Nova Act SDK. "
                f"Received task: '{req.task}' for URL: '{req.start_url}'."
            ),
        )

    # TODO: replace with real Nova Act SDK call (see docstring above)
    raise HTTPException(
        status_code=501,
        detail="Nova Act SDK integration not yet implemented. See /api/act docstring.",
    )


# =============================================================================
# ROUTES — Embeddings & Semantic Search
# =============================================================================
@app.post("/api/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    """
    Embed text (+ optional image) and store in the in-memory vector store.

    Model routing:
    - Text only  → amazon.titan-embed-text-v2:0  (default EMBED_MODEL_ID)
    - Text+image → amazon.titan-embed-image-v1   (set EMBED_MODEL_ID to this for multimodal)

    The stored document can be retrieved via /api/search.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty.")

    doc_id = req.doc_id or str(uuid.uuid4())

    if req.image_base64:
        embedding = _get_multimodal_embedding(req.text, req.image_base64)
    else:
        embedding = _get_text_embedding(req.text)

    _embed_store[doc_id] = {
        "text": req.text,
        "image_b64": req.image_base64,
        "embedding": embedding,
        "metadata": req.metadata,
    }

    logger.info(f"[Embed] Stored doc_id={doc_id}, dim={len(embedding)}, has_image={req.image_base64 is not None}")
    return EmbedResponse(doc_id=doc_id, embedding_dim=len(embedding))


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    Semantic search over the in-memory embedding store.
    Embeds the query and returns the top-k most similar documents by cosine similarity.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty.")

    if not _embed_store:
        return SearchResponse(hits=[])

    query_embedding = _get_text_embedding(req.query)

    scored = [
        (doc_id, _cosine_similarity(query_embedding, doc["embedding"]))
        for doc_id, doc in _embed_store.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[: req.top_k]

    hits = [
        SearchHit(
            doc_id=doc_id,
            score=round(score, 6),
            text=_embed_store[doc_id]["text"],
            metadata=_embed_store[doc_id]["metadata"],
        )
        for doc_id, score in top
    ]

    logger.info(f"[Search] query='{req.query[:60]}' → {len(hits)} hits")
    return SearchResponse(hits=hits)


@app.delete("/api/embed/{doc_id}")
def delete_embed(doc_id: str):
    """Remove a document from the embedding store."""
    if doc_id not in _embed_store:
        raise HTTPException(status_code=404, detail=f"doc_id '{doc_id}' not found.")
    del _embed_store[doc_id]
    return {"deleted": doc_id}


@app.get("/api/embed")
def list_embeds():
    """List all stored document IDs and their metadata (no embeddings returned)."""
    return {
        doc_id: {
            "text_preview": doc["text"][:80],
            "has_image": doc["image_b64"] is not None,
            "metadata": doc["metadata"],
        }
        for doc_id, doc in _embed_store.items()
    }


# =============================================================================
# RUN
# =============================================================================
# Local dev:
#   python -m uvicorn server:app --reload --port 8000
#
# .env example:
#   AWS_ACCESS_KEY_ID=...
#   AWS_SECRET_ACCESS_KEY=...
#   AWS_REGION=us-east-1
#
#   # Text + Vision (Nova 2 Lite — multimodal, handles both)
#   MODEL_ID=us.amazon.nova-lite-v2:0
#   VISION_MODEL_ID=us.amazon.nova-lite-v2:0
#
#   # Embeddings — choose one:
#   #   Text only:      amazon.titan-embed-text-v2:0   (default)
#   #   Text + image:   amazon.titan-embed-image-v1
#   EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
#
#   # Voice (Nova Sonic — enable when access is granted)
#   SONIC_MODEL_ID=us.amazon.nova-sonic-v1:0
#
#   # Nova Act (UI automation — enable when SDK is available)
#   NOVA_ACT_ENDPOINT=https://...
#   NOVA_ACT_API_KEY=...
#
#   GOOGLE_MAPS_API_KEY=...
