# server.py
import os
import json
import base64
import logging
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

MODEL_ID_TEXT = os.getenv("MODEL_ID", "us.amazon.nova-micro-v1:0")
MODEL_ID_VISION = os.getenv("VISION_MODEL_ID", "us.amazon.nova-lite-v1:0")

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))  # 8MB

# Optional: expose to frontend if you want to fetch it from backend instead of hardcoding in index.html
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

BEDROCK_CONFIG = Config(
    connect_timeout=int(os.getenv("BEDROCK_CONNECT_TIMEOUT", "60")),
    read_timeout=int(os.getenv("BEDROCK_READ_TIMEOUT", "300")),
    retries={"max_attempts": int(os.getenv("BEDROCK_MAX_ATTEMPTS", "2"))},
)

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=BEDROCK_CONFIG)


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
# MODELS
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
        # allow empty text if image is present? frontend sends text usually; keep strict but trim
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
        f"AWS_REGION={'SET' if os.getenv('AWS_REGION') else 'MISSING'} "
        f"AWS_DEFAULT_REGION={'SET' if os.getenv('AWS_DEFAULT_REGION') else 'MISSING'}"
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

        # Defensive parsing
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
                "Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION (or AWS_DEFAULT_REGION)."
            ),
        )
    except ClientError as e:
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        logger.error(f"Bedrock ClientError: {error_msg}")
        raise HTTPException(status_code=500, detail=f"AWS Bedrock Error: {error_msg}")
    except Exception as e:
        logger.error(f"Unexpected model invocation error: {e}")
        raise HTTPException(status_code=500, detail="Model invocation failed due to an unexpected server error.")


# =============================================================================
# ROUTES
# =============================================================================
@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/favicon.ico")
def favicon():
    # Avoid noisy 404 logs if you don't have a favicon yet
    raise HTTPException(status_code=204, detail="No favicon")


@app.get("/health")
def health():
    return {
        "ok": True,
        "region": AWS_REGION,
        "default_region": os.getenv("AWS_DEFAULT_REGION"),
        "text_model": MODEL_ID_TEXT,
        "vision_model": MODEL_ID_VISION,
    }


@app.get("/debug-env")
def debug_env():
    """
    Safe-ish debug route: shows whether keys exist, not the secret values.
    Remove in production if you want.
    """
    return {
        "AWS_ACCESS_KEY_ID": "SET" if os.getenv("AWS_ACCESS_KEY_ID") else "MISSING",
        "AWS_SECRET_ACCESS_KEY": "SET" if os.getenv("AWS_SECRET_ACCESS_KEY") else "MISSING",
        "AWS_SESSION_TOKEN": "SET" if os.getenv("AWS_SESSION_TOKEN") else "MISSING",
        "AWS_REGION": os.getenv("AWS_REGION"),
        "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION"),
        "GOOGLE_MAPS_API_KEY": "SET" if GOOGLE_MAPS_API_KEY else "MISSING",
    }


@app.get("/config")
def public_config():
    """
    Optional frontend config route.
    If you want to avoid hardcoding GOOGLE_MAPS_API_KEY in index.html,
    fetch it from here.
    """
    return {
        "google_maps_api_key": GOOGLE_MAPS_API_KEY,
    }


@app.post("/api/triage", response_model=TextResponse)
def triage(req: TriageRequest):
    # Allow image-only requests with empty message
    if not req.message and not req.image_base64:
        raise HTTPException(status_code=400, detail="Provide a message or an image.")

    system_list = [
        {
            "text": (
                "You are PulseNova, a medical triage assistant.\n"
                "GOAL: Help users understand symptoms and decide next steps (Emergency, Urgent Care, Home care).\n"
                "RULES:\n"
                "1) NEVER diagnose. Use language like 'This could be consistent with...'\n"
                "2) If emergency warning signs (e.g., chest pain, severe breathing trouble, stroke signs, fainting, severe bleeding), "
                "tell them to call 911 / local emergency immediately.\n"
                "3) Ask ONE clarifying question at a time when needed.\n"
                "4) Be concise and practical.\n"
                "5) End with a short safety note: 'Not a diagnosis.'\n"
            )
        }
    ]

    messages = _sanitize_history_for_nova(req.history)

    user_content = []
    if req.message:
        user_content.append({"text": req.message})

    active_model_id = MODEL_ID_TEXT

    if req.image_base64:
        try:
            raw = _b64_to_bytes(req.image_base64)
            fmt = _guess_image_format(raw)
            active_model_id = MODEL_ID_VISION
            user_content.append(
                {
                    "image": {
                        "format": fmt,
                        # Bedrock Nova expects raw base64 string here
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

    text = _nova_invoke(active_model_id, request_body)
    return TextResponse(text=text)


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
                "You are a careful medical assistant that describes medical images and documents.\n"
                "Do NOT diagnose; describe observable patterns and suggest appropriate next steps.\n"
                "If the content suggests an emergency, advise seeking urgent medical care.\n"
                "Always include: 'Not a diagnosis.'\n"
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
# RUN
# =============================================================================
# Local:
#   python -m uvicorn server:app --reload --port 8000
#
# .env example:
#   AWS_ACCESS_KEY_ID=...
#   AWS_SECRET_ACCESS_KEY=...
#   AWS_REGION=us-east-1
#   MODEL_ID=us.amazon.nova-micro-v1:0
#   VISION_MODEL_ID=us.amazon.nova-lite-v1:0
#   GOOGLE_MAPS_API_KEY=...
#
# If you're deploying (Railway/Render/etc.), set env vars in the platform settings.
