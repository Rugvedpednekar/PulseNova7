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


# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("pulsenova")


# =============================================================================
# CONFIG
# =============================================================================
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", AWS_REGION)

# Text-only triage (Nova Micro is cheap/fast; good for chat)
MODEL_ID_TEXT = os.getenv("MODEL_ID", "us.amazon.nova-micro-v1:0")

# Vision tasks (Nova Lite/Pro)
MODEL_ID_VISION = os.getenv("VISION_MODEL_ID", "us.amazon.nova-lite-v1:0")

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))  # 8MB default

# Increase timeouts
BEDROCK_CONFIG = Config(
    connect_timeout=int(os.getenv("BEDROCK_CONNECT_TIMEOUT", "60")),
    read_timeout=int(os.getenv("BEDROCK_READ_TIMEOUT", "300")),
    retries={"max_attempts": int(os.getenv("BEDROCK_MAX_ATTEMPTS", "2"))},
)


# =============================================================================
# AWS / BEDROCK (LAZY CLIENT)
# =============================================================================
def _env_status(name: str) -> str:
    return "SET" if (os.getenv(name) and os.getenv(name).strip()) else "MISSING"


def get_bedrock_client():
    """
    Create a Bedrock Runtime client using explicit env vars.
    This avoids "import-time" client creation and makes Railway env/debugging easier.
    """
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

    akid = os.getenv("AWS_ACCESS_KEY_ID")
    sak = os.getenv("AWS_SECRET_ACCESS_KEY")
    token = os.getenv("AWS_SESSION_TOKEN")

    # Helpful logs (never print actual secret values)
    logger.info(
        "AWS env status: AWS_ACCESS_KEY_ID=%s AWS_SECRET_ACCESS_KEY=%s AWS_SESSION_TOKEN=%s AWS_REGION=%s AWS_DEFAULT_REGION=%s",
        _env_status("AWS_ACCESS_KEY_ID"),
        _env_status("AWS_SECRET_ACCESS_KEY"),
        _env_status("AWS_SESSION_TOKEN"),
        _env_status("AWS_REGION"),
        _env_status("AWS_DEFAULT_REGION"),
    )

    # If creds are missing, let boto3 raise NoCredentialsError later,
    # but we prefer a clearer early error.
    if not (akid and akid.strip()) or not (sak and sak.strip()):
        raise NoCredentialsError()

    session = boto3.session.Session(
        aws_access_key_id=akid.strip(),
        aws_secret_access_key=sak.strip(),
        aws_session_token=token.strip() if (token and token.strip()) else None,
        region_name=region,
    )
    return session.client("bedrock-runtime", config=BEDROCK_CONFIG)


# =============================================================================
# FASTAPI
# =============================================================================
app = FastAPI(title="PulseNova Server (Amazon Nova via Bedrock)")

cors_origins = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000",
).split(",")

# Trim whitespace to avoid subtle CORS mismatches
cors_origins = [o.strip() for o in cors_origins if o and o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# MODELS (REQUEST/RESPONSE)
# =============================================================================
Role = Literal["user", "assistant"]


class ChatTurn(BaseModel):
    role: Role
    text: str


class TriageRequest(BaseModel):
    message: str = Field(..., description="User message (symptoms, questions, etc.)")
    history: List[ChatTurn] = Field(default_factory=list, description="Prior chat turns")
    image_base64: Optional[str] = Field(
        default=None,
        description="Optional base64 image (no data: prefix). Use PNG/JPEG/WEBP/GIF.",
    )
    temperature: float = 0.3
    max_tokens: int = 700

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, v: str):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty or just whitespace.")
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
    raise ValueError("Unsupported file format. Please upload a valid PNG, JPEG, WEBP, or GIF image.")


def _nova_invoke(model_id: str, request_body: dict) -> str:
    """
    Nova Invoke format:
      request: { schemaVersion:"messages-v1", system:[...], messages:[...], inferenceConfig:{...} }
      response text: data["output"]["message"]["content"][0]["text"]
    """
    try:
        logger.info(f"Invoking Bedrock model: {model_id}")
        bedrock = get_bedrock_client()

        resp = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body),
        )
        data = json.loads(resp["body"].read())
        return data["output"]["message"]["content"][0]["text"]

    except (NoCredentialsError, EndpointConnectionError) as e:
        logger.error(f"AWS Connection Error: {e}")
        raise HTTPException(status_code=500, detail="AWS Bedrock credentials/connection error.")
    except ClientError as e:
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        logger.error(f"Bedrock ClientError: {error_msg}")
        raise HTTPException(status_code=500, detail=f"AWS Bedrock Error: {error_msg}")
    except Exception as e:
        logger.error(f"Unexpected error during invocation: {e}")
        raise HTTPException(status_code=500, detail="Model invocation failed due to an unexpected server error.")


# =============================================================================
# ROUTES
# =============================================================================
@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {
        "ok": True,
        "region": AWS_REGION,
        "default_region": AWS_DEFAULT_REGION,
        "text_model": MODEL_ID_TEXT,
        "vision_model": MODEL_ID_VISION,
    }


@app.get("/debug-env")
def debug_env():
    """
    Safe debug endpoint: shows SET/MISSING only (never values).
    Remove after debugging.
    """
    keys = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "MODEL_ID",
        "VISION_MODEL_ID",
        "CORS_ALLOW_ORIGINS",
    ]
    return {k: _env_status(k) for k in keys}


@app.post("/api/triage", response_model=TextResponse)
def triage(req: TriageRequest):
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

    # Convert history to Nova messages
    messages = []
    for turn in req.history[-20:]:
        messages.append({"role": turn.role, "content": [{"text": turn.text}]})

    # Current user message
    user_content = [{"text": req.message}]

    # Model Selection Logic
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
                        "source": {"bytes": _strip_data_url(req.image_base64)},
                    }
                }
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

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
                {"image": {"format": fmt, "source": {"bytes": _strip_data_url(req.image_base64)}}},
                {"text": req.prompt},
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
# RUN (uvicorn)
# =============================================================================
# Run locally:
#   python -m uvicorn server:app --reload --port 8000
#
# Railway Procfile:
#   web: uvicorn server:app --host 0.0.0.0 --port $PORT
#
# Required Railway Variables (Production):
#   AWS_ACCESS_KEY_ID
#   AWS_SECRET_ACCESS_KEY
#   AWS_REGION (or AWS_DEFAULT_REGION)
#   MODEL_ID
#   VISION_MODEL_ID
#   CORS_ALLOW_ORIGINS (include your Railway domain)
