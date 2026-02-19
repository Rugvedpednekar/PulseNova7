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
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Text-only triage (Nova Micro is cheap/fast; good for chat)
MODEL_ID_TEXT = os.getenv("MODEL_ID", "us.amazon.nova-micro-v1:0")

# Vision tasks (X-ray/Labs/Chat Images need a vision-capable model: Nova Lite/Pro)
MODEL_ID_VISION = os.getenv("VISION_MODEL_ID", "us.amazon.nova-lite-v1:0")

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))  # 8MB default

# Increase timeouts (Nova docs recommend longer SDK timeouts for inference)
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

# Security: Default to localhost instead of "*" to prevent unauthorized access
cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
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
        description="Optional base64 image (no data: prefix). Use PNG/JPEG.",
    )
    temperature: float = 0.3
    max_tokens: int = 700

    @field_validator('message')
    @classmethod
    def message_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty or just whitespace.')
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
    # If caller accidentally sends a data URL, strip it.
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
    # Basic magic bytes detection for png/jpeg/webp/gif
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if raw.startswith(b"RIFF") and b"WEBP" in raw[8:16]:
        return "webp"
    if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
        return "gif"
    
    # Strict validation: reject unknown formats to prevent Bedrock errors
    raise ValueError("Unsupported file format. Please upload a valid PNG, JPEG, WEBP, or GIF image.")

def _nova_invoke(model_id: str, request_body: dict) -> str:
    """
    Amazon Nova Invoke API format:
    - request: { schemaVersion: "messages-v1", system: [...], messages: [...], inferenceConfig: {...} }
    - response text: response_json["output"]["message"]["content"][0]["text"]
    """
    try:
        logger.info(f"Invoking Bedrock model: {model_id}")
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
        raise HTTPException(status_code=500, detail="AWS Bedrock connection or credentials error.")
    except ClientError as e:
        error_msg = e.response.get('Error', {}).get('Message', str(e))
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
    # Put index.html in the same folder as server.py
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"ok": True, "region": AWS_REGION, "text_model": MODEL_ID_TEXT, "vision_model": MODEL_ID_VISION}

@app.post("/api/triage", response_model=TextResponse)
def triage(req: TriageRequest):
    system_list = [
        {
            "text": (
                "You are PulseNova, a medical triage assistant.\n"
                "GOAL: Help users understand symptoms and decide next steps (Emergency, Urgent Care, Home care).\n"
                "RULES:\n"
                "1) NEVER diagnose. Use language like 'This could be consistent with...'\n"
                "2) If emergency warning signs (e.g., chest pain, severe breathing trouble, stroke signs, fainting, severe bleeding), tell them to call 911 / local emergency immediately.\n"
                "3) Ask ONE clarifying question at a time when needed.\n"
                "4) Be concise and practical.\n"
                "5) End with a short safety note: 'Not a diagnosis.'\n"
            )
        }
    ]

    # Convert history to Nova messages
    messages = []
    for turn in req.history[-20:]:  # keep last 20 turns
        messages.append({"role": turn.role, "content": [{"text": turn.text}]})

    # Current user message
    user_content = [{"text": req.message}]
    
    # Model Selection Logic
    # Dynamically select the vision model if an image is provided, otherwise stick to text model
    active_model_id = MODEL_ID_TEXT

    if req.image_base64:
        try:
            raw = _b64_to_bytes(req.image_base64)
            fmt = _guess_image_format(raw)
            # Switch to Vision model for processing
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
            # Catch the strictly validated image format exception
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
                {
                    "image": {
                        "format": fmt,
                        "source": {"bytes": _strip_data_url(req.image_base64)},
                    }
                },
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
# Run:  python -m uvicorn server:app --reload --port 8000
#
# Env examples (PowerShell):
#   $env:AWS_REGION="us-east-1"
#   $env:MODEL_ID="us.amazon.nova-micro-v1:0"
#   $env:VISION_MODEL_ID="us.amazon.nova-lite-v1:0"
#
# Make sure your AWS credentials are available (AWS_PROFILE or env keys) and
# the Bedrock models are enabled in your AWS account/region.
