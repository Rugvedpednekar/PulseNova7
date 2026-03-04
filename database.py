import os
import uuid
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, String, Integer, Boolean, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

# Get the database URL from Railway
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL or "sqlite:///./pulsenova_local.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ==========================================
# 1. User Profile Table
# ==========================================
class User(Base):
    __tablename__ = "users"

    sub                   = Column(String,  primary_key=True, index=True)
    email                 = Column(String,  index=True, nullable=True)
    consent_store_history = Column(Boolean, default=False)
    data_retention_days   = Column(Integer, default=30)
    age                   = Column(String,  nullable=True)
    gender                = Column(String,  nullable=True)
    height                = Column(String,  nullable=True)
    weight                = Column(String,  nullable=True)
    allergies             = Column(Text,    nullable=True)
    medical_history       = Column(Text,    nullable=True)
    created_at            = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==========================================
# 2. Triage Chat History Table
# ==========================================
class TriageSession(Base):
    __tablename__ = "triage_sessions"

    id       = Column(String,   primary_key=True, default=lambda: str(uuid.uuid4()))
    user_sub = Column(String,   ForeignKey("users.sub"), nullable=False)
    title    = Column(String,   default="New Triage Chat")
    messages = Column(JSON,     default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==========================================
# 3. Vitals History Table
# ==========================================
class VitalReading(Base):
    __tablename__ = "vitals_history"

    id             = Column(String,  primary_key=True, default=lambda: str(uuid.uuid4()))
    user_sub       = Column(String,  ForeignKey("users.sub"), nullable=False)
    bpm            = Column(Integer, nullable=False)
    context        = Column(String,  nullable=False)
    signal_quality = Column(Integer, nullable=True)
    created_at     = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==========================================
# 4. Scans & Labs History Table
# ==========================================
class MedicalDocument(Base):
    __tablename__ = "medical_documents"

    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_sub    = Column(String, ForeignKey("users.sub"), nullable=False)
    doc_type    = Column(String, nullable=False)   # 'xray' or 'lab'
    image_b64   = Column(Text,   nullable=False)
    report_html = Column(Text,   nullable=False)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==========================================
# 5. Prescriptions Table  ← NEW
# ==========================================
class Prescription(Base):
    __tablename__ = "prescriptions"

    id       = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_sub = Column(String, ForeignKey("users.sub"), nullable=False, index=True)

    # Core medication fields
    name     = Column(String, nullable=False)          # e.g. "Metformin"
    dose     = Column(String, nullable=True, default="")  # e.g. "500 mg"
    times    = Column(JSON,   nullable=True, default=list) # e.g. ["08:00", "20:00"]
    repeat   = Column(String, nullable=True, default="DAILY")  # DAILY | WEEKDAYS | CUSTOM_DAYS
    days     = Column(JSON,   nullable=True, default=list)     # ["MO","WE","FR"] when CUSTOM_DAYS
    notes    = Column(Text,   nullable=True, default="")       # "Take with food"

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==========================================
# Initialize Database
# ==========================================
def init_db():
    # Creates all tables that don't already exist — safe to call on every startup
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()