import os
import uuid
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, String, Integer, Boolean, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

# Get the database URL from Railway
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    # SQLAlchemy 1.4+ requires "postgresql://" instead of "postgres://"
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Set up the SQLAlchemy engine
# (If testing locally without a DB, this defaults to a local SQLite file)
engine = create_engine(DATABASE_URL or "sqlite:///./pulsenova_local.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==========================================
# 1. User Profile Table
# ==========================================
class User(Base):
    __tablename__ = "users"
    
    sub = Column(String, primary_key=True, index=True) # AWS Cognito User ID
    email = Column(String, index=True, nullable=True)
    
    # Preferences
    consent_store_history = Column(Boolean, default=False)
    data_retention_days = Column(Integer, default=30)
    
    # Medical Profile (from your new accounts.html)
    age = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    height = Column(String, nullable=True)
    weight = Column(String, nullable=True)
    allergies = Column(Text, nullable=True)
    medical_history = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# ==========================================
# 2. Triage Chat History Table
# ==========================================
class TriageSession(Base):
    __tablename__ = "triage_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_sub = Column(String, ForeignKey("users.sub"), nullable=False)
    title = Column(String, default="New Triage Chat") # e.g., "Stomach pain & fever"
    
    # Store the entire conversation array as JSON
    messages = Column(JSON, default=list) 
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# ==========================================
# 3. Vitals History Table
# ==========================================
class VitalReading(Base):
    __tablename__ = "vitals_history"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_sub = Column(String, ForeignKey("users.sub"), nullable=False)
    
    bpm = Column(Integer, nullable=False)
    context = Column(String, nullable=False) # 'resting', 'after_exercise', etc.
    signal_quality = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# ==========================================
# 4. Scans & Labs History Table
# ==========================================
class MedicalDocument(Base):
    __tablename__ = "medical_documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_sub = Column(String, ForeignKey("users.sub"), nullable=False)
    
    doc_type = Column(String, nullable=False) # 'xray' or 'lab'
    image_b64 = Column(Text, nullable=False)  # The base64 image data
    report_html = Column(Text, nullable=False) # The AI-generated markdown/HTML report
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# ==========================================
# Initialize Database
# ==========================================
def init_db():
    # This creates all the tables if they don't already exist
    Base.metadata.create_all(bind=engine)

# Dependency to get DB session in FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()