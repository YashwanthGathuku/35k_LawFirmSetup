"""
Tegifa Legal — Database Models
Multi-tenant user and chat session persistence.
"""
import os
import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    JSON,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Default to SQLite for development; override with DATABASE_URI env var for production.
DATABASE_URI = os.getenv("DATABASE_URI", "sqlite:///tegifa_memory.db")

engine = create_engine(DATABASE_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    """Multi-tenant user model."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    firm_id = Column(String(50), nullable=True)

    sessions = relationship("ChatSession", back_populates="user")
    matters = relationship("Matter", back_populates="user")

class Matter(Base):
    """A legal case or deal workspace."""
    __tablename__ = "matters"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String(150), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="matters")
    sessions = relationship("ChatSession", back_populates="matter")


class ChatSession(Base):
    """Persistent case memory for a specific user."""

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    matter_id = Column(Integer, ForeignKey("matters.id"), nullable=True)
    case_name = Column(String(100), default="New Legal Research")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    messages_json = Column(JSON, default=list)

    user = relationship("User", back_populates="sessions")
    matter = relationship("Matter", back_populates="sessions")


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


# Auto-initialize on import (idempotent via create_all)
init_db()
