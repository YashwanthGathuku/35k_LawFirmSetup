"""
Tegifa Legal — Chat Persistence Layer
Handles saving and loading chat sessions with proper session lifecycle.
"""
import logging
from contextlib import contextmanager
from sqlalchemy.orm import Session
from .models import SessionLocal, User, ChatSession, Matter

logger = logging.getLogger("tegifa.persistence")


@contextmanager
def get_db():
    """Context manager for database sessions with guaranteed cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_or_create_user(db, username: str) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        user = User(
            username=username,
            email=f"{username}@tegifa.local",
            hashed_password="***",
        )
        db.add(user)
        db.flush()
    return user


def get_matters(username: str) -> list:
    """Get all matters for a user."""
    with get_db() as db:
        user = get_or_create_user(db, username)
        matters = db.query(Matter).filter(Matter.user_id == user.id).all()
        return [{"id": m.id, "name": m.name} for m in matters]


def create_matter(username: str, name: str) -> dict:
    """Create a new matter workspace."""
    with get_db() as db:
        user = get_or_create_user(db, username)
        matter = Matter(user_id=user.id, name=name)
        db.add(matter)
        db.commit()
        db.refresh(matter)
        return {"id": matter.id, "name": matter.name}


def save_chat_session(username: str, matter_id: int, messages: list) -> None:
    """Persist chat history to the database for a specific matter."""
    with get_db() as db:
        try:
            user = get_or_create_user(db, username)
            chat_session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.matter_id == matter_id)
                .first()
            )
            if not chat_session:
                chat_session = ChatSession(user_id=user.id, matter_id=matter_id, messages_json=messages)
                db.add(chat_session)
            else:
                chat_session.messages_json = messages
            db.commit()
        except Exception:
            db.rollback()
            raise


def load_chat_session(username: str, matter_id: int) -> list:
    """Load persistent chat history from the database for a specific matter."""
    with get_db() as db:
        user = get_or_create_user(db, username)
        chat_session = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user.id, ChatSession.matter_id == matter_id)
            .first()
        )
        if chat_session and chat_session.messages_json:
            return chat_session.messages_json
        return []
