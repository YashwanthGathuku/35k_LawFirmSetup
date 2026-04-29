"""
Tegifa Legal — Chat Persistence Layer
Handles saving and loading chat sessions with proper session lifecycle.
"""
import logging
from contextlib import contextmanager
from sqlalchemy.orm import Session
from .models import SessionLocal, User, ChatSession

logger = logging.getLogger("tegifa.persistence")


@contextmanager
def get_db():
    """Context manager for database sessions with guaranteed cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_chat_session(username: str, messages: list) -> None:
    """
    Persist the Streamlit chat history to the database.

    Creates a stub user if one doesn't exist. Updates the active chat session
    or creates a new one.
    """
    with get_db() as db:
        try:
            user = db.query(User).filter(User.username == username).first()
            if not user:
                user = User(
                    username=username,
                    email=f"{username}@tegifa.local",
                    hashed_password="***",
                )
                db.add(user)
                db.flush()  # Get user.id without committing yet

            chat_session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id)
                .first()
            )
            if not chat_session:
                chat_session = ChatSession(user_id=user.id, messages_json=messages)
                db.add(chat_session)
            else:
                chat_session.messages_json = messages

            db.commit()
        except Exception:
            db.rollback()
            raise


def load_chat_session(username: str) -> list:
    """
    Load persistent chat history from the database.
    Returns an empty list if no history exists.
    """
    with get_db() as db:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return []

        chat_session = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user.id)
            .first()
        )
        if chat_session and chat_session.messages_json:
            return chat_session.messages_json
        return []
