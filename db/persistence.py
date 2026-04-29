"""Tegifa Legal — Chat Persistence Layer."""
import logging
import os
from contextlib import contextmanager
from .models import SessionLocal, User, ChatSession

logger = logging.getLogger("tegifa.persistence")


def _get_non_negative_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        logger.warning("Invalid value for %s; using default %d.", name, default)
        return default


def _get_retention_limits() -> tuple[int, int]:
    max_turns = _get_non_negative_env_int("CHAT_MAX_TURNS", 20)
    max_chars = _get_non_negative_env_int("CHAT_MAX_CHARS", 20000)
    return max_turns, max_chars


def _apply_retention(messages: list) -> tuple[list, str]:
    max_turns, max_chars = _get_retention_limits()
    ordered = list(messages)
    kept = ordered[-max_turns:] if max_turns > 0 else ordered
    total_chars = sum(len(str(m.get("content", ""))) for m in kept)
    while kept and total_chars > max_chars:
        total_chars -= len(str(kept[0].get("content", "")))
        kept = kept[1:]
    summary = ""
    if len(ordered) > len(kept):
        dropped = len(ordered) - len(kept)
        summary = f"Trimmed {dropped} older message(s) due to retention policy."
    return kept, summary


@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_chat_session(username: str, case_name: str, messages: list) -> None:
    with get_db() as db:
        try:
            user = db.query(User).filter(User.username == username).first()
            if not user:
                user = User(username=username, email=f"{username}@tegifa.local", hashed_password="***")
                db.add(user)
                db.flush()

            bounded_messages, summary = _apply_retention(messages)
            chat_session = db.query(ChatSession).filter(
                ChatSession.user_id == user.id,
                ChatSession.case_name == case_name,
            ).first()
            if not chat_session:
                chat_session = ChatSession(
                    user_id=user.id, case_name=case_name, messages_json=bounded_messages, summary_text=summary
                )
                db.add(chat_session)
            else:
                chat_session.messages_json = bounded_messages
                chat_session.summary_text = summary
            db.commit()
        except Exception:
            db.rollback()
            raise


def load_chat_session(username: str, case_name: str) -> list:
    with get_db() as db:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return []
        chat_session = db.query(ChatSession).filter(
            ChatSession.user_id == user.id,
            ChatSession.case_name == case_name,
        ).first()
        return chat_session.messages_json if chat_session and chat_session.messages_json else []


def list_case_sessions(username: str) -> list[str]:
    with get_db() as db:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return []
        rows = db.query(ChatSession.case_name).filter(ChatSession.user_id == user.id).order_by(ChatSession.updated_at.desc()).all()
        return [r[0] for r in rows]
