"""
Tegifa Legal — Database Tests
Tests for persistence layer with in-memory SQLite.
"""
import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Override env BEFORE importing models
os.environ["DATABASE_URI"] = "sqlite:///:memory:"

from db.models import Base, User, ChatSession
import db.models
import db.persistence

# Override engine and session factory for testing
engine = create_engine("sqlite:///:memory:")
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db.models.engine = engine
db.models.SessionLocal = TestingSessionLocal
db.persistence.SessionLocal = TestingSessionLocal


@pytest.fixture(autouse=True)
def setup_database():
    """Create tables before each test and drop them after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_save_new_user_and_session():
    messages = [{"role": "user", "content": "Hello"}]
    db.persistence.save_chat_session("testuser", messages)

    session = TestingSessionLocal()
    user = session.query(User).filter_by(username="testuser").first()
    assert user is not None
    assert user.email == "testuser@tegifa.local"

    chat_session = session.query(ChatSession).filter_by(user_id=user.id).first()
    assert chat_session is not None
    assert chat_session.messages_json == messages
    session.close()


def test_update_existing_session():
    messages = [{"role": "user", "content": "Hello"}]
    db.persistence.save_chat_session("testuser", messages)

    new_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    db.persistence.save_chat_session("testuser", new_messages)

    session = TestingSessionLocal()
    user = session.query(User).filter_by(username="testuser").first()
    sessions = session.query(ChatSession).filter_by(user_id=user.id).all()
    assert len(sessions) == 1
    assert sessions[0].messages_json == new_messages
    session.close()


def test_load_existing_session():
    messages = [{"role": "user", "content": "Hello"}]
    db.persistence.save_chat_session("loaduser", messages)

    loaded = db.persistence.load_chat_session("loaduser")
    assert loaded == messages


def test_load_nonexistent_user():
    loaded = db.persistence.load_chat_session("ghostuser")
    assert loaded == []


def test_load_user_without_session():
    session = TestingSessionLocal()
    user = User(
        username="nouser", email="nouser@tegifa.local", hashed_password="***"
    )
    session.add(user)
    session.commit()
    session.close()

    loaded = db.persistence.load_chat_session("nouser")
    assert loaded == []


def test_save_rollback_on_error(mocker):
    """Verify that a failed commit triggers rollback, not silent corruption."""
    mocker.patch.object(
        TestingSessionLocal, "commit", side_effect=Exception("DB Error")
    )
    with pytest.raises(Exception, match="DB Error"):
        db.persistence.save_chat_session("erroruser", [])
