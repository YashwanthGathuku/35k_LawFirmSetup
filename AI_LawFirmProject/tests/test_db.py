import pytest
import os
import sys

# Ensure we can import from the main app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Mock the environment variable before importing models
os.environ["POSTGRES_URI"] = "sqlite:///:memory:"

from db.models import Base, User, ChatSession, init_db
import db.models
import db.persistence

# Override the engine and SessionLocal in the modules to use in-memory SQLite for testing
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

def test_init_db(capsys):
    """Test that init_db executes without error."""
    init_db()
    captured = capsys.readouterr()
    assert "Database Initialized." in captured.out

def test_save_new_user_and_session():
    messages = [{"role": "user", "content": "Hello"}]
    db.persistence.save_chat_session("testuser", messages)

    # Verify user was created
    db_session = TestingSessionLocal()
    user = db_session.query(User).filter_by(username="testuser").first()
    assert user is not None
    assert user.email == "testuser@lexai.com"

    # Verify chat session was created
    chat_session = db_session.query(ChatSession).filter_by(user_id=user.id).first()
    assert chat_session is not None
    assert chat_session.messages_json == messages
    db_session.close()

def test_update_existing_session():
    # Setup initial
    messages = [{"role": "user", "content": "Hello"}]
    db.persistence.save_chat_session("testuser", messages)

    # Update
    new_messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    db.persistence.save_chat_session("testuser", new_messages)

    db_session = TestingSessionLocal()
    user = db_session.query(User).filter_by(username="testuser").first()
    # There should still only be 1 chat session for this user
    sessions = db_session.query(ChatSession).filter_by(user_id=user.id).all()
    assert len(sessions) == 1
    assert sessions[0].messages_json == new_messages
    db_session.close()

def test_load_existing_session():
    messages = [{"role": "user", "content": "Hello"}]
    db.persistence.save_chat_session("loaduser", messages)

    loaded_messages = db.persistence.load_chat_session("loaduser")
    assert loaded_messages == messages

def test_load_nonexistent_user():
    loaded_messages = db.persistence.load_chat_session("ghostuser")
    assert loaded_messages == []

def test_load_user_without_session():
    db_session = TestingSessionLocal()
    user = User(username="nouser", email="nouser@lexai.com", hashed_password="***")
    db_session.add(user)
    db_session.commit()
    db_session.close()

    loaded_messages = db.persistence.load_chat_session("nouser")
    assert loaded_messages == []
