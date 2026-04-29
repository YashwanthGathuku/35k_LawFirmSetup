import pytest, os, sys
from unittest.mock import patch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["DATABASE_URI"] = "sqlite:///:memory:"
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base, User, ChatSession
import db.models, db.persistence

engine = create_engine("sqlite:///:memory:")
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db.models.engine = engine; db.models.SessionLocal = TestingSessionLocal; db.persistence.SessionLocal = TestingSessionLocal

@pytest.fixture(autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_multiple_case_sessions_per_user_create_read_update():
    db.persistence.save_chat_session("u1", "CaseA", [{"role":"user","content":"a"}])
    db.persistence.save_chat_session("u1", "CaseB", [{"role":"user","content":"b"}])
    db.persistence.save_chat_session("u1", "CaseA", [{"role":"assistant","content":"updated"}])
    assert db.persistence.load_chat_session("u1", "CaseA")[0]["content"] == "updated"
    assert db.persistence.load_chat_session("u1", "CaseB")[0]["content"] == "b"


def test_list_case_sessions_ordering():
    db.persistence.save_chat_session("u2", "Case1", [{"role": "user", "content": "1"}])
    db.persistence.save_chat_session("u2", "Case2", [{"role": "user", "content": "2"}])
    # Re-save Case1 so it has the most recent updated_at
    db.persistence.save_chat_session("u2", "Case1", [{"role": "assistant", "content": "updated"}])
    names = db.persistence.list_case_sessions("u2")
    assert names == ["Case1", "Case2"]


def test_retention_truncation(monkeypatch):
    monkeypatch.setenv("CHAT_MAX_TURNS", "2")
    monkeypatch.setenv("CHAT_MAX_CHARS", "5")
    msgs = [{"role":"user","content":"1111"},{"role":"assistant","content":"2222"},{"role":"user","content":"3333"}]
    db.persistence.save_chat_session("u3", "CaseR", msgs)
    loaded = db.persistence.load_chat_session("u3", "CaseR")
    assert loaded == [{"role":"user","content":"3333"}]


def test_save_chat_session_commit_failure_rolls_back():
    """Verify that a commit failure triggers rollback and nothing is persisted."""
    session = TestingSessionLocal()
    with patch.object(session, "commit", side_effect=RuntimeError("DB error")):
        with patch("db.persistence.SessionLocal", return_value=session):
            with pytest.raises(RuntimeError, match="DB error"):
                db.persistence.save_chat_session("u4", "CaseF", [{"role": "user", "content": "fail"}])
    # Nothing should have been persisted
    assert db.persistence.load_chat_session("u4", "CaseF") == []
