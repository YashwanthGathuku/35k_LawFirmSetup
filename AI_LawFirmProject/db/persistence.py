from sqlalchemy.orm import Session
from .models import SessionLocal, User, ChatSession

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_chat_session(username: str, messages: list):
    """
    Saves the Streamlit session_state.messages array directly into the database.
    """
    db = SessionLocal()

    # 1. Ensure User exists
    user = db.query(User).filter(User.username == username).first()
    if not user:
        # Create a stub user if it doesn't exist (in real app, register handles this)
        user = User(username=username, email=f"{username}@lexai.com", hashed_password="***")
        db.add(user)
        db.commit()
        db.refresh(user)

    # 2. Get or Create active session (For simplicity, just updating the first session)
    chat_session = db.query(ChatSession).filter(ChatSession.user_id == user.id).first()
    if not chat_session:
        chat_session = ChatSession(user_id=user.id, messages_json=messages)
        db.add(chat_session)
    else:
        chat_session.messages_json = messages

    db.commit()
    db.close()

def load_chat_session(username: str):
    """
    Loads the persistent chat history from the database.
    """
    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    if not user:
        db.close()
        return []

    chat_session = db.query(ChatSession).filter(ChatSession.user_id == user.id).first()
    db.close()

    if chat_session and chat_session.messages_json:
        return chat_session.messages_json
    return []
