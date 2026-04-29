import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Use POSTGRES_URI if set; otherwise default to a writable runtime data directory
# outside the repository tree so we never store the DB file inside the source tree.
DATABASE_URI = os.getenv("POSTGRES_URI")
if not DATABASE_URI:
    data_dir = os.getenv("LEXAI_DATA_DIR", "/app/data")
    # Require an absolute path to prevent path traversal via the env variable.
    if not os.path.isabs(data_dir):
        raise ValueError(f"LEXAI_DATA_DIR must be an absolute path, got: {data_dir!r}")
    data_dir = os.path.realpath(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    sqlite_db_path = os.path.join(data_dir, "lexai_memory.db")
    DATABASE_URI = f"sqlite:///{sqlite_db_path}"

engine = create_engine(DATABASE_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    """
    Enterprise Multi-Tenant User Model
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    firm_id = Column(String(50), nullable=True) # For Multi-Tenancy

    sessions = relationship("ChatSession", back_populates="user")

class ChatSession(Base):
    """
    Persistent Case Memory for a specific User
    """
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    case_name = Column(String(100), default="New Legal Research")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Store the entire LangGraph thought stream and messages as JSON
    messages_json = Column(JSON, default=list)

    user = relationship("User", back_populates="sessions")

def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database Initialized.")

if __name__ == "__main__":
    init_db()
