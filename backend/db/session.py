from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.db.config import Settings
import os

SQLALCHEMY_DATABASE_URL = Settings.DATABASE_URL  # 서버에서 직접 URL 생성
engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)