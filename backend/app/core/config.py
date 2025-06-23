# backend/app/core/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "data/vector_db")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

settings = Settings()
