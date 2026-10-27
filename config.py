import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")



class Config:
    BASE_DIR = BASE_DIR
    SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)
    DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_USER = os.getenv("DB_USER", "cropsense_app")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_NAME = os.getenv("DB_NAME", "cropsense_ai")

    DATABASE_CONFIG = {
        "host": DB_HOST,
        "port": DB_PORT,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "database": DB_NAME,
    }


    MODEL_PATH = BASE_DIR / os.getenv(
        "MODEL_PATH", "model/crop_disease_model.keras"
    )

    CLASS_NAMES_PATH = BASE_DIR / os.getenv(
        "CLASS_NAMES_PATH", "model/class_names.json"
    )

    UPLOAD_FOLDER = BASE_DIR / os.getenv(
        "UPLOAD_FOLDER", "instance/uploads"
    )

    HEATMAP_FOLDER = BASE_DIR / os.getenv(
        "HEATMAP_FOLDER", "instance/heatmaps"
    )