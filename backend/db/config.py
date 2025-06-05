import os
from dotenv import load_dotenv

_ = load_dotenv()

class Settings:
    DB_USERNAME : str = os.getenv("MARIADB_USER")
    DB_PASSWORD = os.getenv("MARIADB_PASSWORD")
    DB_HOST : str = os.getenv("MARIADB_HOST", "localhost")
    DB_PORT : str = os.getenv("MARIADB_PORT", 3306)
    DB_DATABASE : str = os.getenv("MARIADB_DATABASE")
    DATABASE_URL = f"mariadb+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"