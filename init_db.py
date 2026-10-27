import logging

from config import Config
from database import initialize_database


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


if __name__ == "__main__":
    try:
        initialize_database(Config.DATABASE_CONFIG, Config.BASE_DIR / "schema.sql")
    except Exception as exc:
        logging.error("Database initialization failed: %s", exc)
        raise SystemExit(1) from None
    logging.info("Database '%s' and tables are ready.", Config.DATABASE_CONFIG["database"])