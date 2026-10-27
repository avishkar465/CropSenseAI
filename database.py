import re
from pathlib import Path
from typing import Any

import mysql.connector
from mysql.connector import Error


class Database:
    """Small MySQL helper that creates short-lived, parameterized queries."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def connect(self):
        return mysql.connector.connect(**self.config)

    def fetch_one(self, query: str, params: tuple = ()):
        connection = self.connect()
        cursor = connection.cursor(dictionary=True)
        try:
            cursor.execute(query, params)
            return cursor.fetchone()
        finally:
            cursor.close()
            connection.close()

    def fetch_all(self, query: str, params: tuple = ()):
        connection = self.connect()
        cursor = connection.cursor(dictionary=True)
        try:
            cursor.execute(query, params)
            return cursor.fetchall()
        finally:
            cursor.close()
            connection.close()

    def execute(self, query: str, params: tuple = ()) -> int:
        connection = self.connect()
        cursor = connection.cursor()
        try:
            cursor.execute(query, params)
            connection.commit()
            return cursor.lastrowid
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()


def initialize_database(config: dict[str, Any], schema_path: Path) -> None:
    database_name = config["database"]
    if not re.fullmatch(r"[A-Za-z0-9_]+", database_name):
        raise ValueError("DB_NAME may contain only letters, numbers, and underscores.")

    server_config = {key: value for key, value in config.items() if key != "database"}
    connection = mysql.connector.connect(**server_config)
    cursor = connection.cursor()
    try:
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        connection.commit()
    finally:
        cursor.close()
        connection.close()

    db = Database(config)
    statements = [part.strip() for part in schema_path.read_text(encoding="utf-8").split(";") if part.strip()]
    connection = db.connect()
    cursor = connection.cursor()
    try:
        for statement in statements:
            cursor.execute(statement)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


__all__ = ["Database", "Error", "initialize_database"]