import unittest
from pathlib import Path
from unittest.mock import patch

from database import Database, initialize_database


class DatabaseTests(unittest.TestCase):
    def test_fetch_one_uses_configured_connection_and_closes_resources(self):
        config = {
            "host": "db.test.local", "port": 3307, "user": "test-user",
            "password": "test-password", "database": "test-cropsense",
        }
        cursor = unittest.mock.MagicMock()
        cursor.fetchone.return_value = {"id": 4}
        connection = unittest.mock.MagicMock()
        connection.cursor.return_value = cursor
        query = "SELECT id FROM users WHERE email = %s LIMIT 1"

        with patch("database.mysql.connector.connect", return_value=connection) as connect:
            result = Database(config).fetch_one(query, ("grower@example.com",))

        self.assertEqual(result, {"id": 4})
        connect.assert_called_once_with(**config)
        connection.cursor.assert_called_once_with(dictionary=True)
        cursor.execute.assert_called_once_with(query, ("grower@example.com",))
        cursor.close.assert_called_once()
        connection.close.assert_called_once()

    def test_execute_commits_and_closes_resources(self):
        cursor = unittest.mock.MagicMock()
        cursor.lastrowid = 12
        connection = unittest.mock.MagicMock()
        connection.cursor.return_value = cursor

        with patch("database.mysql.connector.connect", return_value=connection):
            inserted_id = Database({"database": "test-cropsense"}).execute(
                "INSERT INTO users (username) VALUES (%s)", ("grower",)
            )

        self.assertEqual(inserted_id, 12)
        connection.commit.assert_called_once()
        cursor.close.assert_called_once()
        connection.close.assert_called_once()

    def test_database_name_is_validated_before_connecting(self):
        with patch("database.mysql.connector.connect") as connect:
            with self.assertRaisesRegex(ValueError, "DB_NAME"):
                initialize_database({"database": "bad-name; DROP DATABASE x"}, Path("schema.sql"))
            connect.assert_not_called()

    def test_connection_errors_are_propagated_for_clear_cli_reporting(self):
        with patch("database.mysql.connector.connect", side_effect=RuntimeError("mysql unavailable")):
            with self.assertRaisesRegex(RuntimeError, "mysql unavailable"):
                Database({"host": "127.0.0.1"}).fetch_one("SELECT 1")


if __name__ == "__main__":
    unittest.main()