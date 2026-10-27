import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import create_app
from predictor import ModelUnavailable
from tests.helpers import FakeDatabase, FakePredictor


def csrf_value(response):
    match = re.search(rb'name="csrf_token" value="([^"]+)"', response.data)
    if not match:
        raise AssertionError("CSRF token not found in form")
    return match.group(1).decode()


class CropSenseAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.database = FakeDatabase()
        self.predictor = FakePredictor()
        self.app = create_app({
            "TESTING": True,
            "SECRET_KEY": "test-only-secret",
            "UPLOAD_FOLDER": root / "uploads",
            "HEATMAP_FOLDER": root / "heatmaps",
        }, self.database, self.predictor)
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def login(self):
        with self.client.session_transaction() as current_session:
            current_session["user_id"] = 1
            current_session["username"] = "field-user"

    def test_public_pages_start_without_database(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get("/register").status_code, 200)
        self.assertEqual(self.client.get("/login").status_code, 200)

    def test_session_secret_fallback_allows_auth_pages_without_test_override(self):
        app = create_app({"TESTING": True}, database=self.database)

        self.assertTrue(app.secret_key)
        self.assertEqual(app.test_client().get("/login").status_code, 200)

    def test_protected_routes_redirect_to_login(self):
        self.assertEqual(self.client.get("/dashboard").status_code, 302)
        self.assertEqual(self.client.get("/history").status_code, 302)
        self.assertEqual(self.client.get("/profile").status_code, 302)

    def test_authenticated_pages_render(self):
        self.login()
        for path in ("/dashboard", "/predict", "/history", "/disease-guide", "/profile"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_state_changing_request_requires_csrf_token(self):
        response = self.client.post("/login", data={"email": "field@example.com", "password": "x"})
        self.assertEqual(response.status_code, 400)

    def test_registration_validates_password_and_hashes_valid_password(self):
        form = self.client.get("/register")
        token = csrf_value(form)
        response = self.client.post("/register", data={
            "csrf_token": token, "username": "new-grower", "email": "new@example.com", "password": "short",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"at least 8 characters", response.data)

        form = self.client.get("/register")
        response = self.client.post("/register", data={
            "csrf_token": csrf_value(form), "username": "new-grower", "email": "new@example.com",
            "password": "a-longer-password",
        })
        self.assertEqual(response.status_code, 302)
        user = self.database.users[2]
        self.assertNotEqual(user["password_hash"], "a-longer-password")
        self.assertTrue(user["password_hash"].startswith("scrypt:"))

    def test_login_checks_password_and_creates_session(self):
        form = self.client.get("/login")
        response = self.client.post("/login", data={
            "csrf_token": csrf_value(form), "email": "field@example.com", "password": "wrong-password",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"was not recognized", response.data)

        form = self.client.get("/login")
        response = self.client.post("/login", data={
            "csrf_token": csrf_value(form), "email": "field@example.com", "password": "a-secure-password",
        })
        self.assertEqual(response.status_code, 302)
        with self.client.session_transaction() as current_session:
            self.assertEqual(current_session["user_id"], 1)

    def test_registration_and_login_use_configured_database_helper(self):
        database = FakeDatabase()
        database_config = {
            "host": "db.test.local", "port": 3307, "user": "test-user",
            "password": "test-password", "database": "test-cropsense",
        }
        app = create_app({
            "TESTING": True, "SECRET_KEY": "test-only-secret", "DATABASE_CONFIG": database_config,
        }, predictor=self.predictor)
        client = app.test_client()

        with patch("app.Database", return_value=database) as database_factory:
            register_form = client.get("/register")
            registration = client.post("/register", data={
                "csrf_token": csrf_value(register_form), "username": "db-grower",
                "email": "db-grower@example.com", "password": "a-long-password",
            })
            login_form = client.get("/login")
            login = client.post("/login", data={
                "csrf_token": csrf_value(login_form), "email": "db-grower@example.com",
                "password": "a-long-password",
            })

        self.assertEqual(registration.status_code, 302)
        self.assertEqual(login.status_code, 302)
        database_factory.assert_called_once_with(database_config)
        self.assertEqual([call[0] for call in database.calls], ["fetch_one", "execute", "fetch_one"])
        self.assertIn("email = %s OR username = %s", database.calls[0][1])
        self.assertEqual(database.calls[0][2], ("db-grower@example.com", "db-grower"))
        self.assertIn("VALUES (%s, %s, %s)", database.calls[1][1])
        self.assertNotEqual(database.calls[1][2][2], "a-long-password")
        self.assertIn("WHERE email = %s", database.calls[2][1])
        self.assertEqual(database.calls[2][2], ("db-grower@example.com",))

    def test_registration_database_error_is_logged_but_not_exposed(self):
        class FailingDatabase:
            def fetch_one(self, _query, _params):
                raise RuntimeError("safe-test-db-failure")

        app = create_app({"TESTING": True, "SECRET_KEY": "test-only-secret"}, FailingDatabase(), self.predictor)
        client = app.test_client()
        form = client.get("/register")
        with self.assertLogs("app", level="ERROR") as captured:
            response = client.post("/register", data={
                "csrf_token": csrf_value(form), "username": "db-grower",
                "email": "db-grower@example.com", "password": "a-long-password",
            })

        self.assertEqual(response.status_code, 200)
        self.assertIn("safe-test-db-failure", "\n".join(captured.output))
        self.assertIn(b"Database operation failed", response.data)
        self.assertNotIn(b"safe-test-db-failure", response.data)

    def test_upload_rejects_invalid_image_before_model_call(self):
        self.login()
        form = self.client.get("/predict")
        response = self.client.post("/predict", data={
            "csrf_token": csrf_value(form), "image": (io.BytesIO(b"not an image"), "leaf.jpg"),
        }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.database.predictions, [])

    def test_prediction_is_saved_with_user_and_top_three(self):
        self.login()
        form = self.client.get("/predict")
        with patch("app.generate_gradcam", return_value=__import__("numpy").zeros((7, 7))), patch("app.create_overlay"):
            response = self.client.post("/predict", data={
                "csrf_token": csrf_value(form), "image": (io.BytesIO(_png_image()), "leaf.png"),
            }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(len(self.database.predictions), 1)
        row = self.database.predictions[0]
        self.assertEqual(row["user_id"], 1)
        self.assertEqual(row["crop_name"], "Tomato")
        self.assertEqual(len(__import__("json").loads(row["top_predictions"])), 3)

    def test_missing_model_does_not_create_a_prediction(self):
        class UnavailablePredictor:
            def predict(self, _image_path):
                raise ModelUnavailable("Trained model missing.")

        self.login()
        self.app.extensions["cropsense_predictor"] = UnavailablePredictor()
        form = self.client.get("/predict")
        response = self.client.post("/predict", data={
            "csrf_token": csrf_value(form), "image": (io.BytesIO(_png_image()), "leaf.png"),
        }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.database.predictions, [])

    def test_history_and_detail_are_scoped_to_current_user(self):
        self.login()
        self.database.predictions.extend([
            {"id": 10, "user_id": 1, "image_path": "uploads/a.jpg", "heatmap_path": None,
             "crop_name": "Tomato", "disease_name": "Early blight", "confidence": 0.8,
             "top_predictions": "[]", "created_at": "2026-01-01 10:00:00"},
            {"id": 11, "user_id": 2, "image_path": "uploads/b.jpg", "heatmap_path": None,
             "crop_name": "Potato", "disease_name": "Late blight", "confidence": 0.7,
             "top_predictions": "[]", "created_at": "2026-01-01 10:00:00"},
        ])
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Tomato", response.data)
        self.assertNotIn(b"Potato", response.data)
        self.assertEqual(self.client.get("/prediction/11").status_code, 404)


def _png_image():
    from PIL import Image
    image = Image.new("RGB", (32, 32), (50, 130, 60))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


if __name__ == "__main__":
    unittest.main()