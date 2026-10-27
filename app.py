import csv
import io
import json
import secrets
import logging
from functools import wraps
from pathlib import Path
from uuid import uuid4

from flask import Flask, abort, flash, make_response, redirect, render_template, request, send_from_directory, session, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.security import check_password_hash, generate_password_hash

from config import Config
from database import Database
from disease_info import find_guidance, guide_matches
from gradcam import GradCAMError, create_overlay, generate_gradcam
from predictor import CropDiseasePredictor, InvalidImage, ModelUnavailable


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def create_app(test_config=None, database=None, predictor=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
    app.extensions["cropsense_database"] = database
    app.extensions["cropsense_predictor"] = predictor
    database_config = app.config.get("DATABASE_CONFIG", {})
    logger.info(
        "CropSense configuration initialized (database_password_configured=%s, session_secret_available=%s)",
        bool(database_config.get("password")),
        bool(app.secret_key),
    )

    def get_db():
        if app.extensions["cropsense_database"] is None:
            app.extensions["cropsense_database"] = Database(app.config["DATABASE_CONFIG"])
        return app.extensions["cropsense_database"]

    app.get_db = get_db

    @app.before_request
    def protect_forms():
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            submitted_token = request.form.get("csrf_token", "")
            expected_token = session.get("csrf_token", "")
            if not expected_token or not secrets.compare_digest(submitted_token, expected_token):
                abort(400, description="The form expired. Refresh the page and try again.")

    @app.after_request
    def add_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return response

    @app.context_processor
    def inject_template_helpers():
        def csrf_token():
            if "csrf_token" not in session:
                session["csrf_token"] = secrets.token_urlsafe(32)
            return session["csrf_token"]

        return {"csrf_token": csrf_token}

    def login_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not session.get("user_id"):
                flash("Please sign in to continue.", "info")
                return redirect(url_for("login"))
            return view(*args, **kwargs)

        return wrapped

    app.login_required = login_required

    def get_predictor():
        if app.extensions["cropsense_predictor"] is None:
            app.extensions["cropsense_predictor"] = CropDiseasePredictor(
                app.config["MODEL_PATH"], app.config["CLASS_NAMES_PATH"]
            )
        return app.extensions["cropsense_predictor"]

    @app.get("/")
    def index():
        if session.get("user_id"):
            return redirect(url_for("dashboard"))
        return render_template("index.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            if len(username) < 3 or len(username) > 40:
                flash("Username must be between 3 and 40 characters.", "error")
            elif "@" not in email or len(email) > 254:
                flash("Enter a valid email address.", "error")
            elif len(password) < 8:
                flash("Password must contain at least 8 characters.", "error")
            else:
                try:
                    existing = get_db().fetch_one(
                        "SELECT id FROM users WHERE email = %s OR username = %s LIMIT 1", (email, username)
                    )
                    if existing:
                        flash("That email or username is already registered.", "error")
                    else:
                        user_id = get_db().execute(
                            "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                            (username, email, generate_password_hash(password)),
                        )
                        session.clear()
                        session["user_id"] = user_id
                        session["username"] = username
                        session.permanent = True
                        flash("Your account is ready.", "success")
                        return redirect(url_for("dashboard"))
                except Exception:
                    logger.exception("Registration database operation failed")
                    flash(
                        "Database operation failed. Verify the MySQL settings in .env and restart Flask if they changed.",
                        "error",
                    )
        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            if not email or not password:
                flash("Enter both your email and password.", "error")
            else:
                try:
                    user = get_db().fetch_one(
                        "SELECT id, username, password_hash FROM users WHERE email = %s LIMIT 1", (email,)
                    )
                    if user and check_password_hash(user["password_hash"], password):
                        session.clear()
                        session["user_id"] = user["id"]
                        session["username"] = user["username"]
                        session.permanent = True
                        flash("Welcome back.", "success")
                        return redirect(url_for("dashboard"))
                    flash("Email or password was not recognized.", "error")
                except Exception:
                    logger.exception("Login database operation failed")
                    flash(
                        "Database operation failed. Verify the MySQL settings in .env and restart Flask if they changed.",
                        "error",
                    )
        return render_template("login.html")

    @app.post("/logout")
    @login_required
    def logout():
        session.clear()
        flash("You have been signed out.", "success")
        return redirect(url_for("index"))

    @app.get("/dashboard")
    @login_required
    def dashboard():
        stats = {"total": 0, "healthy": 0, "diseased": 0}
        recent = []
        chart_labels = []
        chart_values = []
        activity_labels = []
        activity_values = []
        database_error = False
        try:
            db = get_db()
            summary = db.fetch_one(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN LOWER(disease_name) LIKE '%healthy%' THEN 1 ELSE 0 END) AS healthy "
                "FROM predictions WHERE user_id = %s",
                (session["user_id"],),
            ) or {}
            stats["total"] = summary.get("total", 0) or 0
            stats["healthy"] = summary.get("healthy", 0) or 0
            stats["diseased"] = stats["total"] - stats["healthy"]
            recent = db.fetch_all(
                "SELECT id, crop_name, disease_name, confidence, created_at FROM predictions "
                "WHERE user_id = %s ORDER BY created_at DESC LIMIT 5",
                (session["user_id"],),
            )
            distribution = db.fetch_all(
                "SELECT disease_name, COUNT(*) AS count FROM predictions WHERE user_id = %s "
                "GROUP BY disease_name ORDER BY count DESC LIMIT 6",
                (session["user_id"],),
            )
            chart_labels = [row["disease_name"] for row in distribution]
            chart_values = [row["count"] for row in distribution]
            activity = db.fetch_all(
                "SELECT DATE_FORMAT(created_at, '%b %Y') AS month, COUNT(*) AS count "
                "FROM predictions WHERE user_id = %s AND created_at >= DATE_SUB(CURDATE(), INTERVAL 5 MONTH) "
                "GROUP BY YEAR(created_at), MONTH(created_at), DATE_FORMAT(created_at, '%b %Y') "
                "ORDER BY YEAR(created_at), MONTH(created_at)",
                (session["user_id"],),
            )
            activity_labels = [row["month"] for row in activity]
            activity_values = [row["count"] for row in activity]
        except Exception:
            logger.exception("Dashboard database operation failed")
            database_error = True
        return render_template(
            "dashboard.html", stats=stats, recent=recent, chart_labels=chart_labels,
            chart_values=chart_values, activity_labels=activity_labels,
            activity_values=activity_values, database_error=database_error,
        )

    @app.route("/predict", methods=["GET", "POST"])
    @login_required
    def predict():
        if request.method == "GET":
            return render_template("predict.html")
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            flash("Choose a JPG or PNG leaf image to continue.", "error")
            return render_template("predict.html"), 400
        extension = Path(upload.filename).suffix.lower()
        if extension not in {".jpg", ".jpeg", ".png"}:
            flash("Only JPG, JPEG, and PNG images are supported.", "error")
            return render_template("predict.html"), 400
        try:
            upload.stream.seek(0)
            with Image.open(upload.stream) as image:
                image.verify()
            upload.stream.seek(0)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            flash("That file is not a valid image. Choose a readable JPG or PNG.", "error")
            return render_template("predict.html"), 400

        filename = f"{uuid4().hex}{extension}"
        upload_folder = Path(app.config["UPLOAD_FOLDER"])
        heatmap_folder = Path(app.config["HEATMAP_FOLDER"])
        upload_folder.mkdir(parents=True, exist_ok=True)
        heatmap_folder.mkdir(parents=True, exist_ok=True)
        image_path = upload_folder / filename
        heatmap_filename = None
        try:
            upload.save(image_path)
            model_predictor = get_predictor()
            try:
                results = model_predictor.predict(image_path)
            except (InvalidImage, ModelUnavailable):
                raise
            except Exception as exc:
                raise ModelUnavailable("The model could not analyze this image. Check the model files and try again.") from exc
            if not results:
                raise ModelUnavailable("The model did not return any class predictions.")
            best = results[0]
            try:
                processed = model_predictor.preprocess(image_path)
                import numpy as np
                import tensorflow as tf

                model_input = tf.keras.applications.mobilenet_v2.preprocess_input(processed.copy())
                heatmap = generate_gradcam(model_predictor.model, model_input, best["class_index"])
                candidate_heatmap = f"{Path(filename).stem}_heatmap.jpg"
                create_overlay(image_path, heatmap, heatmap_folder / candidate_heatmap)
                heatmap_filename = candidate_heatmap
            except GradCAMError:
                logger.info("Grad-CAM is unavailable for the loaded model", exc_info=True)
            except Exception:
                logger.info("Grad-CAM generation failed; the prediction will still be shown", exc_info=True)

            relative_image_path = f"uploads/{filename}"
            relative_heatmap_path = f"heatmaps/{heatmap_filename}" if heatmap_filename else None
            prediction_id = get_db().execute(
                "INSERT INTO predictions "
                "(user_id, image_path, heatmap_path, crop_name, disease_name, confidence, top_predictions) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    session["user_id"], relative_image_path, relative_heatmap_path,
                    best["crop_name"], best["disease_name"], best["confidence"], json.dumps(results),
                ),
            )
        except InvalidImage as exc:
            image_path.unlink(missing_ok=True)
            flash(str(exc), "error")
            return render_template("predict.html"), 400
        except ModelUnavailable as exc:
            image_path.unlink(missing_ok=True)
            flash(str(exc), "error")
            return render_template("predict.html"), 503
        except Exception:
            image_path.unlink(missing_ok=True)
            if heatmap_filename:
                (heatmap_folder / heatmap_filename).unlink(missing_ok=True)
            logger.exception("Prediction could not be completed")
            flash("Prediction could not be saved. Check the database connection and try again.", "error")
            return render_template("predict.html"), 503
        return redirect(url_for("prediction_detail", prediction_id=prediction_id))

    @app.get("/history")
    @login_required
    def history():
        search = request.args.get("q", "").strip()[:100]
        page = max(request.args.get("page", 1, type=int) or 1, 1)
        per_page = 12
        pattern = f"%{search}%"
        where = "user_id = %s"
        params = [session["user_id"]]
        if search:
            where += " AND (crop_name LIKE %s OR disease_name LIKE %s)"
            params.extend([pattern, pattern])
        try:
            db = get_db()
            total = db.fetch_one(f"SELECT COUNT(*) AS total FROM predictions WHERE {where}", tuple(params))["total"]
            rows = db.fetch_all(
                f"SELECT id, image_path, crop_name, disease_name, confidence, created_at "
                f"FROM predictions WHERE {where} ORDER BY created_at DESC LIMIT %s OFFSET %s",
                tuple(params + [per_page, (page - 1) * per_page]),
            )
        except Exception:
            logger.exception("History query failed")
            flash("Prediction history is temporarily unavailable. Check MySQL and try again.", "error")
            total, rows = 0, []
        return render_template(
            "history.html", rows=rows, search=search, page=page,
            pages=max(1, (total + per_page - 1) // per_page), total=total,
        )

    @app.get("/history/export.csv")
    @login_required
    def export_history():
        try:
            rows = get_db().fetch_all(
                "SELECT id, crop_name, disease_name, confidence, created_at FROM predictions "
                "WHERE user_id = %s ORDER BY created_at DESC", (session["user_id"],)
            )
        except Exception:
            logger.exception("History export query failed")
            abort(503, description="History export is temporarily unavailable.")
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["Prediction ID", "Crop", "Predicted disease", "Confidence", "Created at"])
        for row in rows:
            writer.writerow([row["id"], row["crop_name"], row["disease_name"], row["confidence"], row["created_at"]])
        response = make_response(output.getvalue())
        response.headers["Content-Type"] = "text/csv; charset=utf-8"
        response.headers["Content-Disposition"] = "attachment; filename=cropsense-history.csv"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/prediction/<int:prediction_id>")
    @login_required
    def prediction_detail(prediction_id):
        try:
            record = get_db().fetch_one(
                "SELECT * FROM predictions WHERE id = %s AND user_id = %s LIMIT 1",
                (prediction_id, session["user_id"]),
            )
        except Exception:
            logger.exception("Prediction detail query failed")
            abort(503, description="Prediction details are temporarily unavailable.")
        if not record:
            abort(404)
        predictions = record["top_predictions"]
        if isinstance(predictions, str):
            predictions = json.loads(predictions)
        record["top_predictions"] = predictions
        guidance = find_guidance(record["crop_name"], record["disease_name"])
        return render_template(
            "result.html", prediction=record, guidance=guidance,
            uncertain=float(record["confidence"]) < 0.60,
        )

    @app.get("/prediction/<int:prediction_id>/image")
    @login_required
    def prediction_image(prediction_id):
        return serve_prediction_file(prediction_id, "image_path", app.config["UPLOAD_FOLDER"], "uploads/")

    @app.get("/prediction/<int:prediction_id>/heatmap")
    @login_required
    def prediction_heatmap(prediction_id):
        return serve_prediction_file(prediction_id, "heatmap_path", app.config["HEATMAP_FOLDER"], "heatmaps/")

    def serve_prediction_file(prediction_id, field, folder, prefix):
        try:
            record = get_db().fetch_one(
                f"SELECT {field} FROM predictions WHERE id = %s AND user_id = %s LIMIT 1",
                (prediction_id, session["user_id"]),
            )
        except Exception:
            logger.exception("Prediction file lookup failed")
            abort(503)
        if not record or not record[field]:
            abort(404)
        relative_path = record[field]
        if not relative_path.startswith(prefix) or Path(relative_path).name != relative_path[len(prefix):]:
            abort(404)
        return send_from_directory(folder, Path(relative_path).name, as_attachment=False, max_age=0)

    @app.get("/disease-guide")
    @login_required
    def disease_guide():
        query = request.args.get("q", "")[:100]
        return render_template("disease_guide.html", entries=guide_matches(query), search=query)

    @app.route("/profile", methods=["GET", "POST"])
    @login_required
    def profile():
        try:
            db = get_db()
            if request.method == "POST":
                username = request.form.get("username", "").strip()
                if not 3 <= len(username) <= 40:
                    flash("Name must be between 3 and 40 characters.", "error")
                else:
                    conflict = db.fetch_one(
                        "SELECT id FROM users WHERE username = %s AND id <> %s LIMIT 1",
                        (username, session["user_id"]),
                    )
                    if conflict:
                        flash("That name is already in use.", "error")
                    else:
                        db.execute("UPDATE users SET username = %s WHERE id = %s", (username, session["user_id"]))
                        session["username"] = username
                        flash("Your profile was updated.", "success")
            user = db.fetch_one(
                "SELECT username, email, created_at FROM users WHERE id = %s LIMIT 1", (session["user_id"],)
            )
        except Exception:
            logger.exception("Profile query failed")
            flash("Profile information is temporarily unavailable.", "error")
            user = {"username": session.get("username", ""), "email": "", "created_at": None}
        return render_template("profile.html", user=user)

    @app.errorhandler(400)
    def bad_request(error):
        return render_template("error.html", code=400, message=getattr(error, "description", "Invalid request.")), 400

    @app.errorhandler(413)
    def too_large(_error):
        return render_template("error.html", code=413, message="That file is larger than the 8 MB upload limit."), 413

    @app.errorhandler(500)
    def internal_error(_error):
        logger.exception("Unhandled application error")
        return render_template("error.html", code=500, message="Something went wrong. Please try again."), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=app.config["DEBUG"])