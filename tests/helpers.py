from werkzeug.security import generate_password_hash


class FakeDatabase:
    def __init__(self):
        self.calls = []
        self.users = {
            1: {
                "id": 1,
                "username": "field-user",
                "email": "field@example.com",
                "password_hash": generate_password_hash("a-secure-password"),
                "created_at": "2026-01-01 10:00:00",
            }
        }
        self.predictions = []

    def fetch_one(self, query, params=()):
        self.calls.append(("fetch_one", query, params))
        if "SELECT id FROM users WHERE email = %s OR username = %s" in query:
            email, username = params
            return next((user for user in self.users.values() if user["email"] == email or user["username"] == username), None)
        if "SELECT id, username, password_hash FROM users WHERE email" in query:
            return next((user for user in self.users.values() if user["email"] == params[0]), None)
        if "SELECT id FROM users WHERE username" in query:
            username, user_id = params
            return next((user for user in self.users.values() if user["username"] == username and user["id"] != user_id), None)
        if "SELECT username, email, created_at FROM users" in query:
            return self.users.get(params[0])
        if "SELECT COUNT(*) AS total FROM predictions" in query:
            return {"total": sum(row["user_id"] == params[0] for row in self.predictions)}
        if "WHERE id = %s AND user_id = %s" in query:
            record_id, user_id = params
            return next((row for row in self.predictions if row["id"] == record_id and row["user_id"] == user_id), None)
        if "AS total" in query and "FROM predictions WHERE user_id" in query:
            owned = [row for row in self.predictions if row["user_id"] == params[0]]
            return {"total": len(owned), "healthy": sum("healthy" in row["disease_name"].lower() for row in owned)}
        return None

    def fetch_all(self, query, params=()):
        self.calls.append(("fetch_all", query, params))
        if "FROM predictions" not in query:
            return []
        user_id = params[0]
        rows = [row.copy() for row in self.predictions if row["user_id"] == user_id]
        if "LIKE %s" in query:
            term = params[1].strip("%")
            rows = [row for row in rows if term.casefold() in (row["crop_name"] + row["disease_name"]).casefold()]
        if "GROUP BY disease_name" in query:
            counts = {}
            for row in rows:
                counts[row["disease_name"]] = counts.get(row["disease_name"], 0) + 1
            return [{"disease_name": name, "count": count} for name, count in counts.items()]
        if "DATE_FORMAT" in query:
            return []
        if "LIMIT %s OFFSET %s" in query:
            rows = rows[(params[-1]):(params[-1] + params[-2])]
        return rows

    def execute(self, query, params=()):
        self.calls.append(("execute", query, params))
        if query.startswith("INSERT INTO users"):
            user_id = max(self.users, default=0) + 1
            self.users[user_id] = {
                "id": user_id, "username": params[0], "email": params[1],
                "password_hash": params[2], "created_at": "2026-01-01 10:00:00",
            }
            return user_id
        if query.startswith("INSERT INTO predictions"):
            record_id = len(self.predictions) + 1
            self.predictions.append({
                "id": record_id, "user_id": params[0], "image_path": params[1],
                "heatmap_path": params[2], "crop_name": params[3], "disease_name": params[4],
                "confidence": params[5], "top_predictions": params[6], "created_at": "2026-01-01 10:00:00",
            })
            return record_id
        if query.startswith("UPDATE users SET username"):
            self.users[params[1]]["username"] = params[0]
        return 1


class FakePredictor:
    model = object()

    def __init__(self):
        self.predictions = [{
            "label": "Tomato___Early_blight", "crop_name": "Tomato", "disease_name": "Early blight",
            "confidence": 0.82, "class_index": 2,
        }, {
            "label": "Tomato___Late_blight", "crop_name": "Tomato", "disease_name": "Late blight",
            "confidence": 0.12, "class_index": 3,
        }, {
            "label": "Tomato___healthy", "crop_name": "Tomato", "disease_name": "healthy",
            "confidence": 0.06, "class_index": 4,
        }]

    def predict(self, _image_path):
        return self.predictions

    def preprocess(self, _image_path):
        import numpy as np
        return np.zeros((224, 224, 3), dtype=np.float32)