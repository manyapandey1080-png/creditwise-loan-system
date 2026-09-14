import os
import pickle
import sqlite3
from datetime import datetime

import numpy as np
from flask import Flask, render_template, request, jsonify, g

app = Flask(__name__)

MODEL_DIR = "model"
DB_PATH = "predictions.db"

# ---------------------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------------------
with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb") as f:
    encoders = pickle.load(f)
with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)

# metrics.pkl is optional (only present if you re-ran the updated train_model.py)
metrics = None
metrics_path = os.path.join(MODEL_DIR, "metrics.pkl")
if os.path.exists(metrics_path):
    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)

CATEGORICAL_COLS = list(encoders.keys())  # e.g. person_gender, person_education, ...
NUMERIC_COLS = [c for c in feature_names if c not in CATEGORICAL_COLS]

# Dropdown options come straight from the encoders that were fit during
# training, so the UI can never offer a value the model doesn't understand.
DROPDOWN_OPTIONS = {
    col: list(encoders[col].classes_) for col in CATEGORICAL_COLS
}


# ---------------------------------------------------------------------------
# Tiny SQLite history log
# ---------------------------------------------------------------------------
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
    return db


@app.teardown_appcontext
def close_db(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            prediction TEXT NOT NULL,
            probability REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------------------
# Core prediction logic (shared by the form route and the JSON API route)
# ---------------------------------------------------------------------------
class ValidationError(Exception):
    pass


def build_feature_vector(payload: dict) -> np.ndarray:
    """Validate + encode + scale a raw input dict into model-ready shape."""
    row = []
    for col in feature_names:
        if col not in payload or str(payload[col]).strip() == "":
            raise ValidationError(f"Missing value for '{col}'")

        raw_val = payload[col]

        if col in CATEGORICAL_COLS:
            le = encoders[col]
            if raw_val not in le.classes_:
                raise ValidationError(
                    f"'{raw_val}' is not a valid value for '{col}'. "
                    f"Expected one of: {list(le.classes_)}"
                )
            row.append(le.transform([raw_val])[0])
        else:
            try:
                num = float(raw_val)
            except (TypeError, ValueError):
                raise ValidationError(f"'{col}' must be a number")
            if num < 0:
                raise ValidationError(f"'{col}' cannot be negative")
            row.append(num)

    X = np.array(row).reshape(1, -1)
    X_scaled = scaler.transform(X)
    return X_scaled


def predict(payload: dict) -> dict:
    X_scaled = build_feature_vector(payload)
    pred = int(model.predict(X_scaled)[0])
    proba = float(model.predict_proba(X_scaled)[0][1])  # P(approved)

    result = {
        "prediction": "Approved" if pred == 1 else "Rejected",
        "probability": round(proba, 4),
    }

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (created_at, inputs_json, prediction, probability) VALUES (?, ?, ?, ?)",
        (
            datetime.utcnow().isoformat(),
            str(payload),
            result["prediction"],
            result["probability"],
        ),
    )
    conn.commit()
    conn.close()

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        numeric_cols=NUMERIC_COLS,
        categorical_cols=CATEGORICAL_COLS,
        dropdown_options=DROPDOWN_OPTIONS,
    )


@app.route("/predict", methods=["POST"])
def predict_form():
    payload = {col: request.form.get(col) for col in feature_names}
    try:
        result = predict(payload)
    except ValidationError as e:
        return render_template(
            "index.html",
            numeric_cols=NUMERIC_COLS,
            categorical_cols=CATEGORICAL_COLS,
            dropdown_options=DROPDOWN_OPTIONS,
            error=str(e),
            form_values=payload,
        )
    return render_template(
        "result.html",
        prediction=result["prediction"],
        probability=result["probability"],
    )


@app.route("/api/predict", methods=["POST"])
def predict_api():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON"}), 400
    try:
        result = predict(payload)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(result), 200


@app.route("/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()

    total = len(rows)
    approved = sum(1 for r in rows if r["prediction"] == "Approved")
    approval_rate = round((approved / total) * 100, 1) if total else 0

    return render_template(
        "history.html", rows=rows, approval_rate=approval_rate, total=total
    )


@app.route("/metrics")
def metrics_page():
    return render_template("metrics.html", metrics=metrics)


if __name__ == "__main__":
    app.run(debug=True)
