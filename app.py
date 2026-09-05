import os
import csv
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
MODEL_PATH = "depression_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    raise SystemExit(f"[ERROR] Model file '{MODEL_PATH}' not found. "
                     "Run the training script first.")

# ─────────────────────────────────────────────
# LABEL ENCODERS
# Rebuilt from the training CSV if available,
# otherwise from hardcoded fallback lists.
# Unknown/custom values (e.g. "Other" cities)
# are handled by safe_encode() below.
# ─────────────────────────────────────────────
CATEGORICAL_COLS = [
    "Gender", "City", "Profession", "Sleep Duration",
    "Dietary Habits", "Degree",
    "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness",
]

FALLBACK_VALUES = {
    "Gender":      ["Female", "Male"],
    "City":        ["Ahmedabad", "Bangalore", "Bhopal", "Chennai", "Delhi",
                    "Faridabad", "Ghaziabad", "Hyderabad", "Indore", "Jaipur",
                    "Kalyan", "Kolkata", "Lucknow", "Ludhiana", "Meerut",
                    "Mumbai", "Nagpur", "Nashik", "Patna", "Pune",
                    "Rajkot", "Srinagar", "Surat", "Thane", "Vadodara",
                    "Varanasi", "Visakhapatnam"],
    "Profession":  ["Business", "Doctor", "Engineer", "Manager", "Student", "Teacher"],
    "Sleep Duration": ["5-6 hours", "7-8 hours", "Less than 5 hours", "More than 8 hours"],
    "Dietary Habits": ["Healthy", "Moderate", "Unhealthy"],
    "Degree":      ["B.Arch", "B.Com", "B.Ed", "B.Pharm", "BSc", "BCA",
                    "BE", "BTech", "BBA", "BHM",
                    "Class 12", "LLB", "MBA", "MCA", "MD",
                    "MSc", "MTech", "MPharm", "PhD"],
    "Have you ever had suicidal thoughts ?": ["No", "Yes"],
    "Family History of Mental Illness":      ["No", "Yes"],
}

encoders: dict[str, LabelEncoder] = {}

TRAINING_CSV = "student_depression_dataset.csv"
if os.path.exists(TRAINING_CSV):
    try:
        ref = pd.read_csv(TRAINING_CSV)
        for col in CATEGORICAL_COLS:
            if col in ref.columns:
                le = LabelEncoder()
                le.fit(ref[col].dropna().astype(str).unique())
                encoders[col] = le
        print("[OK] Encoders built from training CSV")
    except Exception as e:
        print(f"[WARN] Could not read training CSV ({e}), using fallback encoders")

if not encoders:
    for col, vals in FALLBACK_VALUES.items():
        le = LabelEncoder()
        le.fit(sorted(vals))
        encoders[col] = le
    print("[OK] Fallback encoders loaded")


def safe_encode(col: str, val: str) -> int:
    """
    Encode val for col using its LabelEncoder.
    If the value is unknown (e.g. a custom city the model never saw),
    we map it to the index of the most common/neutral class or 0.
    This prevents a hard crash while still returning a valid integer.
    """
    le = encoders.get(col)
    if le is None:
        return 0
    try:
        return int(le.transform([str(val)])[0])
    except ValueError:
        # Unknown label — return the middle index as a neutral fallback
        return len(le.classes_) // 2


# ─────────────────────────────────────────────
# FEATURE COLUMNS  (exact training order)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "Gender", "Age", "City", "Profession",
    "Academic Pressure", "Work Pressure", "CGPA",
    "Study Satisfaction", "Job Satisfaction",
    "Sleep Duration", "Dietary Habits", "Degree",
    "Have you ever had suicidal thoughts ?",
    "Work/Study Hours", "Financial Stress",
    "Family History of Mental Illness",
]

# Fields removed from the form — injected with defaults
DEFAULTS = {
    "Profession":      "Student",
    "Work Pressure":   0.0,
    "Job Satisfaction": 0.0,
}

# ─────────────────────────────────────────────
# CSV LOGGING
# ─────────────────────────────────────────────
LOG_FILE   = "student_predictions.csv"
LOG_HEADER = (
    ["Student Name", "Timestamp"]
    + FEATURE_COLS
    + ["Prediction", "Depression Probability (%)"]
)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(LOG_HEADER)


def log_prediction(name: str, features: dict, prediction: str, prob: float):
    row = (
        [name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        + [features.get(col, "") for col in FEATURE_COLS]
        + [prediction, f"{prob:.1f}"]
    )
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        student_name = form.get("student_name", "").strip() or "Anonymous"

        # ── City: hidden field holds the resolved value (dropdown or typed) ──
        city_val = form.get("city", "").strip()
        if not city_val:
            return jsonify({"error": "Please select or enter a city."}), 400

        # ── Degree: same pattern ────────────────────────────────────────────
        degree_val = form.get("degree", "").strip()
        if not degree_val:
            return jsonify({"error": "Please select or enter a degree."}), 400

        # ── Collect all raw inputs ──────────────────────────────────────────
        raw = {
            "Gender":        form.get("gender", "Male"),
            "Age":           float(form.get("age", 20)),
            "City":          city_val,
            "Profession":    DEFAULTS["Profession"],
            "Academic Pressure":  float(form.get("academic_pressure", 3)),
            "Work Pressure":      DEFAULTS["Work Pressure"],
            "CGPA":          float(form.get("cgpa", 7.0)),
            "Study Satisfaction": float(form.get("study_satisfaction", 3)),
            "Job Satisfaction":   DEFAULTS["Job Satisfaction"],
            "Sleep Duration":     form.get("sleep_duration", "7-8 hours"),
            "Dietary Habits":     form.get("dietary_habits", "Moderate"),
            "Degree":             degree_val,
            "Have you ever had suicidal thoughts ?": form.get("suicidal_thoughts", "No"),
            "Work/Study Hours":  float(form.get("work_study_hours", 6)),
            "Financial Stress":  float(form.get("financial_stress", 3)),
            "Family History of Mental Illness": form.get("family_history", "No"),
        }

        # ── Validation ──────────────────────────────────────────────────────
        if not (10 <= raw["Age"] <= 60):
            return jsonify({"error": "Age must be between 10 and 60."}), 400
        if not (0.0 <= raw["CGPA"] <= 10.0):
            return jsonify({"error": "CGPA must be between 0 and 10."}), 400
        if not (0 <= raw["Work/Study Hours"] <= 24):
            return jsonify({"error": "Work/Study Hours must be between 0 and 24."}), 400

        # ── Encode categorical columns ──────────────────────────────────────
        encoded = {}
        for col in FEATURE_COLS:
            val = raw[col]
            if col in encoders:
                encoded[col] = safe_encode(col, str(val))
            else:
                encoded[col] = float(val)

        # ── Build DataFrame and predict ─────────────────────────────────────
        input_df = pd.DataFrame([encoded], columns=FEATURE_COLS)

        prob_depression = float(model.predict_proba(input_df)[0][1])
        prob_pct        = round(prob_depression * 100, 1)
        prediction      = "Depressed" if prob_depression >= 0.5 else "Not Depressed"

        log_prediction(student_name, raw, prediction, prob_pct)

        return jsonify({
            "student_name": student_name,
            "prediction":   prediction,
            "probability":  prob_pct,
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid input value: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
