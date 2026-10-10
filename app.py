import os
import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
FEATURES = bundle["features"]

# ---- Form definition (order of prediction comes from the trained model) -----
NUMERIC_FIELDS = {
    "age": ("Age", "years", 1, 120),
    "bp": ("Blood Pressure", "mm/Hg", 40, 220),
    "bgr": ("Blood Glucose Random", "mgs/dl", 20, 600),
    "bu": ("Blood Urea", "mgs/dl", 1, 400),
    "sc": ("Serum Creatinine", "mgs/dl", 0.1, 80),
    "sod": ("Sodium", "mEq/L", 80, 200),
    "pot": ("Potassium", "mEq/L", 1, 50),
    "hemo": ("Hemoglobin", "gms", 1, 25),
    "pcv": ("Packed Cell Volume", "%", 5, 70),
    "wc": ("White Blood Cell Count", "cells/cumm", 1000, 30000),
    "rc": ("Red Blood Cell Count", "millions/cmm", 1, 10),
}

# Ordinal lab values given as a fixed set of choices
CHOICE_FIELDS = {
    "al": ("Albumin", ["0", "1", "2", "3", "4", "5"]),
    "su": ("Sugar", ["0", "1", "2", "3", "4", "5"]),
}

# Binary fields: user selects a label, we convert it to 0 or 1.
# 0 = not present / normal / good   |   1 = present / abnormal / poor
BINARY_FIELDS = {
    "rbc": ("Red Blood Cells", "Normal", "Abnormal"),
    "pc": ("Pus Cells", "Normal", "Abnormal"),
    "pcc": ("Pus Cell Clumps", "Not Present", "Present"),
    "ba": ("Bacteria", "Not Present", "Present"),
    "htn": ("Hypertension", "Not Present", "Present"),
    "dm": ("Diabetes Mellitus", "Not Present", "Present"),
    "cad": ("Coronary Artery Disease", "Not Present", "Present"),
    "appet": ("Appetite", "Good", "Poor"),
    "pe": ("Pedal Edema", "Not Present", "Present"),
    "ane": ("Anemia", "Not Present", "Present"),
}


def build_context(**extra):
    ctx = dict(
        numeric=NUMERIC_FIELDS,
        choices=CHOICE_FIELDS,
        binary=BINARY_FIELDS,
        values={},
        result=None,
        error=None,
    )
    ctx.update(extra)
    return ctx


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", **build_context())


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    try:
        row = []
        for name in FEATURES:
            raw = form.get(name, "").strip()
            if raw == "":
                raise ValueError(f"Please fill in: {name}")
            # Select options send "0" or "1" already; numbers are parsed as floats
            row.append(float(raw))

        x = np.array(row, dtype=float).reshape(1, -1)
        pred = int(model.predict(x)[0])
        proba = float(model.predict_proba(x)[0][1]) * 100

        result = {
            "present": pred == 1,
            "label": "Chronic Kidney Disease Present" if pred == 1 else "Chronic Kidney Disease Not Present",
            "probability": round(proba, 1),
        }
        return render_template("index.html", **build_context(values=form.to_dict(), result=result))
    except ValueError as e:
        return render_template("index.html", **build_context(values=form.to_dict(), error=str(e))), 400


if __name__ == "__main__":
    app.run(debug=True)
