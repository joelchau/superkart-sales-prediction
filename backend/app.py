# SuperKart Sales Forecasting – Flask backend API

import json
import joblib                                   # to load the serialized pipeline
import pandas as pd                             # to build the model input
from flask import Flask, request, jsonify       # to create the API

API_VERSION = "v10"

# Initialize the Flask app
superkart_api = Flask("SuperKart")

# Load the trained pipeline (imputers + encoder + tuned model) and its model card once, at start-up
model = joblib.load("superkart_model.joblib")
with open("model_info.json") as f:
    MODEL_INFO = json.load(f)

FEATURES = MODEL_INFO["features"]                       # in the training order
ALLOWED_VALUES = MODEL_INFO["categorical_values"]       # labels seen in training
NUMERIC_RANGES = {                                      # (minimum, maximum) – None = no upper limit
    "Product_Weight": (0, None),
    "Product_Allocated_Area": (0, 1),
    "Product_MRP": (0, None),
    "Store_Age_Years": (0, None),
}


def rows_text(index):
    """Formats a list of row numbers for messages (first 10 only)."""
    rows = [int(i) for i in index]
    return str(rows[:10]) + (" ..." if len(rows) > 10 else "")


def validate(df):
    """Checks and cleans the input. Returns (clean DataFrame, list of errors, list of warnings)."""
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        return None, [f"Missing features: {missing}"], []

    df = df[FEATURES].copy()
    errors, warnings = [], []

    # Numeric features: must be numbers within a sensible range (blank values are imputed by the pipeline)
    for col, (low, high) in NUMERIC_RANGES.items():
        values = pd.to_numeric(df[col], errors="coerce")
        not_numeric = values.isna() & df[col].notna()
        if not_numeric.any():
            errors.append(f"{col} must be a number (rows {rows_text(df.index[not_numeric])})")
            continue
        out_of_range = values < low
        if high is not None:
            out_of_range = out_of_range | (values > high)
        if out_of_range.any():
            limit = f"between {low} and {high}" if high is not None else f"at least {low}"
            errors.append(f"{col} must be {limit} (rows {rows_text(df.index[out_of_range])})")
        df[col] = values

    # Categorical features: clean the text and flag labels the model never saw in training
    for col, allowed in ALLOWED_VALUES.items():
        df[col] = df[col].where(df[col].isna(), df[col].astype(str).str.strip())
        if col == "Product_Sugar_Content":
            df[col] = df[col].replace("reg", "Regular")          # same cleaning as in training
        unseen = df[col].notna() & ~df[col].isin(allowed)
        for label in df.loc[unseen, col].unique():
            rows = df.index[df[col] == label]
            warnings.append(f"{col} '{label}' (rows {rows_text(rows)}) was not seen in training; the model "
                            f"ignores it for this feature. Allowed values: {allowed}")
    return df, errors, warnings


# Welcome page – a quick check that the API is running
@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Prediction API"


# Health check – status and model details, used by the frontend and for monitoring
@superkart_api.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "api_version": API_VERSION,
        "model": MODEL_INFO["model_name"],
        "trained_on": MODEL_INFO["trained_on"],
        "test_metrics": MODEL_INFO["test_metrics"],
        "features": FEATURES,
        "categorical_values": ALLOWED_VALUES,
    })


# Online inference – predict sales for a single product (JSON object)
@superkart_api.post("/v1/predict")
def predict_sales():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object with the product and store features"}), 400

    input_data, errors, warnings = validate(pd.DataFrame([data]))
    if errors:
        return jsonify({"error": "; ".join(errors)}), 400

    prediction = float(model.predict(input_data)[0])
    result = {"Sales": round(prediction, 2)}
    if warnings:
        result["warnings"] = warnings
    return jsonify(result)


# Batch inference – predict sales for every row of an uploaded CSV file (key "file") or a JSON list
@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    if "file" in request.files:
        try:
            input_data = pd.read_csv(request.files["file"])
        except Exception:
            return jsonify({"error": "The uploaded file could not be read as a CSV"}), 400
    elif isinstance(request.get_json(silent=True), list):
        input_data = pd.DataFrame(request.get_json(silent=True))
    else:
        return jsonify({"error": "Upload a CSV file using the key 'file', or send a JSON list of products"}), 400

    if input_data.empty:
        return jsonify({"error": "The input contains no rows"}), 400

    input_data = input_data.reset_index(drop=True)
    clean_data, errors, warnings = validate(input_data)
    if errors:
        return jsonify({"error": "; ".join(errors)}), 400

    predictions = model.predict(clean_data)

    # Map each row index to its predicted sales
    output = {str(i): round(float(pred), 2) for i, pred in enumerate(predictions)}
    if warnings:
        output["warnings"] = warnings
    return jsonify(output)


# Run with Flask's development server when started directly (Gunicorn is used in Docker)
if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
