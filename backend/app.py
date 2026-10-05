# SuperKart Sales Forecasting – Flask backend API

import joblib                                   # to load the serialized model
import pandas as pd                             # to build the model input
from flask import Flask, request, jsonify       # to create the API

# Initialize the Flask app
superkart_api = Flask("SuperKart")

# Load the trained pipeline (preprocessing + tuned model) once, when the server starts
model = joblib.load("superkart_model.joblib")

# Features the model expects, in the same order used during training
FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]


# Home page – a quick check that the API is running
@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Prediction API"


# Online inference – predict sales for a single product (JSON input)
@superkart_api.post("/v1/predict")
def predict_sales():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    # Check that every required feature has been sent
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    # Build a one-row DataFrame with the features in the training order
    input_data = pd.DataFrame([{f: data[f] for f in FEATURES}])

    prediction = float(model.predict(input_data)[0])
    return jsonify({"Sales": round(prediction, 2)})


# Batch inference – predict sales for every row of an uploaded CSV file
@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    if "file" not in request.files:
        return jsonify({"error": "Upload a CSV file using the key 'file'"}), 400

    input_data = pd.read_csv(request.files["file"])

    missing = [f for f in FEATURES if f not in input_data.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    predictions = model.predict(input_data[FEATURES])

    # Map each row index to its predicted sales
    output = {str(i): round(float(pred), 2) for i, pred in enumerate(predictions)}
    return jsonify(output)


# Run with Flask's development server when started directly (Gunicorn is used in Docker)
if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
