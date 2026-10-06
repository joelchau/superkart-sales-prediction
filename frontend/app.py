# SuperKart Sales Forecasting – Streamlit frontend

import os
import pandas as pd
import requests
import streamlit as st

# Address of the Flask backend. "backend" is the backend container's name on the shared Docker network.
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860")

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="wide")


@st.cache_data(ttl=60)
def get_model_info():
    """Reads the model card from the backend's /health endpoint (None if the backend is unreachable)."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None


def error_text(response):
    """Returns the API's error message, or the raw text if the reply is not JSON."""
    try:
        return response.json().get("error", response.text)
    except ValueError:
        return response.text


def default_index(choices, preferred):
    return choices.index(preferred) if preferred in choices else 0


def show_warnings(result):
    for message in result.get("warnings", []):
        st.warning(message)


info = get_model_info()
metrics = info["test_metrics"] if info else None
options = info["categorical_values"] if info else {
    "Product_Sugar_Content": ["Low Sugar", "No Sugar", "Regular"],
    "Product_Id_char": ["DR", "FD", "NC"],
    "Product_Type_Category": ["Non Perishables", "Perishables"],
    "Store_Type": ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"],
    "Store_Size": ["High", "Medium", "Small"],
    "Store_Location_City_Type": ["Tier 1", "Tier 2", "Tier 3"],
}

# ------------------------------------------------------------------ Sidebar: business context and backend status
with st.sidebar:
    st.header("About")
    st.write(
        "SuperKart plans procurement, inventory and regional sales strategy each quarter. "
        "This app forecasts the **total sales revenue of a product in a store** from product and store "
        "attributes, using a machine-learning model served by a Flask API."
    )
    st.header("Backend connection")
    if info:
        st.success(f"Connected to the Flask API at {BACKEND_URL} over the Docker network")
        st.write(f"**Model:** {info['model']}  \n**API version:** {info['api_version']}  \n"
                 f"**Trained on:** {info['trained_on']}")
        st.write(f"**Test accuracy:** R² {metrics['r2']:.3f} · RMSE {metrics['rmse']:,.0f} · "
                 f"average error {100 * metrics['mape']:.1f}%")
    else:
        st.error(f"The prediction API at {BACKEND_URL} is not reachable.")

st.title("🛒 SuperKart Sales Forecast")
st.write(
    "Forecast product–store sales revenue for **one product** (Single prediction) or a **whole product list** "
    "(Batch prediction) to support quarterly procurement and inventory planning."
)

tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

# ------------------------------------------------------------------ Single prediction
with tab_single:
    st.subheader("Product details")
    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product weight", min_value=0.0, max_value=50.0, value=12.66, step=0.1)
        product_mrp = st.number_input("Product MRP (maximum retail price)", min_value=0.0, max_value=500.0,
                                      value=117.08, step=1.0)
        product_allocated_area = st.number_input(
            "Allocated display area (share of the store's display area, 0–1)", min_value=0.0, max_value=1.0,
            value=0.027, step=0.001, format="%.3f"
        )
    with col2:
        product_sugar_content = st.selectbox("Sugar content", options["Product_Sugar_Content"],
                                             index=default_index(options["Product_Sugar_Content"], "Low Sugar"))
        product_id_char = st.selectbox(
            "Product group (first 2 letters of Product ID)", options["Product_Id_char"],
            index=default_index(options["Product_Id_char"], "FD"),
            help="FD = Food, DR = Drinks, NC = Non-Consumables",
        )
        product_type_category = st.selectbox(
            "Product type category", options["Product_Type_Category"],
            help="Perishables = Dairy, Meat, Fruits and Vegetables, Breakfast, Breads, Seafood",
        )

    st.subheader("Store details")
    col3, col4 = st.columns(2)
    with col3:
        store_type = st.selectbox("Store type", options["Store_Type"],
                                  index=default_index(options["Store_Type"], "Supermarket Type2"))
        store_size = st.selectbox("Store size", options["Store_Size"],
                                  index=default_index(options["Store_Size"], "Medium"))
    with col4:
        store_location_city_type = st.selectbox("City tier", options["Store_Location_City_Type"],
                                                index=default_index(options["Store_Location_City_Type"], "Tier 2"))
        store_age_years = st.number_input("Store age (years)", min_value=0, max_value=100, value=16, step=1)

    product_data = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar_content,
        "Product_Allocated_Area": product_allocated_area,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_location_city_type,
        "Store_Type": store_type,
        "Product_Id_char": product_id_char,
        "Store_Age_Years": store_age_years,
        "Product_Type_Category": product_type_category,
    }

    if st.button("Predict sales", type="primary"):
        try:
            response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                sales = result["Sales"]
                st.metric("Predicted product–store sales", f"{sales:,.2f}")
                if metrics:
                    low, high = sales * (1 - metrics["mape"]), sales * (1 + metrics["mape"])
                    st.info(
                        f"**How to read this:** on unseen test data the model's average error was "
                        f"{100 * metrics['mape']:.1f}%, so a typical range is **{low:,.0f} – {high:,.0f}**; "
                        f"{metrics['within_10_pct']:.0f}% of test predictions were within ±10%. "
                        "Price (MRP), product weight and store type have the most influence on the forecast."
                    )
                show_warnings(result)
            else:
                st.error(f"Prediction failed ({response.status_code}): {error_text(response)}")
        except requests.exceptions.RequestException as e:
            st.error(f"Unable to connect to the prediction API: {e}")

# ------------------------------------------------------------------ Batch prediction
with tab_batch:
    st.write(
        "Upload a CSV file with these columns: Product_Weight, Product_Sugar_Content, Product_Allocated_Area, "
        "Product_MRP, Store_Size, Store_Location_City_Type, Store_Type, Product_Id_char, Store_Age_Years, "
        "Product_Type_Category."
    )
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"{len(batch_df)} rows loaded")
        st.dataframe(batch_df.head())

        if st.button("Predict for batch", type="primary"):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/v1/predictbatch",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                    timeout=60,
                )
                if response.status_code == 200:
                    result = response.json()
                    batch_df["Predicted_Sales"] = [result[str(i)] for i in range(len(batch_df))]
                    st.success("Predictions completed successfully!")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Products scored", f"{len(batch_df)}")
                    m2.metric("Total predicted sales", f"{batch_df['Predicted_Sales'].sum():,.2f}")
                    m3.metric("Average per product", f"{batch_df['Predicted_Sales'].mean():,.2f}")
                    st.dataframe(batch_df)
                    if "Store_Type" in batch_df.columns:
                        st.write("**Total predicted sales by store type**")
                        st.bar_chart(batch_df.groupby("Store_Type")["Predicted_Sales"].sum())
                    show_warnings(result)
                    st.download_button(
                        "Download predictions as CSV",
                        batch_df.to_csv(index=False).encode("utf-8"),
                        file_name="superkart_predictions.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(f"Prediction failed ({response.status_code}): {error_text(response)}")
            except requests.exceptions.RequestException as e:
                st.error(f"Unable to connect to the prediction API: {e}")
