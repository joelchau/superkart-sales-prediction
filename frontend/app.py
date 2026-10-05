# SuperKart Sales Forecasting – Streamlit frontend

import os
import pandas as pd
import requests
import streamlit as st

# Address of the Flask backend. "backend" is the backend container's name on the shared Docker network.
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860")

st.set_page_config(page_title="SuperKart Sales Prediction", page_icon="🛒")
st.title("🛒 SuperKart Sales Prediction")
st.write("Predict the total sales of a product in a store, for a single product or a whole CSV file.")

tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

# ------------------------------------------------------------------ Single prediction
with tab_single:
    st.subheader("Product details")
    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product weight", min_value=0.0, max_value=50.0, value=12.66, step=0.1)
        product_mrp = st.number_input("Product MRP", min_value=0.0, max_value=500.0, value=117.08, step=1.0)
        product_allocated_area = st.number_input(
            "Allocated display area (ratio)", min_value=0.0, max_value=1.0, value=0.027, step=0.001, format="%.3f"
        )
    with col2:
        product_sugar_content = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
        product_id_char = st.selectbox(
            "Product group (first 2 letters of Product ID)", ["FD", "DR", "NC"],
            help="FD = Food, DR = Drinks, NC = Non-Consumables",
        )
        product_type_category = st.selectbox("Product type category", ["Perishables", "Non Perishables"])

    st.subheader("Store details")
    col3, col4 = st.columns(2)
    with col3:
        store_type = st.selectbox(
            "Store type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"]
        )
        store_size = st.selectbox("Store size", ["Small", "Medium", "High"])
    with col4:
        store_location_city_type = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"])
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
                st.success(f"Predicted Product Store Sales Total: {response.json()['Sales']:,.2f}")
            else:
                st.error(f"Prediction failed ({response.status_code}): {response.text}")
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
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("Predict for batch", type="primary"):
            try:
                uploaded_file.seek(0)
                response = requests.post(
                    f"{BACKEND_URL}/v1/predictbatch",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                    timeout=60,
                )
                if response.status_code == 200:
                    predictions = response.json()
                    batch_df["Predicted_Sales"] = [predictions[str(i)] for i in range(len(batch_df))]
                    st.success("Predictions completed successfully!")
                    st.dataframe(batch_df, use_container_width=True)
                    st.metric("Total predicted sales", f"{batch_df['Predicted_Sales'].sum():,.2f}")
                    st.download_button(
                        "Download predictions as CSV",
                        batch_df.to_csv(index=False).encode("utf-8"),
                        file_name="superkart_predictions.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(f"Prediction failed ({response.status_code}): {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Unable to connect to the prediction API: {e}")
