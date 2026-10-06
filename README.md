# SuperKart Sales Prediction

Forecasts the sales revenue of each product in each SuperKart store. The deployed model is **Random Forest Tuned**:
test R² 0.926, RMSE 290.8, MAPE 5.1%.

## Structure
- `backend/` – Flask REST API (Gunicorn, port 7860) serving the full scikit-learn pipeline
  - `GET /`, `GET /health`, `POST /v1/predict` (JSON), `POST /v1/predictbatch` (CSV key `file`, or a JSON list)
- `frontend/` – Streamlit app (port 8501) that calls the backend at `http://backend:7860`

## Run with Docker
```bash
docker network create superkart-network
docker build -t superkart-backend ./backend
docker run -d --name backend --network superkart-network -p 7860:7860 superkart-backend
docker build -t superkart-frontend ./frontend
docker run -d --name frontend --network superkart-network -p 8501:8501 superkart-frontend
```
