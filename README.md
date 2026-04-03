# AI Early Warning System (XGBoost + TCN Ensemble)

This project provides a full-stack app for early warning of patient physiological deterioration.

## Stack

- Frontend: React + Vite
- Backend: FastAPI
- Models: XGBoost + PyTorch TCN ensemble
- Explainability: SHAP + LIME (XGBoost only)

## Required model artifacts

- `models/xgboost/xgb_fold_0.json`
- `models/tcn/best_tcn_fold_0.pt`
- `models/best_threshold.pkl`

## Prediction logic

- `ensemble = 0.6 * xgboost + 0.4 * tcn`
- Threshold loaded from `models/best_threshold.pkl` (fallback `0.24`)

## Backend setup

From project root:

```bash
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

Python compatibility notes:

- Python 3.11 uses `torch==2.5.1` and `numpy==1.26.4`
- Python 3.13 uses `torch==2.11.0` and `numpy==2.3.2`
- These are handled automatically by environment markers in `requirements.txt`

If you see `WinError 10013` when starting Uvicorn, the port is blocked/forbidden in your environment. Use another port:

```bash
uvicorn backend.app.main:app --reload --port 8001
```

If backend is on `8001`, update frontend proxy target in `frontend/vite.config.js` to `http://localhost:8001`.

If PyTorch installation fails on Windows due to long-path CUDA header extraction, install CPU PyTorch first and then rerun requirements:

```bash
python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Frontend setup

From `frontend/`:

```bash
npm install
npm run dev
```

Frontend runs on `http://localhost:5173` and proxies `/predict` to backend `http://localhost:8000`.

## API

- `POST /predict` accepts raw patient input fields from the UI and returns:
  - Risk level (Low/Medium/High)
  - Risk score (%)
  - Component probabilities (XGBoost, TCN, ensemble)
  - SHAP global + local explanations
  - LIME local explanation

## Notes on pipeline parity

- Backend reproduces feature engineering equations from notebooks:
  - MAP, pulse pressure, SpO2/FiO2
  - ROC/acceleration, rolling stats/CV
  - tachycardia/fever trend
  - NEWS2, sofa_proxy, cyclical hour features
- TCN input is a 6-step sequence constructed from current input context for real-time inference.