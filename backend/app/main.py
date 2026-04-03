from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from lime.lime_tabular import LimeTabularExplainer
from pydantic import BaseModel, Field


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"

XGB_MODEL_PATH = MODELS_ROOT / "xgboost" / "xgb_fold_0.json"
TCN_MODEL_PATH = MODELS_ROOT / "tcn" / "best_tcn_fold_0.pt"
THRESHOLD_PATH = MODELS_ROOT / "best_threshold.pkl"

FEATURES_PATH = DATA_ROOT / "features" / "features.csv"
FEATURE_META_PATH = DATA_ROOT / "features" / "feature_metadata.json"
FEATURE_BG_SAMPLE_PATH = DATA_ROOT / "features" / "shap_background_sample.csv"
RAW_TRAIN_PATH = DATA_ROOT / "processed" / "eda_enriched.csv"

TARGET = "deterioration_next_12h"
SEQ_LEN = 6
XGB_WEIGHT = 0.6
TCN_WEIGHT = 0.4
DEFAULT_THRESHOLD = 0.24


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out[:, :, : -self.conv.padding[0]]
        return self.relu(out)


class TCN(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(num_features, 64, dilation=1),
            TCNBlock(64, 64, dilation=2),
            TCNBlock(64, 64, dilation=4),
            TCNBlock(64, 64, dilation=8),
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]
        return self.fc(out).squeeze(-1)


class PatientInput(BaseModel):
    hour_from_admission: float = Field(..., ge=0)
    heart_rate: float = Field(..., ge=0)
    respiratory_rate: float = Field(..., ge=0)
    spo2_pct: float = Field(..., ge=0, le=100)
    temperature_c: float = Field(..., ge=25, le=45)
    systolic_bp: float = Field(..., ge=0)
    diastolic_bp: float = Field(..., ge=0)
    oxygen_flow: float = Field(..., ge=0)
    mobility_score: float = Field(..., ge=0, le=5)
    wbc_count: float = Field(..., ge=0)
    lactate: float = Field(..., ge=0)
    creatinine: float = Field(..., ge=0)
    crp_level: float = Field(..., ge=0)
    hemoglobin: float = Field(..., ge=0)
    age: float = Field(..., ge=0)
    comorbidity_index: float = Field(..., ge=0)
    gender: str
    admission_type: str
    oxygen_device: str
    nurse_alert: bool


class ModelArtifacts:
    def __init__(self) -> None:
        self.feature_columns: list[str] = []
        self.feature_defaults: dict[str, float] = {}
        self.category_maps: dict[str, dict[str, int]] = {}
        self.hour_bins: np.ndarray | None = None
        self.threshold: float = DEFAULT_THRESHOLD

        self.xgb_model: xgb.XGBClassifier | None = None
        self.tcn_model: TCN | None = None

        self.shap_explainer: shap.TreeExplainer | None = None
        self.shap_global_importance: list[dict[str, float | str]] = []
        self.lime_explainer: LimeTabularExplainer | None = None


artifacts = ModelArtifacts()
app = FastAPI(title="AI Early Warning API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_reference_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    feature_df = pd.read_csv(FEATURES_PATH) if FEATURES_PATH.exists() else pd.DataFrame()
    raw_df = pd.read_csv(RAW_TRAIN_PATH)
    meta = None
    if FEATURE_META_PATH.exists():
        import json

        with open(FEATURE_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return feature_df, raw_df, meta


def _build_category_maps(raw_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for col in ["oxygen_device", "gender", "admission_type"]:
        cats = raw_df[col].astype("category").cat.categories
        maps[col] = {str(cat): int(code) for code, cat in enumerate(cats)}
    return maps


def _load_models(num_features: int) -> tuple[xgb.XGBClassifier, TCN, float]:
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)

    tcn_model = TCN(num_features)
    tcn_model.load_state_dict(torch.load(TCN_MODEL_PATH, map_location="cpu"))
    tcn_model.eval()

    threshold = float(joblib.load(THRESHOLD_PATH)) if THRESHOLD_PATH.exists() else DEFAULT_THRESHOLD
    return xgb_model, tcn_model, threshold


def _make_hour_bins(raw_df: pd.DataFrame) -> np.ndarray:
    bins = np.unique(np.quantile(raw_df["hour_from_admission"], np.linspace(0, 1, 6)))
    if len(bins) < 2:
        return np.array([0, 6, 12, 24, 48, 1000], dtype=float)
    return bins


def _normalize_categorical_value(field: str, value: str) -> str:
    v = value.strip().lower()

    if field == "gender":
        aliases = {"male": "M", "m": "M", "female": "F", "f": "F"}
        return aliases.get(v, value)

    if field == "admission_type":
        aliases = {
            "elective": "Elective",
            "emergency": "ED",
            "ed": "ED",
            "urgent": "Transfer",
            "transfer": "Transfer",
        }
        return aliases.get(v, value)

    if field == "oxygen_device":
        aliases = {
            "none": "none",
            "nasal": "nasal",
            "nasal cannula": "nasal",
            "mask": "mask",
            "ventilator": "niv",
            "niv": "niv",
            "hfnc": "hfnc",
        }
        return aliases.get(v, value)

    return value


def _encode_category(field: str, value: str) -> int:
    normalized = _normalize_categorical_value(field, value)
    category_map = artifacts.category_maps[field]

    if normalized in category_map:
        return category_map[normalized]

    lowered = {k.lower(): v for k, v in category_map.items()}
    if normalized.lower() in lowered:
        return lowered[normalized.lower()]

    raise HTTPException(status_code=422, detail=f"Unsupported value '{value}' for {field}.")


def _compute_news2(row: pd.Series) -> int:
    score = 0

    rr = row["respiratory_rate"]
    if rr <= 8 or rr >= 25:
        score += 3
    elif 21 <= rr <= 24:
        score += 2
    elif 9 <= rr <= 11:
        score += 1

    spo2 = row["spo2_pct"]
    if spo2 <= 91:
        score += 3
    elif spo2 <= 93:
        score += 2
    elif spo2 <= 95:
        score += 1

    temp = row["temperature_c"]
    if temp <= 35:
        score += 3
    elif temp >= 39.1:
        score += 2
    elif temp <= 36:
        score += 1

    sbp = row["systolic_bp"]
    if sbp <= 90:
        score += 3
    elif sbp <= 100:
        score += 2
    elif sbp <= 110:
        score += 1

    hr = row["heart_rate"]
    if hr <= 40 or hr >= 131:
        score += 3
    elif hr >= 111:
        score += 2
    elif hr >= 91:
        score += 1

    return score


def _apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["patient_id", "hour_from_admission"]).reset_index(drop=True)

    df["MAP"] = (df["systolic_bp"] + 2 * df["diastolic_bp"]) / 3
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["spo2_fio2"] = df["spo2_pct"] / (21 + 4 * df["oxygen_flow"])

    vitals = ["heart_rate", "respiratory_rate", "spo2_pct", "temperature_c", "lactate"]

    for col in vitals:
        df[f"{col}_roc"] = df.groupby("patient_id")[col].diff(2) / 2

    for col in vitals:
        df[f"{col}_acc"] = df.groupby("patient_id")[f"{col}_roc"].diff()

    window = 6
    for col in vitals:
        rolling_mean = df.groupby("patient_id")[col].rolling(window).mean().reset_index(level=0, drop=True)
        rolling_std = df.groupby("patient_id")[col].rolling(window).std().reset_index(level=0, drop=True)

        df[f"{col}_roll_mean"] = rolling_mean
        df[f"{col}_roll_std"] = rolling_std
        df[f"{col}_cv"] = rolling_std / (rolling_mean + 1e-6)

    df["tachycardia_flag"] = (df["heart_rate"] > 100).astype(int)
    df["tachycardia_flag"] = (
        df.groupby("patient_id")["tachycardia_flag"].rolling(3).sum().reset_index(level=0, drop=True) >= 3
    ).astype(int)

    df["temp_diff"] = df.groupby("patient_id")["temperature_c"].diff()
    df["fever_trend"] = (df["temp_diff"] > 0).astype(int)
    df["fever_trend"] = (
        df.groupby("patient_id")["fever_trend"].rolling(3).sum().reset_index(level=0, drop=True) >= 3
    ).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_from_admission"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_from_admission"] / 24)

    df["time_bucket"] = pd.cut(
        df["hour_from_admission"],
        bins=artifacts.hour_bins,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )

    df["NEWS2"] = df.apply(_compute_news2, axis=1)

    df["sofa_proxy"] = (
        df["lactate"]
        + df["creatinine"]
        + (100 - df["spo2_pct"])
        + (100 - df["systolic_bp"])
    )

    df = df.fillna(method="ffill")
    df = df.fillna(method="bfill")
    df = df.fillna(0)

    return df


def _build_history_df(payload: PatientInput) -> pd.DataFrame:
    gender_code = _encode_category("gender", payload.gender)
    admission_code = _encode_category("admission_type", payload.admission_type)
    oxygen_code = _encode_category("oxygen_device", payload.oxygen_device)

    hour = float(payload.hour_from_admission)
    start_hour = max(0.0, hour - (SEQ_LEN - 1))
    hours = [start_hour + i for i in range(SEQ_LEN)]

    base = {
        "heart_rate": payload.heart_rate,
        "respiratory_rate": payload.respiratory_rate,
        "spo2_pct": payload.spo2_pct,
        "temperature_c": payload.temperature_c,
        "systolic_bp": payload.systolic_bp,
        "diastolic_bp": payload.diastolic_bp,
        "oxygen_device": oxygen_code,
        "oxygen_flow": payload.oxygen_flow,
        "mobility_score": payload.mobility_score,
        "nurse_alert": int(payload.nurse_alert),
        "wbc_count": payload.wbc_count,
        "lactate": payload.lactate,
        "creatinine": payload.creatinine,
        "crp_level": payload.crp_level,
        "hemoglobin": payload.hemoglobin,
        "age": payload.age,
        "gender": gender_code,
        "comorbidity_index": payload.comorbidity_index,
        "admission_type": admission_code,
        # These two existed pre-engineering in training data.
        "sepsis_risk_score": float(artifacts.feature_defaults.get("sepsis_risk_score", 0.0)),
        "hr_std": 0.0,
        "patient_id": -1,
    }

    rows: list[dict[str, Any]] = []
    for h in hours:
        row = dict(base)
        row["hour_from_admission"] = h
        rows.append(row)

    history_df = pd.DataFrame(rows)
    history_df["hr_std"] = float(np.std(history_df["heart_rate"].values))
    return history_df


def _align_features(df: pd.DataFrame) -> pd.DataFrame:
    aligned = df.copy()
    for col in artifacts.feature_columns:
        if col not in aligned.columns:
            aligned[col] = artifacts.feature_defaults.get(col, 0.0)
    aligned = aligned[artifacts.feature_columns]
    aligned = aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return aligned


def _predict_xgb(x: pd.DataFrame) -> float:
    prob = artifacts.xgb_model.predict_proba(x)[:, 1]
    return float(prob[0])


def _predict_tcn(x_seq: np.ndarray) -> float:
    with torch.no_grad():
        tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0)
        prob = torch.sigmoid(artifacts.tcn_model(tensor)).item()
    return float(prob)


def _risk_level(score: float, threshold: float) -> str:
    if score >= threshold + 0.2:
        return "High"
    if score >= threshold:
        return "Medium"
    return "Low"


def _build_xgb_predict_fn() -> Any:
    def _predict_fn(data: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(data, columns=artifacts.feature_columns)
        return artifacts.xgb_model.predict_proba(frame)

    return _predict_fn


def _shap_local_and_text(x_row: pd.DataFrame, top_k: int = 8) -> tuple[list[dict[str, float | str]], str]:
    shap_values = artifacts.shap_explainer.shap_values(x_row)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    row_values = shap_values[0]
    contributions: list[dict[str, float | str]] = []

    for idx, feat in enumerate(artifacts.feature_columns):
        contributions.append(
            {
                "feature": feat,
                "value": float(x_row.iloc[0, idx]),
                "shap_value": float(row_values[idx]),
                "impact": "increase" if row_values[idx] >= 0 else "decrease",
            }
        )

    contributions.sort(key=lambda d: abs(float(d["shap_value"])), reverse=True)
    top = contributions[:top_k]

    positive = [c["feature"] for c in top if float(c["shap_value"]) > 0][:3]
    negative = [c["feature"] for c in top if float(c["shap_value"]) < 0][:3]

    parts: list[str] = []
    if positive:
        parts.append(f"Risk increased mostly by {', '.join(positive)}")
    if negative:
        parts.append(f"while {', '.join(negative)} reduced risk")
    text = "; ".join(parts) if parts else "No dominant SHAP contributors for this prediction."

    return top, text


def _lime_local(x_row: pd.DataFrame, top_k: int = 8) -> tuple[list[dict[str, float | str]], str]:
    exp = artifacts.lime_explainer.explain_instance(
        data_row=x_row.values[0],
        predict_fn=_build_xgb_predict_fn(),
        num_features=top_k,
    )
    pairs = exp.as_list()
    features: list[dict[str, float | str]] = []
    for feat, weight in pairs:
        features.append(
            {
                "rule": feat,
                "weight": float(weight),
                "impact": "increase" if weight >= 0 else "decrease",
            }
        )

    increase_reasons = [p[0] for p in pairs if p[1] > 0][:2]
    sentence = (
        f"Higher XGBoost risk is mainly driven by {', '.join(increase_reasons)}."
        if increase_reasons
        else "LIME found no strong positive contributors for this XGBoost prediction."
    )
    return features, sentence


@app.on_event("startup")
def startup_event() -> None:
    feature_df, raw_df, meta = _load_reference_data()

    if meta is not None:
        artifacts.feature_columns = list(meta["feature_columns"])
        artifacts.feature_defaults = {k: float(v) for k, v in meta["feature_defaults"].items()}
    else:
        artifacts.feature_columns = [c for c in feature_df.columns if c not in ["patient_id", TARGET]]
        artifacts.feature_defaults = feature_df[artifacts.feature_columns].median(numeric_only=True).to_dict()

    artifacts.category_maps = _build_category_maps(raw_df)
    artifacts.hour_bins = _make_hour_bins(raw_df)

    xgb_model, tcn_model, threshold = _load_models(num_features=len(artifacts.feature_columns))
    artifacts.xgb_model = xgb_model
    artifacts.tcn_model = tcn_model
    artifacts.threshold = threshold

    if FEATURE_BG_SAMPLE_PATH.exists():
        background = pd.read_csv(FEATURE_BG_SAMPLE_PATH)
    elif not feature_df.empty:
        background = feature_df[artifacts.feature_columns].sample(min(500, len(feature_df)), random_state=42)
    else:
        raise RuntimeError("Missing SHAP background data. Provide features.csv or shap_background_sample.csv.")

    background = background[artifacts.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    artifacts.shap_explainer = shap.TreeExplainer(artifacts.xgb_model)
    bg_shap = artifacts.shap_explainer.shap_values(background)
    if isinstance(bg_shap, list):
        bg_shap = bg_shap[1]
    mean_abs = np.abs(bg_shap).mean(axis=0)

    ranked = sorted(
        [{"feature": artifacts.feature_columns[i], "mean_abs_shap": float(v)} for i, v in enumerate(mean_abs)],
        key=lambda d: d["mean_abs_shap"],
        reverse=True,
    )
    artifacts.shap_global_importance = ranked[:10]

    artifacts.lime_explainer = LimeTabularExplainer(
        training_data=background.values,
        feature_names=artifacts.feature_columns,
        class_names=["No Deterioration", "Deterioration"],
        mode="classification",
        discretize_continuous=True,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PatientInput) -> dict[str, Any]:
    if artifacts.xgb_model is None or artifacts.tcn_model is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    history_df = _build_history_df(payload)
    engineered = _apply_feature_engineering(history_df)
    aligned = _align_features(engineered)

    xgb_input = aligned.tail(1)
    seq_input = aligned.tail(SEQ_LEN).to_numpy(dtype=np.float32)

    xgb_prob = _predict_xgb(xgb_input)
    tcn_prob = _predict_tcn(seq_input)
    ensemble_prob = XGB_WEIGHT * xgb_prob + TCN_WEIGHT * tcn_prob

    threshold = artifacts.threshold
    pred_label = int(ensemble_prob >= threshold)
    risk_level = _risk_level(ensemble_prob, threshold)

    shap_local, shap_text = _shap_local_and_text(xgb_input)
    lime_local, lime_text = _lime_local(xgb_input)

    return {
        "risk_level": risk_level,
        "risk_score_pct": round(ensemble_prob * 100, 2),
        "predicted_class": pred_label,
        "threshold": threshold,
        "components": {
            "xgboost_probability": round(xgb_prob, 6),
            "tcn_probability": round(tcn_prob, 6),
            "ensemble_probability": round(ensemble_prob, 6),
            "weights": {"xgboost": XGB_WEIGHT, "tcn": TCN_WEIGHT},
        },
        "explanations": {
            "shap": {
                "global_top_features": artifacts.shap_global_importance,
                "local_top_contributors": shap_local,
                "summary": shap_text,
            },
            "lime": {
                "local_rules": lime_local,
                "summary": lime_text,
            },
        },
    }