"""Verified model loading and creation-time-only prediction."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
import json
import re

import pandas as pd


FIELDS = {"complaint_type", "borough", "created_date"}
BOROUGHS = {"BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND", "Unspecified"}
LOCAL_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?$")
FEATURES = ["complaint_type", "borough", "month", "hour", "weekday"]


def make_features(payload: dict) -> pd.DataFrame:
    if not isinstance(payload, dict) or set(payload) != FIELDS:
        raise ValueError(f"Supply exactly these fields: {sorted(FIELDS)}")
    complaint = payload["complaint_type"]
    borough = payload["borough"]
    timestamp = payload["created_date"]
    if not isinstance(complaint, str) or not complaint.strip():
        raise ValueError("complaint_type must be nonempty text")
    if not isinstance(borough, str) or borough not in BOROUGHS:
        raise ValueError("borough is not a recognized NYC borough")
    if not isinstance(timestamp, str) or not LOCAL_ISO.fullmatch(timestamp):
        raise ValueError("created_date must be NYC local ISO date-time text")
    try:
        moment = datetime.fromisoformat(timestamp)
        pd.Timestamp(moment).tz_localize("America/New_York", ambiguous="raise", nonexistent="raise")
    except Exception as error:
        raise ValueError(f"Invalid NYC created_date: {error}") from error
    return pd.DataFrame([{
        "complaint_type": complaint,
        "borough": borough,
        "month": moment.month,
        "hour": moment.hour,
        "weekday": moment.weekday(),
    }], columns=FEATURES)


class ShadowService:
    def __init__(self, model, release: dict):
        self.model = model
        self.release = release
        encoder = model.named_steps["prepare"]
        self.known_types = set(map(str, encoder.categories_[0]))

    def predict(self, payload: dict) -> dict:
        features = make_features(payload)
        probability = float(self.model.predict_proba(features)[0][1])
        threshold = float(self.release["threshold"])
        return {
            "probability_late": round(probability, 4),
            "predicted_label": int(probability >= threshold),
            "threshold": threshold,
            "model_id": self.release.get("model_id", self.release.get("final_evaluation_id")),
            "new_complaint_type": payload["complaint_type"] not in self.known_types,
            "operational_status": "HOLD_FOR_OPERATIONAL_USE",
            "mode": "shadow",
            "review_warning": (
                "HEAT/HOT WATER had zero recall in the final test"
                if payload["complaint_type"] == "HEAT/HOT WATER" else None
            ),
        }


def load_release(directory: str | Path) -> ShadowService:
    directory = Path(directory)
    metadata = json.loads((directory / "release.json").read_text(encoding="utf-8"))
    if metadata["operational_status"] != "HOLD_FOR_OPERATIONAL_USE" or metadata.get("mode", "shadow") != "shadow":
        raise ValueError("This package only supports the held shadow release")
    if metadata["features"] != FEATURES or metadata["threshold"] != 0.2:
        raise ValueError("Unexpected feature contract or threshold")
    model_file = directory / "model.joblib"
    if hashlib.sha256(model_file.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Model file checksum changed")

    import joblib
    import numpy
    import sklearn

    actual = {
        "pandas": pd.__version__, "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
    }
    expected = {key: metadata["versions"][key] for key in actual}
    if actual != expected:
        raise RuntimeError(f"Model environment differs: expected {expected}, got {actual}")
    return ShadowService(joblib.load(model_file), metadata)
