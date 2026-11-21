"""Shortage prediction service, request validation, and FastAPI application."""
from __future__ import annotations

import time
from typing import Literal

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from nour.fallback import rule_based_risk
from nour.guards import redact


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branch_id: str = Field(pattern=r"^BR-\d{3}$")
    item_code: str = Field(pattern=r"^[A-Z]{2}-\d{5}$")
    weeks_of_cover: float = Field(ge=0.0, le=52.0)
    supplier_lead_time_days: int = Field(ge=0, le=180)
    orders_last_4w: int = Field(ge=0)
    supplier_fill_rate: float = Field(ge=0.0, le=1.0)
    pharmacist_note: str = Field(default="", max_length=500)


class PredictResponse(BaseModel):
    risk: float
    band: Literal["low", "medium", "high"]
    source: Literal["model", "fallback"]
    note_redacted: str
    model_version: str


def band_for(risk: float) -> str:
    """Categorize risk score into low, medium, or high band."""
    if risk < 0.3:
        return "low"
    if risk < 0.7:
        return "medium"
    return "high"


class ShortageService:
    def __init__(
        self,
        model,
        model_version: str,
        training_ranges: dict[str, tuple[float, float]],
    ):
        self.model = model
        self.model_version = model_version
        self.training_ranges = training_ranges
        self.log: list[dict] = []

    def predict(self, req: PredictRequest) -> PredictResponse:
        start_time = time.perf_counter()

        # Always redact pharmacist note first to prevent raw PII leaks
        redacted_note, _ = redact(req.pharmacist_note)

        # 1. Check feature training ranges
        out_of_range = False
        for feat, (lo, hi) in self.training_ranges.items():
            if hasattr(req, feat):
                val = getattr(req, feat)
                if not (lo <= val <= hi):
                    out_of_range = True
                    break

        if out_of_range:
            source = "fallback"
            reason = "out_of_range"
            risk = rule_based_risk(
                weeks_of_cover=req.weeks_of_cover,
                supplier_fill_rate=req.supplier_fill_rate,
                supplier_lead_time_days=req.supplier_lead_time_days,
            )
        else:
            # Column order: weeks_of_cover, supplier_lead_time_days, orders_last_4w, supplier_fill_rate
            X = np.array([[
                req.weeks_of_cover,
                req.supplier_lead_time_days,
                req.orders_last_4w,
                req.supplier_fill_rate,
            ]], dtype=float)

            try:
                proba = self.model.predict_proba(X)
                raw_risk = float(proba[0, 1])
                if not np.isfinite(raw_risk) or not (0.0 <= raw_risk <= 1.0):
                    source = "fallback"
                    reason = "invalid_output"
                    risk = rule_based_risk(
                        weeks_of_cover=req.weeks_of_cover,
                        supplier_fill_rate=req.supplier_fill_rate,
                        supplier_lead_time_days=req.supplier_lead_time_days,
                    )
                else:
                    source = "model"
                    reason = None
                    risk = raw_risk
            except Exception:
                source = "fallback"
                reason = "model_error"
                risk = rule_based_risk(
                    weeks_of_cover=req.weeks_of_cover,
                    supplier_fill_rate=req.supplier_fill_rate,
                    supplier_lead_time_days=req.supplier_lead_time_days,
                )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Log entry contains strictly redacted note and operational telemetry
        self.log.append({
            "branch_id": req.branch_id,
            "item_code": req.item_code,
            "risk": risk,
            "source": source,
            "reason": reason,
            "note": redacted_note,
            "latency_ms": latency_ms,
        })

        return PredictResponse(
            risk=risk,
            band=band_for(risk),
            source=source,
            note_redacted=redacted_note,
            model_version=self.model_version,
        )

    def metrics(self) -> dict:
        requests = len(self.log)
        if requests == 0:
            return {
                "requests": 0,
                "fallbacks": 0,
                "fallback_rate": 0.0,
                "p95_latency_ms": 0.0,
            }

        fallbacks = sum(1 for e in self.log if e["source"] == "fallback")
        fallback_rate = float(fallbacks / requests)
        latencies = [e["latency_ms"] for e in self.log]
        p95_latency = float(np.percentile(latencies, 95))

        return {
            "requests": requests,
            "fallbacks": fallbacks,
            "fallback_rate": fallback_rate,
            "p95_latency_ms": p95_latency,
        }


def create_app(service: ShortageService) -> FastAPI:
    """Create FastAPI application exposing /predict, /health, and /metrics endpoints."""
    app = FastAPI(title="Nour Pharmacy Shortage Service")

    @app.post("/predict", response_model=PredictResponse)
    def predict_endpoint(req: PredictRequest):
        return service.predict(req)

    @app.get("/health")
    def health_endpoint():
        return {
            "status": "ok",
            "model_version": service.model_version,
        }

    @app.get("/metrics")
    def metrics_endpoint():
        return service.metrics()

    return app
