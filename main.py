"""
AICodeTrace Backend — FastAPI Application

Endpoints:
  POST /predict  →  Deteksi apakah kode buatan AI atau manusia.

Jalankan lokal:
  uvicorn main:app --host 0.0.0.0 --port 5000

Deploy Railway:
  Railway akan set PORT env variable otomatis.
"""

import os
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import PredictRequest, PredictResponse
from services.model_loader import load_all_models
from services.predictor import predict_codebert, predict_ml2

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Lifespan — load models saat startup
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load semua model saat server startup."""
    load_all_models()
    yield  # Server berjalan
    logger.info("Server shutting down...")


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="AICodeTrace API",
    description="API untuk mendeteksi apakah kode buatan AI atau manusia.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — izinkan semua origin (untuk development & production)
# Jika sudah tahu domain frontend, bisa di-restrict di sini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Routes
# ============================================================

# Map model key → handler
_MODEL_HANDLERS = {
    "codebert": predict_codebert,
    "logistictfidf": predict_ml2,
}

# Model names yang valid (untuk pesan error)
_VALID_MODELS = ["codebert", "logistictfidf", "logisticwordembedding"]


@app.get("/")
async def root():
    """Endpoint health check untuk mencegah server tidur (bisa di-ping)."""
    return {"status": "healthy", "message": "AICodeTrace API is active"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Menerima kode dan nama model, mengembalikan verdict (human/ai),
    confidence (0-100), dan nama model yang dipakai.
    """
    # --- Validasi: kode tidak boleh kosong ---
    code = req.code.strip()
    if not code:
        raise HTTPException(status_code=400, detail="Kode tidak boleh kosong.")

    # --- Validasi: model harus dikenali ---
    model_key = req.model.strip().lower()
    if model_key not in _VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' tidak dikenali. "
                   f"Pilihan: {', '.join(_VALID_MODELS)}",
        )

    # --- ML1 (Word Embedding) — tidak tersedia ---
    if model_key == "logisticwordembedding":
        raise HTTPException(
            status_code=400,
            detail="Model 'Logistic Regression + Word Embedding' tidak tersedia. "
                   "Silakan gunakan model lain.",
        )

    # --- Jalankan prediksi ---
    handler = _MODEL_HANDLERS[model_key]

    try:
        result = handler(code)
    except Exception as e:
        logger.exception("Error saat prediksi dengan model '%s'", model_key)
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi error saat memproses prediksi: {str(e)}",
        )

    return result


# ============================================================
# Entrypoint — untuk Railway dan local
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
