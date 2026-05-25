"""
model_loader.py — Load ML models dari HuggingFace Hub saat startup.
"""

import logging

import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

HF_REPO_ID = "thomatomb/nlp_project"

# ============================================================
# Global containers — diisi saat startup
# ============================================================
codebert_tokenizer = None
codebert_model = None
codebert_device = None

ml2_model = None
ml2_vectorizer = None
ml2_scaler = None


# ============================================================
# CodeBERT loader
# ============================================================
def load_codebert():
    """Load CodeBERT tokenizer + model dari HuggingFace Hub."""
    global codebert_tokenizer, codebert_model, codebert_device

    logger.info("Loading CodeBERT from HF Hub: %s (subfolder=codebert)", HF_REPO_ID)

    codebert_tokenizer = AutoTokenizer.from_pretrained(
        HF_REPO_ID, subfolder="codebert"
    )
    codebert_model = AutoModelForSequenceClassification.from_pretrained(
        HF_REPO_ID, subfolder="codebert"
    )

    # Pilih device
    codebert_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codebert_model.to(codebert_device)
    codebert_model.eval()

    logger.info("✅ CodeBERT loaded (device=%s)", codebert_device)


# ============================================================
# ML2 loader
# ============================================================
def load_ml2():
    """Load ML2 (Logistic Regression + TF-IDF) dari HuggingFace Hub."""
    global ml2_model, ml2_vectorizer, ml2_scaler

    logger.info("Loading ML2 from HF Hub: %s", HF_REPO_ID)

    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="ml2/model.joblib")
    vectorizer_path = hf_hub_download(repo_id=HF_REPO_ID, filename="ml2/vectorizer.joblib")
    scaler_path = hf_hub_download(repo_id=HF_REPO_ID, filename="ml2/scaler.joblib")

    ml2_model = joblib.load(model_path)
    ml2_vectorizer = joblib.load(vectorizer_path)
    ml2_scaler = joblib.load(scaler_path)

    logger.info("✅ ML2 (TF-IDF) loaded")


# ============================================================
# Load all
# ============================================================
def load_all_models():
    """Dipanggil saat FastAPI startup. Load semua model sekaligus."""
    logger.info("=" * 50)
    logger.info("Loading all models from HuggingFace Hub...")
    logger.info("=" * 50)

    load_codebert()
    load_ml2()

    logger.info("=" * 50)
    logger.info("All models loaded successfully! 🚀")
    logger.info("=" * 50)
