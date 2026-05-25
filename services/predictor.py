"""
predictor.py — Logika prediksi untuk CodeBERT dan ML2.
"""

import re
import logging

import torch
import numpy as np
from scipy.sparse import hstack

from models import PredictResponse
from services import model_loader as ml

logger = logging.getLogger(__name__)


# ============================================================
# Meta Features Extraction (untuk ML2)
# ============================================================

def _detect_language(code: str) -> str:
    """
    Auto-detect bahasa kode berdasarkan pola.
    Return 'python' | 'cpp'.  Default: 'python'.
    """
    python_score = 0
    cpp_score = 0

    python_patterns = ["def ", "import ", "print(", "elif ", "from ", "self."]
    cpp_patterns = ["#include", "cout", "int main", "std::", "using namespace", "cin"]

    for p in python_patterns:
        if p in code:
            python_score += 1

    for p in cpp_patterns:
        if p in code:
            cpp_score += 1

    return "cpp" if cpp_score > python_score else "python"


def _extract_meta_features(code: str) -> list[float]:
    """
    Hitung 5 meta features: [lines, code_lines, comments, functions, blank_lines].
    Auto-detect Python/C++ lalu hitung sesuai bahasa.
    """
    lang = _detect_language(code)
    raw_lines = code.split("\n")

    total_lines = len(raw_lines)
    blank_lines = sum(1 for line in raw_lines if line.strip() == "")
    
    if lang == "python":
        comment_lines = sum(1 for line in raw_lines if line.strip().startswith("#"))
        code_lines = total_lines - blank_lines - comment_lines
        # Hitung jumlah definisi fungsi Python
        functions = sum(1 for line in raw_lines if re.match(r"\s*def\s+\w+", line))
    else:
        # C++
        comment_lines = sum(1 for line in raw_lines if line.strip().startswith("//"))
        code_lines = total_lines - blank_lines - comment_lines
        # Pola fungsi C++: return_type function_name(...)
        functions = sum(
            1 for line in raw_lines
            if re.match(r"\s*\w[\w\s\*&:<>]*\s+\w+\s*\(", line)
            and not any(kw in line for kw in ["if", "while", "for", "switch", "return", "#"])
        )

    return [float(total_lines), float(code_lines), float(comment_lines),
            float(functions), float(blank_lines)]


# ============================================================
# CodeBERT Prediction
# ============================================================

def predict_codebert(code: str) -> PredictResponse:
    """Prediksi menggunakan CodeBERT."""
    # Tokenize
    inputs = ml.codebert_tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {k: v.to(ml.codebert_device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = ml.codebert_model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).squeeze()  # [prob_human, prob_ai]

    prediction = torch.argmax(probs).item()  # 0=Human, 1=AI
    confidence = probs[prediction].item() * 100

    verdict = "ai" if prediction == 1 else "human"

    return PredictResponse(
        verdict=verdict,
        confidence=round(confidence, 2),
        model="CodeBERT",
    )


# ============================================================
# ML2 (TF-IDF) Prediction
# ============================================================

def predict_ml2(code: str) -> PredictResponse:
    """Prediksi menggunakan Logistic Regression + TF-IDF + meta features."""
    # 1. TF-IDF transform
    tfidf_vector = ml.ml2_vectorizer.transform([code])

    # 2. Meta features
    meta_raw = np.array([_extract_meta_features(code)])  # shape (1, 5)
    meta_scaled = ml.ml2_scaler.transform(meta_raw)

    # 3. Gabungkan
    combined = hstack([tfidf_vector, meta_scaled])

    # 4. Prediksi
    prediction = ml.ml2_model.predict(combined)[0]       # 0 atau 1
    proba = ml.ml2_model.predict_proba(combined)[0]       # [prob_human, prob_ai]

    confidence = proba[prediction] * 100
    verdict = "ai" if prediction == 1 else "human"

    return PredictResponse(
        verdict=verdict,
        confidence=round(confidence, 2),
        model="Logistic Regression + TF-IDF",
    )
