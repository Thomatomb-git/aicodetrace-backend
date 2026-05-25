# Gunakan image Python slim resmi yang ringan
FROM python:3.10-slim

# Buat user baru dengan UID 1000 (diwajibkan oleh Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Copy requirements.txt dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy seluruh source code backend ke dalam container dengan ownership user 1000
COPY --chown=user . .

# Gunakan non-root user untuk keamanan dan kompatibilitas HF Spaces
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app

# Expose port 7860 (wajib untuk Hugging Face Spaces)
EXPOSE 7860

# Jalankan FastAPI menggunakan uvicorn di port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
