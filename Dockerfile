# Multi-stage build — smaller final image
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Final Stage ----
FROM python:3.11-slim

WORKDIR /app

# Dependencies copy karo
COPY --from=builder /root/.local /root/.local

# Source code copy karo
COPY serve.py .
COPY models/ models/
COPY metrics/ metrics/

# Python path
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV MODEL_VERSION=1.0.0
ENV MODEL_PATH=models/model.pkl

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
