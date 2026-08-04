# ml-service container — see CONTAINERIZATION.md container #5.
#
# Build context is the repo root (see docker-compose.yml), not Predictive_ML/:
# it imports config.py and captcha_utils.py (for the shared Redis client) directly.
#
# Kept off the shared base image on purpose (torch/xgboost/scikit-learn/CUDA are
# 5-8GB+) — installs the slim root requirements.txt first, then layers
# Predictive_ML/requirements.txt (the heavy deps trimmed out of the root file)
# on top, so `api`/`iot-worker`/`backup-worker`/`notifications-worker` never
# pull any of this down.
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Predictive_ML/requirements.txt ./requirements-ml.txt
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY . .

EXPOSE 8093
CMD ["python3", "-m", "uvicorn", "Predictive_ML.worker:app", "--host", "0.0.0.0", "--port", "8093"]
