# iot-worker container — see CONTAINERIZATION.md container #2.
#
# Build context is the repo root (see docker-compose.yml), not iot_worker/:
# these modules import the root config.py, and edgex_users.json (maintained
# here, read by `api` and `notifications-worker` too — see docker-compose.yml's
# bind mounts) lives at the repo root as a plain relative path, not __file__-relative.
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8092
CMD ["python3", "-m", "uvicorn", "iot_worker.worker:app", "--host", "0.0.0.0", "--port", "8092"]
