# backup-worker container — see CONTAINERIZATION.md container #4.
#
# Build context is the repo root (see docker-compose.yml), not Timescale_db/:
# backup_scheduler.py locates the root config.py via a sys.path hack (its own
# module docstring explains why — it reads SMTP_* from there) to send
# failure-alert emails. Same reasoning as the api image inheriting the full
# requirements.txt — this doesn't need torch/xgboost/etc, only
# psycopg2/paramiko/apscheduler/cryptography, but there's one requirements.txt
# for the whole project today.
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# --app-dir puts Timescale_db/ on sys.path so worker:app resolves as
# Timescale_db/worker.py (a bare top-level import, same convention its sibling
# modules already use) while still running from the repo root (WORKDIR /app),
# so worker.py's own `import config` resolves too.
EXPOSE 8091
CMD ["python3", "-m", "uvicorn", "worker:app", "--app-dir", "Timescale_db", "--host", "0.0.0.0", "--port", "8091"]
