# notifications-worker container — see CONTAINERIZATION.md container #3.
#
# Build context is the repo root (see docker-compose.yml), not Notifications/:
# it imports config.py, auth/database.py, and User_token.py (for the EdgeX
# admin JWT, itself read from edgex_users.json at the repo root — see
# docker-compose.yml's bind mount, shared with `api` since that's the
# container that maintains that file). Polls EdgeX for notifications and
# writes them into auth-db's `notifications` table
# (Notifications/db_notification/models.py shares auth.database's Base).
#
# This loop used to run as a thread started by api_downlink.py's own FastAPI
# startup event — moved out to its own container so `api` doesn't duplicate it.
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "-m", "Notifications.worker"]
