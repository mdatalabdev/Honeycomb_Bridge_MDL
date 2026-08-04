# api container — see CONTAINERIZATION.md container #1.
#
# Runs api_downlink:app directly. The scheduler, backup scheduler, key-rotation
# and MQTT-listener threads that used to run alongside it in a single process
# now belong to the iot-worker/backup-worker containers instead.
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x entrypoint.sh

EXPOSE 4567
CMD ["./entrypoint.sh"]
