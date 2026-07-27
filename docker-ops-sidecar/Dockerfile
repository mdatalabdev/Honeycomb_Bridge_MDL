# Alpine + the official Docker CLI, nothing else — this is the only container in the
# stack that touches /var/run/docker.sock, so it stays as small as possible.
FROM docker:25-cli

RUN apk update && apk upgrade && apk add --no-cache python3 py3-pip

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY main.py config.py ./

EXPOSE 8097
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8097"]
