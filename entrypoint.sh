#!/bin/sh
set -e

python3 -m auth.init_db
exec python3 -m uvicorn api_downlink:app --host 0.0.0.0 --port 4567
