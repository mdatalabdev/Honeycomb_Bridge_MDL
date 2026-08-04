"""
Container entrypoint for `backup-worker` (see CONTAINERIZATION.md container #4).

Runs as a small FastAPI app instead of a bare blocking script for one reason:
api_downlink.py's schedule endpoints and this container no longer share a
process (and so no longer share one in-memory APScheduler) once split into
separate containers. POST /reload lets api_downlink.py tell this container to
re-sync its jobs with the on-disk schedule JSON files right after writing or
deleting one, instead of waiting for a container restart to pick up the change.

Run as `python3 -m uvicorn worker:app --app-dir Timescale_db ...` from the repo
root so `import config` (for BACKUP_WORKER_SHARED_SECRET) resolves the same way
it does for a host run, and `from backup_scheduler import ...` resolves as a
sibling import (backup_scheduler.py's own sys.path self-insertion handles the
rest) — see backup_scheduler.py's module docstring.
"""

import logging
import time

import psycopg2
from fastapi import Depends, FastAPI, Header, HTTPException, status

import config
from backup_scheduler import reload_jobs, start_backup_scheduler
from db_config import APP_DB_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RETRIES = 10
RETRY_DELAY_SECONDS = 3

app = FastAPI(title="backup-worker")


def wait_for_auth_db() -> None:
    """Retry connecting to auth-db before starting the scheduler — mirrors
    auth/init_db.py's wait_for_db(), since _get_alert_emails() needs
    DATABASE_URL reachable to look up who to notify on a failed job."""
    if not APP_DB_URL:
        logger.warning("DATABASE_URL not set — skipping auth-db readiness wait")
        return

    for attempt in range(1, RETRIES + 1):
        try:
            conn = psycopg2.connect(APP_DB_URL)
            conn.close()
            return
        except psycopg2.OperationalError:
            logger.info("auth-db not ready yet (attempt %d/%d)", attempt, RETRIES)
            time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError("auth-db did not become ready in time")


def require_internal_token(x_internal_token: str = Header(default="")):
    if not config.BACKUP_WORKER_SHARED_SECRET or x_internal_token != config.BACKUP_WORKER_SHARED_SECRET:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid internal token")


@app.on_event("startup")
def _startup():
    wait_for_auth_db()
    start_backup_scheduler()
    logger.info("backup-worker running; scheduler active.")


@app.post("/reload", dependencies=[Depends(require_internal_token)])
def reload():
    reload_jobs()
    return {"status": "reloaded"}
