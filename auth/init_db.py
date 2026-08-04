"""
DB bootstrap for the `auth` module: create tables and the default admin.

Run standalone as a container startup step for the `api` image (see
entrypoint.sh).
"""

import logging
import os
import time

from sqlalchemy.exc import OperationalError

from . import auth, database, models
from Notifications.db_notification import models as notification_models  # noqa: F401  (registers Notification on the shared Base)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RETRIES = 10
RETRY_DELAY_SECONDS = 3


def wait_for_db():
    for attempt in range(1, RETRIES + 1):
        try:
            with database.engine.connect():
                return
        except OperationalError:
            logger.info("auth-db not ready yet (attempt %d/%d)", attempt, RETRIES)
            time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError("auth-db did not become ready in time")


def create_default_admin():
    email = os.getenv("DEFAULT_ADMIN_EMAIL")
    secret = os.getenv("DEFAULT_ADMIN_SECRET")

    if not email or not secret:
        logger.warning("DEFAULT_ADMIN_EMAIL or DEFAULT_ADMIN_SECRET not set!")
        return

    db = database.SessionLocal()
    try:
        if db.query(models.User).filter(models.User.email == email).first():
            logger.info("Default admin already exists.")
            return
        db.add(models.User(email=email, secret=auth.get_password_hash(secret)))
        db.commit()
        logger.info("Default admin created.")
    finally:
        db.close()


def bootstrap():
    wait_for_db()
    logger.info("Creating tables...")
    models.Base.metadata.create_all(bind=database.engine)
    create_default_admin()


if __name__ == "__main__":
    bootstrap()
