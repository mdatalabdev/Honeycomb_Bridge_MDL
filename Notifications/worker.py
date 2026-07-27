import os
import time
import logging
from Notifications.edgex_notification_fetcher import ingest_notifications

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_notification_worker(interval: int = 5):
    """
    Runs ingestion in a loop every `interval` seconds
    """
    logger.info(f"Notification worker started (interval={interval}s)")

    while True:
        try:
            ingest_notifications()
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)

        time.sleep(interval)


if __name__ == "__main__":
    # Container entrypoint for `notifications-worker` (see CONTAINERIZATION.md
    # container #3) — run as `python3 -m Notifications.worker` from the repo root.
    run_notification_worker(int(os.environ.get("NOTIFICATION_POLL_INTERVAL", "5")))