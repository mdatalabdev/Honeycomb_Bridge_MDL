import time
import logging
from Notifications.edgex_notification_fetcher import ingest_notifications

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