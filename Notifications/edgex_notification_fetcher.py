from User_token import admin_JWT_token_generator
import logging
import requests
from auth.database import SessionLocal

from Notifications.db_notification.crud import (
    insert_notifications_bulk,
    get_last_notification_timestamp
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

notification_url = "https://rapid.meridiandatalabs.com/support-notifications/api/v3/notification/status/PROCESSED"


def fetch_notifications(token, limit=500, offset=0):
    """
    Fetch notifications from EdgeX
    """
    try:
        # print(f"printing the token {token}")
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }

        params = {
            'limit': limit,
            'offset': offset
        }

        response = requests.get(
            notification_url,
            headers=headers,
            params=params,
            timeout=10
        )
        response.raise_for_status()

        data = response.json()
        notifications = data.get("notifications", [])

        # logger.info(f"Fetched {len(notifications)} notifications (offset={offset})")

        return notifications

    except Exception as e:
        logger.error(f"Error fetching notifications: {e}", exc_info=True)
        return []


def ingest_notifications():
    """
    Fetch + store notifications in DB (optimized + near real-time)
    """
    db = SessionLocal()

    try:
        token = admin_JWT_token_generator()

        offset = 0
        limit = 500
        total_processed = 0

        #  get last processed timestamp
        last_ts = get_last_notification_timestamp(db) or 0

        while True:
            notifications = fetch_notifications(token, limit=limit, offset=offset)

            if not notifications:
                logger.info("No more notifications to fetch. Stopping.")
                break

            #  filter only new notifications
            new_notifications = [
                n for n in notifications
                if n.get("created", 0) > last_ts
            ]

            if new_notifications:
                try:
                    insert_notifications_bulk(db, new_notifications)
                    db.commit()
                except Exception as e:
                    logger.error(f"DB error, rolling back batch: {e}", exc_info=True)
                    db.rollback()
                    break

            batch_count = len(notifications)
            total_processed += len(new_notifications)
            offset += limit

            # logger.info(
            #     f"Batch fetched: {batch_count}, Inserted: {len(new_notifications)}, Total inserted: {total_processed}"
            # )

            #  exit when last page reached
            if batch_count < limit:
                logger.info("Last page reached.")
                break

        logger.info(f"Ingestion completed. Total inserted: {total_processed}")

    finally:
        db.close()