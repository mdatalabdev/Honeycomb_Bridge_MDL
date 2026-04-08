from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import func
from .models import Notification, NotificationAction
from sqlalchemy import or_, desc, asc
from Notifications.db_notification.models import Notification


def get_notifications(
    db: Session,
    status: str = None,
    search: str = None,
    severity: str = None,
    start_time: int = None,   # epoch millis
    end_time: int = None,     # epoch millis
    limit: int = 50,
    offset: int = 0,
    sort_by: str = "edgex_created",
    order: str = "desc"
):
    query = db.query(Notification)

    # Filter: status
    if status:
        query = query.filter(Notification.status == status)

    #  Filter: severity
    if severity:
        query = query.filter(Notification.severity == severity)

    #  Filter: date range (edgex_created)
    if start_time:
        query = query.filter(Notification.edgex_created >= start_time)

    if end_time:
        query = query.filter(Notification.edgex_created <= end_time)

    #  Search (content, description, category)
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                Notification.content.ilike(search_term),
                Notification.description.ilike(search_term),
                Notification.category.ilike(search_term)
            )
        )

    #  TOTAL COUNT (before pagination)
    total = query.count()

    #  Sorting
    sort_column = getattr(Notification, sort_by, Notification.edgex_created)

    if order.lower() == "asc":
        query = query.order_by(asc(sort_column))
    else:
        query = query.order_by(desc(sort_column))

    #  Pagination
    results = query.limit(limit).offset(offset).all()

    return total, results



#  SINGLE INSERT (fallback / small use)
def insert_notification(db: Session, data: dict):
    stmt = insert(Notification).values(
        id=data["id"],
        category=data.get("category"),
        content=data.get("content"),
        description=data.get("description"),
        sender=data.get("sender"),
        severity=data.get("severity"),
        labels=data.get("labels") or [],  #  safe default
        edgex_created=data.get("created"),
        edgex_modified=data.get("modified"),
    ).on_conflict_do_nothing(index_elements=["id"])

    db.execute(stmt)


#  BULK INSERT (PRIMARY for fetcher)
def insert_notifications_bulk(db: Session, data_list: list[dict]):
    if not data_list:
        return

    stmt = insert(Notification).values([
        {
            "id": data["id"],
            "category": data.get("category"),
            "content": data.get("content"),
            "description": data.get("description"),
            "sender": data.get("sender"),
            "severity": data.get("severity"),
            "labels": data.get("labels") or [],
            "edgex_created": data.get("created"),
            "edgex_modified": data.get("modified"),
        }
        for data in data_list
    ]).on_conflict_do_nothing(index_elements=["id"])

    db.execute(stmt)


#  Get notifications with pagination
def get_notifications_by_status(
    db: Session,
    status: str,
    limit: int = 50,
    offset: int = 0
):
    return (
        db.query(Notification)
        .filter(Notification.status == status)
        .order_by(Notification.edgex_created.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )


#  Get latest timestamp (for incremental ingestion)
def get_last_notification_timestamp(db: Session):
    return db.query(func.max(Notification.edgex_created)).scalar()


#  Close notification with remark
def close_notification(db: Session, notification_id: str, remark: str, user: str):
    notif = db.query(Notification).filter(Notification.id == notification_id).first()

    if not notif:
        return None

    if not remark:
        raise ValueError("Remark is required")

    notif.status = "CLOSED"

    action = NotificationAction(
        notification_id=notification_id,
        action_type="CLOSED",
        remark=remark,
        performed_by=user
    )

    db.add(action)
    db.commit()  #  transactional boundary for user action

    return notif