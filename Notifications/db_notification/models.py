from sqlalchemy import Column, String, Text, BigInteger, TIMESTAMP, func, JSON, Integer, ForeignKey
from auth.database import Base


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(String, primary_key=True, index=True)

    category = Column(String)
    content = Column(Text)
    description = Column(Text)
    sender = Column(String)
    severity = Column(String)

    labels = Column(JSON)

    edgex_created = Column(BigInteger)
    edgex_modified = Column(BigInteger)

    status = Column(String, default="NEW", index=True)

    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())


class NotificationAction(Base):
    __tablename__ = "notification_actions"

    id = Column(Integer, primary_key=True, index=True)

    notification_id = Column(String, ForeignKey("notifications.id"))

    action_type = Column(String)  # CLOSED
    remark = Column(Text)
    performed_by = Column(String)

    performed_at = Column(TIMESTAMP, server_default=func.now())