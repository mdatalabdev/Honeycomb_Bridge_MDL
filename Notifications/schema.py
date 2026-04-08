from pydantic import BaseModel
from typing import List, Optional


class NotificationResponse(BaseModel):
    id: str
    category: Optional[str]
    content: Optional[str]
    description: Optional[str]
    sender: Optional[str]
    severity: Optional[str]
    labels: Optional[List[str]]
    status: str

    class Config:
        from_attributes = True


class CloseNotificationRequest(BaseModel):
    remark: str
    user: str