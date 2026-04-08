# Notifications Service (EdgeX Integration)

## Overview

This service ingests notifications from EdgeX, stores them in PostgreSQL, and exposes APIs to manage and query them.

### Key Features

* Fetch notifications from EdgeX (polling)
* Store in PostgreSQL
* Idempotent ingestion (no duplicates)
* NEW → CLOSED workflow
* Mandatory remark on close
* Full query APIs (filter, search, pagination, sorting)
* Background worker for near real-time ingestion

---

## Architecture

```
EdgeX API
   ↓
Background Worker (polling)
   ↓
PostgreSQL (notifications tables)
   ↓
FastAPI APIs
   ↓
Frontend
```

---

## Database Design

### notifications

Stores current state

Fields:

* id (PK)
* category
* content
* description
* sender
* severity
* labels (JSON)
* edgex_created (epoch millis)
* edgex_modified
* status (NEW / CLOSED)
* created_at
* updated_at

---

### notification_actions

Stores audit history

Fields:

* id (PK)
* notification_id (FK)
* action_type (CLOSED)
* remark
* performed_by
* performed_at

---

## Workflow

### Ingestion

1. Worker fetches notifications from EdgeX
2. Filters only new notifications (timestamp-based)
3. Inserts using bulk insert
4. Avoids duplicates using ON CONFLICT DO NOTHING

---

### Notification Lifecycle

```
NEW → CLOSED
```

* All notifications start as NEW
* User must provide remark to close
* Action logged in notification_actions

---

## Background Worker

Runs every 5 seconds via FastAPI startup event.

Responsibilities:

* Fetch notifications
* Filter new ones
* Insert into DB

---

## APIs

### 1. Get Notifications

GET /downlink/notifications

Query params:

* status
* severity
* search
* start_time
* end_time
* limit
* offset
* sort_by
* order

Response:

```
{
  "status": "success",
  "total": 100,
  "count": 20,
  "data": [...]
}
```

---

### 2. Get Notification by ID

GET /downlink/notifications/{id}

---

### 3. Close Notification

POST /downlink/notifications/{id}/close

Body:

```
{
  "remark": "Checked and resolved"
}
```

---

### 4. Get NEW Notifications

GET /downlink/notifications/new

---

### 5. Get CLOSED Notifications

GET /downlink/notifications/closed

---

### 6. Stats

GET /downlink/notifications/stats

---

## Performance Optimizations

* Bulk insert instead of row-by-row
* Timestamp filtering
* Pagination
* Indexed columns:

```
status
severity
edgex_created
```

---

## Important Notes

### 1. Model Import Requirement

Ensure notification models are imported before:

```
Base.metadata.create_all()
```

---

### 2. Worker Execution

Worker runs via FastAPI startup event.

Do NOT start it separately in main.

---

### 3. Token Handling

JWT token must be returned from token generator.

---

## Future Improvements

* WebSocket real-time updates
* Kafka / MQTT streaming ingestion
* Alert deduplication
* Acknowledgement state (ACK)
* Role-based access

---

## Summary

This system provides a scalable, near real-time notification pipeline with:

* Reliable ingestion
* Clean workflow management
* Full query capabilities
* Audit tracking

Ready for production with further scaling enhancements.
