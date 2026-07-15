# Honeycomb Bridge — Containerization Architecture

This document describes the proposed container split for Honeycomb Bridge. It reflects
the current codebase structure (single-process `main.py` running multiple threads) and
how that gets decomposed into independently deployable containers.

Reverse proxy / TLS termination is **not** included here — that's handled by
infrastructure outside this stack.

## Container Map

| # | Container | Source | Image base | Notes |
|---|---|---|---|---|
| 1 | `api` | `api_downlink.py`, `auth/` | `python:3.10-slim` | FastAPI/uvicorn, stateless, scalable, listens on 4567 |
| 2 | `iot-worker` | `scheduler.py`, `event_fetcher_parse.py`, `User_token.py`, `key_rotation.py` | `python:3.10-slim` | device polling, JWT/key rotation, MQTT listener, gRPC to ChirpStack — pulled out of `main.py`'s threads |
| 3 | `notifications-worker` | `Notifications/worker.py` | `python:3.10-slim` | standalone polling loop, no changes needed |
| 4 | `backup-worker` | `Timescale_db/backup_scheduler.py` + friends | `python:3.10-slim` + `ssh`/`pg_dump` client tools | needs NAS SSH egress and reads from both `auth-db` and the Magistrala source DB |
| 5 | `ml-service` | `Predictive_ML/` | separate image, `python:3.10` + torch/xgboost/sklearn | kept off the shared base image (5-8GB+ due to torch/cuda); GPU passthrough only if training on-box |
| 6 | `auth-db` | — | `postgres:15` | replaces the host-installed Postgres currently at `localhost:5435` |
| 7 | `timescaledb` | existing `Timescale_db/docker-compose.yaml` | `timescale/timescaledb:2.14.2-pg15` | reused as-is, joins the shared network |
| 8 | `redis` | — | `redis` (official) | used by `captcha_utils.py` |

## Network

All containers above join one shared bridge network (e.g. `honeycomb-net`) so services
address each other by name — `auth-db`, `timescaledb`, `redis` — instead of `localhost`.
Only `api` and `iot-worker` need egress to the external ChirpStack/EdgeX stack.

## External Dependencies (not containerized in this stack)

Referenced via env-configured hostnames, never `localhost`, in production:

- **Magistrala DB** (`SOURCE_DB_HOST` in `auth/.env`) — backup source only
- **ChirpStack** — gRPC `:8088`, HTTP `:8090`
- **EdgeX Vault** — `:8200`
- **EdgeX notifications service** — `:59860`
- **Superset** — `:8018`

If these run in their own Compose stack, join this network as an `external: true`
network reference rather than duplicating them here.

## Required Changes Before/During Containerization

1. **Secrets currently hardcoded in `config.py`** (ChirpStack API token, AES/AES-GCM
   keys, SMTP app password, encrypted service creds) must move to environment
   variables or a secrets manager before building images. Since these are already
   committed to git history, treat them as compromised and rotate regardless.

2. **`docker exec` calls to sibling containers** (`api_downlink.py`, around lines
   1200–1360) currently shell out to `chirpstack-chirpstack-1`,
   `edgex-security-proxy-setup`, and `edgex-security-secretstore-setup` by container
   name. This won't work from inside the `api` container without mounting
   `/var/run/docker.sock`, which grants host-level Docker control to anything that
   compromises that container. Preferred fix: replace these calls with the
   equivalent APIs (ChirpStack gRPC, Vault HTTP) already used elsewhere in the
   codebase. If that's not feasible short term, confine socket access to one
   narrowly-scoped sidecar instead of the main API container.

3. **`localhost`-based URLs in `config.py`** need to become env-configurable service
   names/hostnames (`auth-db`, `timescaledb`, `redis`, plus the external hosts
   above) rather than assuming everything shares one host network.

4. **`auth/.env`** currently points `DATABASE_URL` at a host-installed Postgres
   (`localhost:5435/test_auth_db`). This becomes
   `postgresql://<user>:<password>@auth-db:5432/test_auth_db`, with the password
   sourced from a secret, not the `.env` file, in production.

5. **Alembic migrations** (`alembic/versions/*`) should run as a one-shot init step
   (init container or entrypoint check) before `api`/`iot-worker` start, rather than
   relying on someone running `alembic upgrade head` manually on the host.

## Next Steps

- [ ] Write Dockerfiles for `api`, `iot-worker`, `notifications-worker`,
      `backup-worker`, `ml-service`
- [ ] Write `docker-compose.yml` wiring all 8 containers + external network reference
- [ ] Move secrets out of `config.py` into env vars / secret store
- [ ] Refactor `docker exec` calls into API calls
- [ ] Add migration init step for `auth-db`
