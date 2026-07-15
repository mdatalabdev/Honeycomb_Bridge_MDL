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
| 9 | `docker-ops-sidecar` | new, see below | `docker:25-cli` + python | only container that mounts `/var/run/docker.sock`; brokers the sibling `docker exec` calls on `api`'s behalf over an internal HTTP API |

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

2. **`docker exec` calls to sibling containers** (`api_downlink.py`) currently shell
   out to four containers by name: `chirpstack-chirpstack-1` (create API key, ~line
   1301), `edgex-security-proxy-setup` (create user, ~line 1244),
   `edgex-security-secretstore-setup` (read the Vault root-init token, ~line 1342),
   and `superset_app` (create user / change password, ~lines 1477 and 1620). This
   won't work from inside the `api` container without mounting
   `/var/run/docker.sock`, which grants host-level Docker control to anything that
   compromises that container.

   Decision: introduce `docker-ops-sidecar` (container #9 above) as the only thing
   that mounts the socket. It's a minimal container exposing one HTTP endpoint per
   operation above — not a generic exec passthrough — reachable only from `api` on
   `honeycomb-net`. `api` calls it over HTTP instead of shelling out directly, so a
   compromise of the internet-facing `api` container no longer implies host-level
   Docker control. Each endpoint is deleted once its underlying operation is
   migrated to a real API (ChirpStack gRPC `ApiKeyService.Create`, Vault HTTP for
   the EdgeX proxy user and for reading secrets instead of `cat`-ing the root-init
   file) — the sidecar is a bridge, not a permanent fixture.

   Two pre-existing issues worth fixing regardless of this migration: the Vault
   root-init token is read live via `docker exec cat` rather than coming from a
   secrets manager, and the Superset password-change endpoint passes the old/new
   password as plaintext argv into `python3 -c` inside the container (visible to
   anything that can read that container's process list).

3. **`localhost`-based URLs in `config.py`** need to become env-configurable service
   names/hostnames (`auth-db`, `timescaledb`, `redis`, plus the external hosts
   above) rather than assuming everything shares one host network.

4. **`auth/.env`** currently points `DATABASE_URL` at a host-installed Postgres
   (`localhost:5435/test_auth_db`). This becomes
   `postgresql://<user>:<password>@auth-db:5432/test_auth_db`, with the password
   sourced from a secret, not the `.env` file, in production.

Note: Alembic (`alembic/versions/*`) is intentionally **out of scope** — it keeps
running as-is (manual `alembic upgrade head` on the host), not as a container or
init step.

## Next Steps

- [ ] Write Dockerfiles for `api`, `iot-worker`, `notifications-worker`,
      `backup-worker`, `ml-service`, `docker-ops-sidecar`
- [ ] Write `docker-compose.yml` wiring all 9 containers + external network reference
- [ ] Move secrets out of `config.py` into env vars / secret store
- [ ] Point the 5 `docker exec` call sites in `api_downlink.py` at `docker-ops-sidecar`
      over HTTP instead of shelling out directly
