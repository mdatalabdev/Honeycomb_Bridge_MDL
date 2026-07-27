.DEFAULT_GOAL := help

COMPOSE := docker compose
SERVICES := api docker-ops-sidecar redis auth-db timescaledb backup-worker notifications-worker iot-worker ml-service

# make logs SERVICE=api / make sh SERVICE=api / make restart SERVICE=api
SERVICE ?=

.PHONY: help build up down restart stop start ps logs logs-tail sh migrate revision test clean prune reset

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## Build (or rebuild) all images; SERVICE=<name> to build just one
	$(COMPOSE) build $(SERVICE)

up: ## Start the full stack in the background (requires magistrala-base-net to already exist)
	$(COMPOSE) up -d $(SERVICE)

down: ## Stop and remove containers (keeps volumes)
	$(COMPOSE) down

restart: ## Restart everything, or just SERVICE=<name>
	$(COMPOSE) restart $(SERVICE)

stop: ## Stop containers without removing them
	$(COMPOSE) stop $(SERVICE)

start: ## Start previously-stopped containers
	$(COMPOSE) start $(SERVICE)

ps: ## Show container status
	$(COMPOSE) ps

logs: ## Tail logs for everything, or just SERVICE=<name>
	$(COMPOSE) logs -n 200 -f $(SERVICE)

sh: ## Open a shell in SERVICE (default: api), e.g. make sh SERVICE=iot-worker
	$(COMPOSE) exec $(if $(SERVICE),$(SERVICE),api) sh

migrate: ## Run Alembic migrations against the host-installed auth Postgres (out of scope for containers, see CONTAINERIZATION.md)
	alembic upgrade head

revision: ## Create a new Alembic revision — usage: make revision m="add some_column"
	alembic revision -m "$(m)"

test: ## Run the test suite
	pytest -v

clean: ## Stop containers and remove volumes (auth-db-data, timescale_data) — DESTROYS DATA
	$(COMPOSE) down -v

prune: ## Remove dangling images/build cache left behind by repeated builds
	docker image prune -f

reset: down build up ## Rebuild images and restart the stack from scratch (keeps volumes)
