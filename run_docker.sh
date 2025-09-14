#!/usr/bin/env bash
set -euo pipefail

# Portable Linux/macOS runner matching run.ps1 behavior.
# - Builds the image
# - Mounts only ./datas to /workspace/datas
# - Starts an interactive bash session
# - Cleans up the container and the generated compose file on exit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
CONTAINER_NAME="model_trainer"

# Ensure datas folder exists so volume mount succeeds
mkdir -p "${SCRIPT_DIR}/datas"

# Pick docker compose command (v2 or v1)
if docker compose version >/dev/null 2>&1; then
  DC=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  DC=(docker-compose)
else
  echo "Docker Compose not found. Install Docker Compose v2 (docker compose) or v1 (docker-compose)." >&2
  exit 1
fi

echo "Generating docker-compose.yml..."
cat >"${COMPOSE_FILE}" <<'YAML'
services:
  model_trainer:
    container_name: model_trainer
    build:
      context: ./
      dockerfile: Dockerfile
    environment:
      - PYTHONDONTWRITEBYTECODE=1
      - PYTHONUNBUFFERED=1
      - NVIDIA_VISIBLE_DEVICES=all
    runtime: nvidia
    volumes:
      - "./datas:/workspace/datas"
    working_dir: /workspace
    command: bash
    stdin_open: true
    tty: true
YAML

cleanup() {
  echo "Stopping and cleaning up Docker container..."
  "${DC[@]}" -f "${COMPOSE_FILE}" down || true
  if [[ -f "${COMPOSE_FILE}" ]]; then
    rm -f "${COMPOSE_FILE}"
    echo "Removed generated docker-compose.yml"
  fi
  echo "Pruning unused Docker images..."
  docker image prune -f >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Starting container via docker compose up --build..."
"${DC[@]}" -f "${COMPOSE_FILE}" up --build

