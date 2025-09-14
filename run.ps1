# ----- Docker information -----
$ContainerName = "model_trainer"
$ContainerWorkspacePath = "/workspace"

$HostDataPath = "./datas"
$ContainerDataPath = "/workspace/datas"

# ----- Docker Compose file path -----
$ComposeFile = "./docker-compose.yml"

Write-Host "Generating docker-compose.yml..."
Set-Content -Path $ComposeFile -Value @"
services:
  ${ContainerName}:
    container_name: ${ContainerName}
    build:
      context: ./
      dockerfile: Dockerfile
    environment:
      - PYTHONDONTWRITEBYTECODE=1
      - PYTHONUNBUFFERED=1
      - NVIDIA_VISIBLE_DEVICES=all
    runtime: nvidia
    volumes:
      - "${HostDataPath}:${ContainerDataPath}"
    working_dir: ${ContainerWorkspacePath}
    command: bash
    stdin_open: true
    tty: true
"@

# ----- Start the Docker container ----- 
try {
    Write-Host "Starting container via docker-compose up --build..."
    docker compose up --build 
}
finally {
    Write-Host "Stopping and cleaning up Docker container..."
    docker compose down

    if (Test-Path $ComposeFile) {
        Remove-Item $ComposeFile -Force
        Write-Host "Removed generated docker-compose.yml"
    }

    Write-Host "Pruning unused Docker images..."
    docker image prune -f
}
