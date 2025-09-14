@echo off
setlocal EnableExtensions

rem Windows CMD runner mirroring run.ps1 behavior
rem - Generates docker-compose.yml
rem - Mounts only ./datas to /workspace/datas
rem - Builds and runs the container interactively (bash)
rem - Cleans up on exit

rem Switch to the directory of this script
cd /d "%~dp0"

set "CONTAINER_NAME=model_trainer"
set "COMPOSE_FILE=docker-compose.yml"

rem Ensure datas exists so the bind mount succeeds
if not exist "datas" mkdir "datas"

rem Detect Docker Compose v2 ("docker compose") or fallback to v1 ("docker-compose")
set "DC_CMD=docker compose"
call %DC_CMD% version >nul 2>&1
if errorlevel 1 (
  where docker-compose >nul 2>&1
  if errorlevel 1 (
    echo Docker Compose not found. Install Docker Desktop with Compose v2 or docker-compose v1.
    exit /b 1
  ) else (
    set "DC_CMD=docker-compose"
  )
)

echo Generating %COMPOSE_FILE%...
>"%COMPOSE_FILE%" echo services:
>>"%COMPOSE_FILE%" echo   %CONTAINER_NAME%:
>>"%COMPOSE_FILE%" echo     container_name: %CONTAINER_NAME%
>>"%COMPOSE_FILE%" echo     build:
>>"%COMPOSE_FILE%" echo       context: ./
>>"%COMPOSE_FILE%" echo       dockerfile: Dockerfile
>>"%COMPOSE_FILE%" echo     environment:
>>"%COMPOSE_FILE%" echo       - PYTHONDONTWRITEBYTECODE=1
>>"%COMPOSE_FILE%" echo       - PYTHONUNBUFFERED=1
>>"%COMPOSE_FILE%" echo       - NVIDIA_VISIBLE_DEVICES=all
>>"%COMPOSE_FILE%" echo     runtime: nvidia
>>"%COMPOSE_FILE%" echo     volumes:
>>"%COMPOSE_FILE%" echo       - "./datas:/workspace/datas"
>>"%COMPOSE_FILE%" echo     working_dir: /workspace
>>"%COMPOSE_FILE%" echo     command: bash
>>"%COMPOSE_FILE%" echo     stdin_open: true
>>"%COMPOSE_FILE%" echo     tty: true

echo Starting container via %DC_CMD% up --build...
call %DC_CMD% -f "%COMPOSE_FILE%" up --build

echo Stopping and cleaning up Docker container...
call %DC_CMD% -f "%COMPOSE_FILE%" down

if exist "%COMPOSE_FILE%" (
  del /f /q "%COMPOSE_FILE%" >nul 2>&1
  echo Removed generated %COMPOSE_FILE%
)

echo Pruning unused Docker images...
docker image prune -f >nul 2>&1

endlocal
exit /b 0
