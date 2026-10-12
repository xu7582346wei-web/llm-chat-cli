#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 "$@"
