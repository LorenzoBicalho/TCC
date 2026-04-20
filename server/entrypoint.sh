#!/bin/bash
set -e
cd /app/db
alembic upgrade head
cd /app
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4