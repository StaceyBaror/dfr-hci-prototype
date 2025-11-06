#!/usr/bin/env bash
pkill -f "uvicorn .*gateway.main:app" 2>/dev/null || true
uvicorn gateway.main:app --reload --port "${PORT:-8000}"
