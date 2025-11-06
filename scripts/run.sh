#!/usr/bin/env bash
set -euo pipefail

APP="gateway.main:app"

# Defaults
CMD="start"
PORT="8000"

# Parse args:
#   run.sh                -> start 8000
#   run.sh 8001           -> start 8001
#   run.sh stop           -> stop 8000
#   run.sh stop 8001      -> stop 8001
#   run.sh start 8001     -> start 8001
if [[ $# -ge 1 ]]; then
  case "$1" in
    start|stop)
      CMD="$1"
      PORT="${2:-8000}"
      ;;
    *)
      # first arg is a port
      CMD="start"
      PORT="$1"
      ;;
  esac
fi

# Validate port is numeric if provided
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
  echo "❌ PORT must be an integer, got: $PORT" >&2
  exit 1
fi

stop_port() {
  echo "🛑 Stopping any running instance on port $PORT..."
  # Kill any process listening on that port
  PIDS=$(lsof -ti:"$PORT" || true)
  if [[ -n "${PIDS}" ]]; then
    echo "$PIDS" | xargs kill -9 || true
    echo "✅ Port $PORT cleared."
  else
    echo "ℹ️  Nothing running on $PORT."
  fi
}

start_port() {
  echo "🚀 Starting DFR-HCI Prototype on http://127.0.0.1:$PORT ..."
  # Clear the port first (handy during dev)
  lsof -ti:"$PORT" | xargs kill -9 2>/dev/null || true
  uvicorn "$APP" --reload --port "$PORT"
}

case "$CMD" in
  stop)
    stop_port
    ;;
  start)
    start_port
    ;;
  *)
    echo "Usage:"
    echo "  $0                # start on 8000"
    echo "  $0 8001           # start on 8001"
    echo "  $0 stop           # stop 8000"
    echo "  $0 stop 8001      # stop 8001"
    echo "  $0 start 8001     # start 8001"
    exit 1
    ;;
esac
