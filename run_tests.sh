#!/usr/bin/env bash
# Every check for the memory system, in one command. Deliberately stdlib unittest:
# neither image has pytest, and a suite that needs an install first is a suite nobody
# runs. No model calls at all — these pin storage, filtering, provenance and the
# agent-facing tools, which is where the SILENT failures live.
#
#   ./run_tests.sh          all of it
#   ./run_tests.sh -v       verbose
#   ./run_tests.sh -k date  just the date tests
#
# Unit tests always run. Integration tests skip cleanly when the memory service is
# unreachable, and the bot-tool tests skip when amul-oan-api is not checked out.
set -uo pipefail
cd "$(dirname "$0")"
export MEMORY_API_URL="${MEMORY_API_URL:-http://localhost:8100}"
export MEMORY_INTERNAL_TOKEN="${MEMORY_INTERNAL_TOKEN:-offline-test-token-32-characters-minimum}"
export QDRANT_HTTP="${QDRANT_HTTP:-http://localhost:6350}"
export AMUL_OAN_API_PATH="${AMUL_OAN_API_PATH:-/amulpfsdata/gautam/amul-oan-api}"
echo "memory service: $MEMORY_API_URL"
python3 -m unittest discover -s tests "$@" 2>&1 \
  | grep -vE "DeprecationWarning|on_event is deprecated|FastAPI docs for Lifespan|Read more about it|@app.on_event|return self.router.on_event|^[[:space:]]*$"
