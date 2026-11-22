#!/bin/sh
# One-shot init: wait for Piston, install the 6 required packages (skipping
# any already installed), then verify all 7 required language/version
# runtimes are actually present before exiting 0.
#
# Package install/list/uninstall endpoints (POST/GET/DELETE /api/v2/packages)
# and the runtimes endpoint (GET /api/v2/runtimes) are Piston's own documented
# REST API (piston.readthedocs.io/en/latest/api-v2/), so this talks to Piston
# directly over HTTP rather than shelling out to the Node.js CLI.
set -eu

PISTON_URL="${PISTON_URL:-http://piston:2000}"
MAX_WAIT="${MAX_WAIT:-300}"

# Piston package name:version -> installed via POST /api/v2/packages.
# Mapping (backend language -> Piston package) is the one the project's own
# README and T4b live test already documented and exercised successfully.
PACKAGES="python:3.12.0 node:18.15.0 typescript:5.0.3 java:15.0.2 mono:6.12.0 gcc:10.2.0"

# Backend-facing language:version pairs -> verified via GET /api/v2/runtimes.
# "c" and "c++" are two separate runtimes both provided by the single "gcc"
# package installed above.
RUNTIMES="python:3.12.0 javascript:18.15.0 typescript:5.0.3 java:15.0.2 csharp:6.12.0 c:10.2.0 c++:10.2.0"

echo "Waiting for Piston API at $PISTON_URL ..."
waited=0
until curl -sf "$PISTON_URL/api/v2/runtimes" >/dev/null 2>&1; do
  waited=$((waited + 5))
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    echo "ERROR: Piston did not become ready within ${MAX_WAIT}s" >&2
    exit 1
  fi
  sleep 5
done
echo "Piston API is up."

echo
echo "Checking already-installed packages (GET /api/v2/packages) ..."
installed_json="$(curl -sf "$PISTON_URL/api/v2/packages")"

is_installed() {
  lang="$1"; ver="$2"
  echo "$installed_json" | jq -e \
    --arg l "$lang" --arg v "$ver" \
    'any(.[]?; .language == $l and .language_version == $v and .installed == true)' \
    >/dev/null 2>&1
}

for pair in $PACKAGES; do
  lang="${pair%%:*}"
  ver="${pair#*:}"
  if is_installed "$lang" "$ver"; then
    echo "  $lang $ver already installed, skipping"
    continue
  fi
  echo "  installing $lang $ver ..."
  http_code=$(curl -s -o /tmp/resp.json -w '%{http_code}' -X POST "$PISTON_URL/api/v2/packages" \
    -H "Content-Type: application/json" \
    -d "{\"language\":\"$lang\",\"version\":\"$ver\"}")
  body="$(cat /tmp/resp.json)"
  if [ "$http_code" != "200" ]; then
    echo "  FAILED to install $lang $ver (HTTP $http_code): $body" >&2
    exit 1
  fi
  echo "  installed: $body"
done

echo
echo "Verifying installed runtimes (GET /api/v2/runtimes) ..."
runtimes_json="$(curl -sf "$PISTON_URL/api/v2/runtimes")"
echo "Raw response:"
echo "$runtimes_json" | jq . 2>/dev/null || echo "$runtimes_json"

missing=""
for pair in $RUNTIMES; do
  lang="${pair%%:*}"
  ver="${pair#*:}"
  found="false"
  if echo "$runtimes_json" | jq -e --arg l "$lang" --arg v "$ver" \
      'any(.[]?; .language == $l and .version == $v)' >/dev/null 2>&1; then
    found="true"
  fi
  if [ "$found" != "true" ]; then
    missing="$missing $lang@$ver"
  fi
done

if [ -n "$missing" ]; then
  echo >&2
  echo "ERROR: these required runtimes were not found in /api/v2/runtimes:$missing" >&2
  echo "Check the raw response above for the actual 'language'/'version' field names Piston used -- this script assumes they match the backend's own language keys and versions exactly, which has not been empirically confirmed in this environment." >&2
  exit 1
fi

echo
echo "All 7 required runtimes verified present."
exit 0
