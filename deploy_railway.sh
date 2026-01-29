#!/usr/bin/env bash

# Simple helper script to deploy this app to Railway.
# Prerequisites:
#   - Railway CLI installed: https://docs.railway.app/develop/cli
#   - Logged in: `railway login`
#   - A Railway project created (the first run of `railway up` will prompt you)
#
# This script assumes:
#   - You have committed the code you want to deploy
#   - You have set environment variables in Railway's dashboard to match `.env`

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if ! command -v railway >/dev/null 2>&1; then
  echo "Railway CLI not found."
  echo "Install it from: https://docs.railway.app/develop/cli"
  exit 1
fi

echo "Deploying WhatsApp bot to Railway..."
echo
echo "If this is your first time, Railway will prompt you to create/link a project."
echo

# Deploy the current directory.
# Railway will detect Python + Flask and build with Nixpacks by default.
railway up

echo
echo "Deployment triggered. Visit the Railway dashboard to see build and logs."

