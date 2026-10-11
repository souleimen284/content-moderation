#!/bin/sh
# Writes a crontab entry from RETRAIN_SCHEDULE_CRON (default: weekly, Monday
# 03:00 — see .env.example) and runs cron in the foreground so the container
# stays alive. Every retrain.py invocation inherits the container's
# environment (DATABASE_URL, model paths, etc.) via `env` in the crontab line,
# since cron does not otherwise pass through the container's environment.
set -e

SCHEDULE="${RETRAIN_SCHEDULE_CRON:-0 3 * * 1}"
LOG_FILE="/app/logs/retrain_cron.log"

mkdir -p /app/logs
touch "$LOG_FILE"

# Dump the current container environment so the cron job can source it —
# cron jobs run with a minimal environment otherwise.
printenv | grep -Ev '^(HOME|PWD|SHLVL|_)=' > /app/cron.env

echo "$SCHEDULE cd /app && . /app/cron.env && export \$(cut -d= -f1 /app/cron.env) && python retrain.py >> $LOG_FILE 2>&1" > /etc/cron.d/retrain-cron
chmod 0644 /etc/cron.d/retrain-cron
crontab /etc/cron.d/retrain-cron

echo "Retrain scheduler starting with cron schedule: $SCHEDULE"
echo "Logs: $LOG_FILE"

# Run once immediately on startup so a freshly deployed system doesn't wait
# a full week for its first model version, then hand off to cron.
if [ "${RETRAIN_ON_STARTUP:-true}" = "true" ]; then
    echo "Running an initial retrain pass on startup..."
    python retrain.py >> "$LOG_FILE" 2>&1 || echo "Initial retrain pass failed; see $LOG_FILE"
fi

cron -f
