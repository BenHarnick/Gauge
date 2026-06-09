#!/bin/sh
# Ensure the /data volume mount point exists and is writable by appuser.
# This script runs as root so it can fix ownership after the volume mounts
# (which replaces any directory created at image-build time).
mkdir -p /data
chown appuser:appuser /data

exec gosu appuser uvicorn gauge.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1
