#!/usr/bin/env bash
# Start the virtual desktop: Xvfb -> window manager -> VNC -> noVNC -> static
# server for the fixture web apps -> Firefox on the first app.
set -euo pipefail

Xvfb "${DISPLAY}" -screen 0 "${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}x24" -nolisten tcp &
sleep 1
fluxbox >/dev/null 2>&1 &
x11vnc -display "${DISPLAY}" -forever -shared -nopw -rfbport "${VNC_PORT}" -quiet &
websockify --web=/usr/share/novnc "${NOVNC_PORT}" "localhost:${VNC_PORT}" >/dev/null 2>&1 &
(cd /srv/webapps && python3 -m http.server 8080 --bind 0.0.0.0 >/dev/null 2>&1) &
sleep 1
firefox --kiosk "http://localhost:8080/dashboard.html#/login" >/dev/null 2>&1 &

echo "sandbox ready: display ${DISPLAY} ${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}, VNC :${VNC_PORT}, noVNC :${NOVNC_PORT}"
wait -n
