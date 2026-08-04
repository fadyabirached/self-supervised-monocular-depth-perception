#!/usr/bin/env bash
#
# Bring up the virtual desktop that Gazebo draws onto, and serve it over
# HTTP so it opens in a browser tab.
#
#   Xvfb        an X server with no monitor attached
#   openbox     a window manager, so Gazebo's windows can be moved/resized
#   x11vnc      exports that X display over the VNC protocol
#   websockify  serves noVNC's HTML client and tunnels VNC over WebSocket
#
# Idempotent: each piece is skipped if it is already running, so this can
# be re-run after a crash without stacking duplicate servers.

set -uo pipefail

DISPLAY_NUM="${DISPLAY_NUM:-1}"
SCREEN_SIZE="${SCREEN_SIZE:-1600x900x24}"
VNC_PORT="${VNC_PORT:-5900}"
WEB_PORT="${WEB_PORT:-6080}"

export DISPLAY=":${DISPLAY_NUM}"

running () { pgrep -f "$1" >/dev/null 2>&1; }

if ! running "Xvfb :${DISPLAY_NUM}"; then
    echo "Starting Xvfb on :${DISPLAY_NUM} (${SCREEN_SIZE})"
    Xvfb ":${DISPLAY_NUM}" -screen 0 "${SCREEN_SIZE}" -nolisten tcp >/tmp/xvfb.log 2>&1 &
    for _ in $(seq 1 30); do
        xdpyinfo -display ":${DISPLAY_NUM}" >/dev/null 2>&1 && break
        sleep 0.5
    done
fi

if ! running "openbox"; then
    echo "Starting openbox"
    openbox >/tmp/openbox.log 2>&1 &
fi

if ! running "x11vnc.*-rfbport ${VNC_PORT}"; then
    echo "Starting x11vnc on :${VNC_PORT}"
    # -nopw is safe here: the port is never published, it is reached only
    # through the editor's authenticated port forwarding.
    x11vnc -display ":${DISPLAY_NUM}" -forever -shared -nopw \
           -rfbport "${VNC_PORT}" -noxdamage >/tmp/x11vnc.log 2>&1 &
fi

if ! running "websockify.*${WEB_PORT}"; then
    echo "Starting noVNC on :${WEB_PORT}"
    websockify --web=/usr/share/novnc "${WEB_PORT}" "localhost:${VNC_PORT}" \
        >/tmp/novnc.log 2>&1 &
fi

cat <<EOF

Desktop is up.

  Open the forwarded port ${WEB_PORT} and go to /vnc.html
  (in Codespaces: PORTS tab -> port ${WEB_PORT} -> open in browser, then
  add /vnc.html to the URL if it does not load straight into the client).

Then, in a terminal:

  scripts/run_depth_gz.sh          # Gazebo + depth model + steering
  scripts/record_sim.sh demo.gif   # record 20 s of it as a GIF

EOF
