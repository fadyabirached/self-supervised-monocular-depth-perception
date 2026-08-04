#!/usr/bin/env bash
#
# Record the running simulation off the X display and write a GIF (or an
# mp4), so the demo can go straight into the README.
#
# Meant for the container in .devcontainer/, where Gazebo draws onto the
# virtual display :1 and there is no screen-recording software to reach
# for. It works on any Linux desktop with ffmpeg and an X display.
#
# Usage:
#   scripts/record_sim.sh                  # 20 s -> demo.gif
#   scripts/record_sim.sh clip.gif 30      # 30 s -> clip.gif
#   scripts/record_sim.sh clip.mp4 15      # 15 s -> clip.mp4, no GIF pass
#
# Start the simulation first, let it settle, then run this.
#
# Env vars:
#   DISPLAY   X display to capture (default: :1)
#   FPS       capture frame rate (default: 10, which is plenty for a GIF
#             of a simulation running under software rendering)
#   WIDTH     GIF width in pixels, height follows the aspect ratio
#             (default: 800, keeps a 20 s clip in the low single-digit MB)

set -euo pipefail

OUTPUT="${1:-demo.gif}"
DURATION="${2:-20}"
DISPLAY="${DISPLAY:-:1}"
FPS="${FPS:-10}"
WIDTH="${WIDTH:-800}"

command -v ffmpeg >/dev/null 2>&1 || {
    echo "ffmpeg not found. apt-get install -y ffmpeg" >&2
    exit 1
}

# Ask X for the real screen size rather than guessing, so the capture is
# not letterboxed or cropped.
if command -v xdpyinfo >/dev/null 2>&1; then
    GEOMETRY="$(xdpyinfo -display "${DISPLAY}" | awk '/dimensions:/ {print $2; exit}')"
else
    GEOMETRY="${GEOMETRY:-1600x900}"
fi

echo "Recording ${DURATION}s of ${DISPLAY} at ${GEOMETRY}, ${FPS} fps -> ${OUTPUT}"

if [[ "${OUTPUT}" == *.gif ]]; then
    WORK="$(mktemp -d)"
    trap 'rm -rf "${WORK}"' EXIT

    ffmpeg -hide_banner -loglevel error -y \
        -f x11grab -framerate "${FPS}" -video_size "${GEOMETRY}" \
        -i "${DISPLAY}" -t "${DURATION}" \
        -c:v libx264 -preset veryfast -pix_fmt yuv420p "${WORK}/raw.mp4"

    # Two passes: build a palette from the whole clip, then apply it.
    # A single-pass GIF is limited to a generic 256-colour palette and
    # looks visibly banded on Gazebo's flat-shaded surfaces.
    ffmpeg -hide_banner -loglevel error -y -i "${WORK}/raw.mp4" \
        -vf "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos,palettegen=stats_mode=diff" \
        "${WORK}/palette.png"

    ffmpeg -hide_banner -loglevel error -y \
        -i "${WORK}/raw.mp4" -i "${WORK}/palette.png" \
        -lavfi "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=3" \
        "${OUTPUT}"
else
    ffmpeg -hide_banner -loglevel error -y \
        -f x11grab -framerate "${FPS}" -video_size "${GEOMETRY}" \
        -i "${DISPLAY}" -t "${DURATION}" \
        -c:v libx264 -preset veryfast -pix_fmt yuv420p "${OUTPUT}"
fi

echo "Wrote ${OUTPUT} ($(du -h "${OUTPUT}" | cut -f1))"
echo
echo "GitHub renders GIFs inline in a README. Drag it into an issue or"
echo "comment to get a user-attachments URL, or commit it and reference"
echo "the file path."
