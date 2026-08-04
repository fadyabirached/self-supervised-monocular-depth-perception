#!/usr/bin/env bash
#
# One-time container setup: link this repo's packages into a colcon
# workspace, build them, and make ROS available in every new shell.
#
# scripts/setup_workspace.sh does the linking and building; this only adds
# the container-specific parts around it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Set up the shell first, so that a terminal is still usable even if the
# build below fails and this script exits early.
#
# The guards matter: ROS's setup files abort under `set -u`, which is on
# by default in some shells, and .bashrc is read before the workspace
# exists on the very first run.
BASHRC="${HOME}/.bashrc"
add_once () {
    grep -qxF "$1" "${BASHRC}" 2>/dev/null || echo "$1" >> "${BASHRC}"
}
add_once 'set +u'
add_once 'source /opt/ros/jazzy/setup.bash'
add_once '[ -f "$HOME/ros2_ws/install/setup.bash" ] && source "$HOME/ros2_ws/install/setup.bash"'
add_once 'export DISPLAY=:1'

echo "Building the colcon workspace..."
"${REPO_ROOT}/scripts/setup_workspace.sh"

cat <<'EOF'

Container ready.

  1. PORTS tab -> open port 6080 in a browser (append /vnc.html if needed)
  2. In a terminal:  scripts/run_depth_gz.sh
  3. To capture it:  scripts/record_sim.sh demo.gif

EOF
