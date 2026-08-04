#!/usr/bin/env bash
#
# One-time container setup: link this repo's packages into a colcon
# workspace, build them, and make ROS available in every new shell.
#
# scripts/setup_workspace.sh does the linking and building; this only adds
# the container-specific parts around it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building the colcon workspace..."
"${REPO_ROOT}/scripts/setup_workspace.sh"

# Source ROS and the workspace in every interactive shell, so a new
# terminal can run ros2 straight away.
BASHRC="${HOME}/.bashrc"
add_once () {
    grep -qxF "$1" "${BASHRC}" 2>/dev/null || echo "$1" >> "${BASHRC}"
}
add_once 'source /opt/ros/jazzy/setup.bash'
add_once '[ -f "$HOME/ros2_ws/install/setup.bash" ] && source "$HOME/ros2_ws/install/setup.bash"'
add_once 'export DISPLAY=:1'

cat <<'EOF'

Container ready.

  1. PORTS tab -> open port 6080 in a browser (append /vnc.html if needed)
  2. In a terminal:  scripts/run_depth_gz.sh
  3. To capture it:  scripts/record_sim.sh demo.gif

EOF
