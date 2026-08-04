# Shared helper for the run scripts. Sourced, never executed.
#
# ROS 2's setup files are not safe to source under `set -u`. They read
# variables like AMENT_TRACE_SETUP_FILES and COLCON_TRACE without first
# checking that they exist, so the shell aborts before ROS is on the path:
#
#   /opt/ros/jazzy/setup.bash: line 8: AMENT_TRACE_SETUP_FILES: unbound variable
#
# `set -u` is worth keeping for the rest of each script, since it catches
# typo'd variable names. So it is disabled only around the sourcing and
# restored immediately afterwards.

# Usage: source_ros <distro> <workspace>
#
# Sources the ROS 2 installation, then the colcon workspace if it has been
# built. Missing workspace is not an error here: setup_workspace.sh has to
# source ROS before the workspace exists in order to build it.
source_ros () {
    local distro="$1"
    local workspace="$2"
    local had_u=""

    case "$-" in *u*) had_u=1 ;; esac
    set +u

    if [ ! -f "/opt/ros/${distro}/setup.bash" ]; then
        echo "ROS 2 ${distro} not found at /opt/ros/${distro}." >&2
        echo "Install it, or use the container in .devcontainer/." >&2
        [ -n "${had_u}" ] && set -u
        return 1
    fi

    # shellcheck disable=SC1090
    . "/opt/ros/${distro}/setup.bash"

    if [ -f "${workspace}/install/setup.bash" ]; then
        # shellcheck disable=SC1090
        . "${workspace}/install/setup.bash"
    fi

    [ -n "${had_u}" ] && set -u
    return 0
}

# Usage: require_workspace <workspace>
#
# Fails with an actionable message rather than letting `ros2 launch` report
# a missing package, which does not hint at the real cause.
require_workspace () {
    local workspace="$1"

    if [ ! -f "${workspace}/install/setup.bash" ]; then
        echo "No built workspace at ${workspace}." >&2
        echo "Run scripts/setup_workspace.sh first." >&2
        return 1
    fi
    return 0
}
