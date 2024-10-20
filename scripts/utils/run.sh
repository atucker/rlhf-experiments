#!/usr/bin/bash
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <default_script> [args...]"
    exit 1
fi

script_name="$1"
shift 1

if [ ! -f "$script_name" ]; then
    echo "Error: $script_name does not exist"
    exit 1
fi

# Read script
script_content=$(cat "$script_name")

# Extract arguments from script content
extract_arguments() {
    local line="$1"
    local args="${line#* }" # ignore the first command
    local args="${args#* }" # ignore the second command
    # match argument patterns: --argument, --argument value, --argument=value
    echo "$args" | grep -oE -- "--[_[:alnum:]-]+([=[:space:]]+[^-[:space:]]*)?"
}

# Get first element in the script
command_name=$(echo "$script_content" | awk '{print $1 " " $2}')
file_name=$(echo "$script_content" | awk '{print $NF}')
args=$(extract_arguments "$script_content")
# Put args on the same line
args=$(echo "$args" | tr '\n' ' ')

# Append given args to read args
args="$args $@"
echo "Running $command_name $args $file_name"
# $command_name $args $file_name