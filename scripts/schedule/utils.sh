# util scripts.

log_info() {
    # Use $* to merge all positional arguments
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $*"
}