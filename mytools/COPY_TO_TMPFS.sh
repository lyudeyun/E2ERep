#!/bin/bash
# Copy Bench2Drive data from NFS/disk into tmpfs for faster I/O
# Requires root

set -e

# Resolve repo root from this script's location (mytools/ -> repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Loading Bench2Drive data into memory (tmpfs)"
echo "Repo root: $REPO_ROOT"
echo "=========================================="

# Require root
if [ "$EUID" -ne 0 ]; then 
    echo "Error: this script requires root privileges"
    echo "  Run: sudo bash $0"
    exit 1
fi

# Memory stats
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $7}')
echo "Total memory: ${TOTAL_MEM}GB"
echo "Available memory: ${AVAILABLE_MEM}GB"

# Expected dataset size (images + maps)
DATA_SIZE_GB=27  # ~27GB (16GB images + ~10.8GB maps)
echo "Dataset size: ${DATA_SIZE_GB}GB (including map files)"

# Need ~30GB tmpfs plus 5GB headroom
REQUIRED_MEM=35
if [ "$AVAILABLE_MEM" -lt "$REQUIRED_MEM" ]; then
    echo "Warning: available memory (${AVAILABLE_MEM}GB) < required (${REQUIRED_MEM}GB)"
    echo "  Need: 30GB tmpfs + 5GB headroom = ${REQUIRED_MEM}GB"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Tmpfs mount point
TMPFS_MOUNT="/mnt/bench2drive_ram"
# Budget ~27GB payload + margin → 40G tmpfs
TMPFS_SIZE="40G"

# Already mounted?
if mountpoint -q "$TMPFS_MOUNT" 2>/dev/null; then
    echo "Warning: $TMPFS_MOUNT is already mounted"
    read -p "Unmount and remount? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        umount "$TMPFS_MOUNT" 2>/dev/null || true
    else
        echo "Reusing the existing tmpfs mount"
    fi
fi

# Prepare mount point
mkdir -p "$TMPFS_MOUNT"

# Mount tmpfs
if ! mountpoint -q "$TMPFS_MOUNT" 2>/dev/null; then
    echo ""
    echo "Mounting tmpfs at $TMPFS_MOUNT (size: ${TMPFS_SIZE})..."
    mount -t tmpfs -o size=${TMPFS_SIZE} tmpfs "$TMPFS_MOUNT"
    echo "tmpfs mounted successfully"
else
    echo "tmpfs is already mounted"
fi

# Source: staged copy under /tmp or repo data tree
if [ -d "/tmp/bench2drive_local" ]; then
    # Local staging still uses bench2drive (not bench2drive_preprocessed)
    SOURCE_DATA="/tmp/bench2drive_local/bench2drive"
    SOURCE_INFOS="/tmp/bench2drive_local/infos"
    SOURCE_MAPS="$REPO_ROOT/Bench2DriveZoo/data/bench2drive/maps"
    SOURCE_MAP_FILE="$REPO_ROOT/Bench2DriveZoo/data/infos/b2d_map_infos.pkl"
    echo "Using local staged data as source (faster)"
else
    # Directory name is bench2drive (not bench2drive_preprocessed)
    SOURCE_DATA="$REPO_ROOT/Bench2DriveZoo/data/bench2drive"
    SOURCE_INFOS="$REPO_ROOT/Bench2DriveZoo/data/infos"
    SOURCE_MAPS="$REPO_ROOT/Bench2DriveZoo/data/bench2drive/maps"
    SOURCE_MAP_FILE="$REPO_ROOT/Bench2DriveZoo/data/infos/b2d_map_infos.pkl"
    echo "Copying from repo data tree (may be on NFS)"
fi

TARGET_DATA="$TMPFS_MOUNT/bench2drive"
TARGET_INFOS="$TMPFS_MOUNT/infos"
TARGET_MAPS="$TMPFS_MOUNT/maps"
TARGET_MAP_FILE="$TMPFS_MOUNT/b2d_map_infos.pkl"

# Destination folders on tmpfs
echo ""
echo "Creating destination directories..."
mkdir -p "$TARGET_DATA"
mkdir -p "$TARGET_INFOS"
mkdir -p "$TARGET_MAPS"

# Rsync images into tmpfs (fast if source is local SSD)
echo ""
echo "Copying data into memory (should be fast if source is a local SSD)..."
echo "  Source: $SOURCE_DATA"
echo "  Destination: $TARGET_DATA"

rsync -av --progress \
    --exclude='*.tmp' \
    --exclude='*.log' \
    "$SOURCE_DATA/" \
    "$TARGET_DATA/"

echo ""
echo "Copying infos directory..."
rsync -av --progress \
    "$SOURCE_INFOS/" \
    "$TARGET_INFOS/"

echo ""
echo "Copying map info file (map_infos.pkl)..."
# Map npz already lives under bench2drive; skip duplicating maps/ on tmpfs.
cp "$SOURCE_MAP_FILE" "$TARGET_MAP_FILE"

# Sanity check
echo ""
echo "Verifying copy..."
if [ -d "$TARGET_DATA" ] && [ "$(ls -A $TARGET_DATA)" ]; then
    echo "  Data directory copied successfully"
    TARGET_COUNT=$(find "$TARGET_DATA" -name "*.jpg" 2>/dev/null | wc -l)
    SOURCE_COUNT=$(find "$SOURCE_DATA" -name "*.jpg" 2>/dev/null | wc -l)
    echo "  Source image count: $SOURCE_COUNT"
    echo "  Destination image count: $TARGET_COUNT"
    
    if [ "$TARGET_COUNT" -eq "$SOURCE_COUNT" ]; then
        echo "  Image counts match"
    else
        echo "  Warning: image counts do not match"
    fi
else
    echo "  Error: data directory copy failed"
    exit 1
fi

# Post-copy instructions
echo ""
echo "=========================================="
echo "Data is now loaded into memory"
echo "=========================================="
echo ""
echo "Next step: update the config file"
echo ""
echo "Edit: Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py"
echo ""
echo "If your config still uses the old paths:"
echo "  data_root = \"data/bench2drive\""
echo "  info_root = \"data/infos\""
echo "  map_root = \"data/bench2drive/maps\""
echo "  map_file = \"data/infos/b2d_map_infos.pkl\""
echo ""
echo "Change them to point at the in-memory data:"
echo "  data_root = \"$TARGET_DATA\""
echo "  info_root = \"$TARGET_INFOS\""
echo "  map_root = \"$TARGET_MAPS\""
echo "  map_file = \"$TARGET_MAP_FILE\""
echo ""
echo "Then re-run the evaluation script."
echo ""
echo "Note: tmpfs is cleared after reboot; re-run this script if needed."
echo ""
