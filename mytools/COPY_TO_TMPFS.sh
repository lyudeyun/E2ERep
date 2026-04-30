#!/bin/bash
# Copy Bench2Drive data from NFS/disk into tmpfs for faster I/O
# Requires root

set -e

echo "=========================================="
echo "将Bench2Drive数据加载到内存（tmpfs）"
echo "=========================================="

# Require root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ 此脚本需要root权限"
    echo "   请使用: sudo bash $0"
    exit 1
fi

# Memory stats
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $7}')
echo "✅ 总内存: ${TOTAL_MEM}GB"
echo "✅ 可用内存: ${AVAILABLE_MEM}GB"

# Expected dataset size (images + maps)
DATA_SIZE_GB=27  # ~27GB (16GB images + ~10.8GB maps)
echo "📊 数据大小: ${DATA_SIZE_GB}GB (包括地图文件)"

# Need ~30GB tmpfs plus 5GB headroom
REQUIRED_MEM=35
if [ "$AVAILABLE_MEM" -lt "$REQUIRED_MEM" ]; then
    echo "⚠️  警告: 可用内存(${AVAILABLE_MEM}GB) < 所需内存(${REQUIRED_MEM}GB)"
    echo "   需要: 30GB tmpfs + 5GB缓冲 = ${REQUIRED_MEM}GB"
    read -p "是否继续? (y/n) " -n 1 -r
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
    echo "⚠️  $TMPFS_MOUNT 已经挂载"
    read -p "是否卸载并重新挂载? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        umount "$TMPFS_MOUNT" 2>/dev/null || true
    else
        echo "使用已挂载的tmpfs"
    fi
fi

# Prepare mount point
mkdir -p "$TMPFS_MOUNT"

# Mount tmpfs
if ! mountpoint -q "$TMPFS_MOUNT" 2>/dev/null; then
    echo ""
    echo "📁 挂载tmpfs到 $TMPFS_MOUNT (大小: ${TMPFS_SIZE})..."
    mount -t tmpfs -o size=${TMPFS_SIZE} tmpfs "$TMPFS_MOUNT"
    echo "✅ tmpfs挂载成功"
else
    echo "✅ tmpfs已挂载"
fi

# Source: staged copy under /tmp or repo data tree
if [ -d "/tmp/bench2drive_local" ]; then
    # Local staging still uses bench2drive (not bench2drive_preprocessed)
    SOURCE_DATA="/tmp/bench2drive_local/bench2drive"
    SOURCE_INFOS="/tmp/bench2drive_local/infos"
    SOURCE_MAPS="/home/deyun/git/B2DRepair/Bench2DriveZoo/data/bench2drive/maps"
    SOURCE_MAP_FILE="/home/deyun/git/B2DRepair/Bench2DriveZoo/data/infos/b2d_map_infos.pkl"
    echo "✅ 使用已复制的本地数据作为源（更快）"
else
    # Directory name is bench2drive (not bench2drive_preprocessed)
    SOURCE_DATA="/home/deyun/git/B2DRepair/Bench2DriveZoo/data/bench2drive"
    SOURCE_INFOS="/home/deyun/git/B2DRepair/Bench2DriveZoo/data/infos"
    SOURCE_MAPS="/home/deyun/git/B2DRepair/Bench2DriveZoo/data/bench2drive/maps"
    SOURCE_MAP_FILE="/home/deyun/git/B2DRepair/Bench2DriveZoo/data/infos/b2d_map_infos.pkl"
    echo "⚠️  从NFS直接复制（较慢）"
fi

TARGET_DATA="$TMPFS_MOUNT/bench2drive"
TARGET_INFOS="$TMPFS_MOUNT/infos"
TARGET_MAPS="$TMPFS_MOUNT/maps"
TARGET_MAP_FILE="$TMPFS_MOUNT/b2d_map_infos.pkl"

# Destination folders on tmpfs
echo ""
echo "📁 创建目标目录..."
mkdir -p "$TARGET_DATA"
mkdir -p "$TARGET_INFOS"
mkdir -p "$TARGET_MAPS"

# Rsync images into tmpfs (fast if source is local SSD)
echo ""
echo "📦 开始复制数据到内存（这应该很快，因为从本地SSD复制）..."
echo "   源: $SOURCE_DATA"
echo "   目标: $TARGET_DATA"

rsync -av --progress \
    --exclude='*.tmp' \
    --exclude='*.log' \
    "$SOURCE_DATA/" \
    "$TARGET_DATA/"

echo ""
echo "📦 复制infos目录..."
rsync -av --progress \
    "$SOURCE_INFOS/" \
    "$TARGET_INFOS/"

echo ""
echo "📦 复制地图信息文件（map_infos.pkl）..."
# Map npz already lives under bench2drive; skip duplicating maps/ on tmpfs.
cp "$SOURCE_MAP_FILE" "$TARGET_MAP_FILE"

# Sanity check
echo ""
echo "✅ 验证复制结果..."
if [ -d "$TARGET_DATA" ] && [ "$(ls -A $TARGET_DATA)" ]; then
    echo "   ✅ 数据目录复制成功"
    TARGET_COUNT=$(find "$TARGET_DATA" -name "*.jpg" 2>/dev/null | wc -l)
    SOURCE_COUNT=$(find "$SOURCE_DATA" -name "*.jpg" 2>/dev/null | wc -l)
    echo "   源图像数: $SOURCE_COUNT"
    echo "   目标图像数: $TARGET_COUNT"
    
    if [ "$TARGET_COUNT" -eq "$SOURCE_COUNT" ]; then
        echo "   ✅ 图像数量匹配"
    else
        echo "   ⚠️  图像数量不匹配"
    fi
else
    echo "   ❌ 数据目录复制失败"
    exit 1
fi

# Post-copy instructions
echo ""
echo "=========================================="
echo "✅ 数据已加载到内存！"
echo "=========================================="
echo ""
echo "📝 下一步：修改配置文件"
echo ""
echo "编辑: Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py"
echo ""
echo "将以下行（如果你的 config 里还是旧路径）："
echo "  data_root = \"data/bench2drive\""
echo "  info_root = \"data/infos\""
echo "  map_root = \"data/bench2drive/maps\""
echo "  map_file = \"data/infos/b2d_map_infos.pkl\""
echo ""
echo "改为（指向内存中的数据）："
echo "  data_root = \"$TARGET_DATA\""
echo "  info_root = \"$TARGET_INFOS\""
echo "  map_root = \"$TARGET_MAPS\""
echo "  map_file = \"$TARGET_MAP_FILE\""
echo ""
echo "然后重新运行评估脚本。"
echo ""
echo "⚠️  注意: 重启后tmpfs会被清空，需要重新运行此脚本"
echo ""
