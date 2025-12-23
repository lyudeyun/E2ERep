#!/bin/bash
# 将数据从NFS复制到tmpfs（内存），以获得最快的数据加载速度
# 需要root权限

set -e

echo "=========================================="
echo "将Bench2Drive数据加载到内存（tmpfs）"
echo "=========================================="

# 检查root权限
if [ "$EUID" -ne 0 ]; then 
    echo "❌ 此脚本需要root权限"
    echo "   请使用: sudo bash $0"
    exit 1
fi

# 检查内存
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $7}')
echo "✅ 总内存: ${TOTAL_MEM}GB"
echo "✅ 可用内存: ${AVAILABLE_MEM}GB"

# 数据大小（包括地图文件）
DATA_SIZE_GB=27  # 约27GB（16GB图像数据 + 10.8GB地图文件）
echo "📊 数据大小: ${DATA_SIZE_GB}GB (包括地图文件)"

# 检查内存是否足够（30GB tmpfs + 5GB缓冲）
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

# tmpfs挂载点
TMPFS_MOUNT="/mnt/bench2drive_ram"
# 总数据大小：16GB图像 + 6.3GB infos + 4.9GB地图 = 约27GB，加上缓冲设为40GB
TMPFS_SIZE="40G"

# 检查是否已经挂载
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

# 创建挂载点
mkdir -p "$TMPFS_MOUNT"

# 挂载tmpfs
if ! mountpoint -q "$TMPFS_MOUNT" 2>/dev/null; then
    echo ""
    echo "📁 挂载tmpfs到 $TMPFS_MOUNT (大小: ${TMPFS_SIZE})..."
    mount -t tmpfs -o size=${TMPFS_SIZE} tmpfs "$TMPFS_MOUNT"
    echo "✅ tmpfs挂载成功"
else
    echo "✅ tmpfs已挂载"
fi

# 数据源（从已复制的本地数据，或直接从NFS）
if [ -d "/tmp/bench2drive_local" ]; then
    # 本地缓存版本，同样使用 bench2drive 而不是 bench2drive_preprocessed
    SOURCE_DATA="/tmp/bench2drive_local/bench2drive"
    SOURCE_INFOS="/tmp/bench2drive_local/infos"
    SOURCE_MAPS="/home/deyun/git/B2DRepair/Bench2DriveZoo/data/bench2drive/maps"
    SOURCE_MAP_FILE="/home/deyun/git/B2DRepair/Bench2DriveZoo/data/infos/b2d_map_infos.pkl"
    echo "✅ 使用已复制的本地数据作为源（更快）"
else
    # 注意：这里的目录名是 bench2drive（而不是 bench2drive_preprocessed）
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

# 创建目标目录
echo ""
echo "📁 创建目标目录..."
mkdir -p "$TARGET_DATA"
mkdir -p "$TARGET_INFOS"
mkdir -p "$TARGET_MAPS"

# 复制数据到tmpfs（从本地SSD复制会很快）
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
# 注意：地图 npz 文件已经包含在 bench2drive 目录中，这里不再单独复制 maps 目录，
# 避免在 tmpfs 中保存两份地图数据。
cp "$SOURCE_MAP_FILE" "$TARGET_MAP_FILE"

# 验证
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

# 显示下一步
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
