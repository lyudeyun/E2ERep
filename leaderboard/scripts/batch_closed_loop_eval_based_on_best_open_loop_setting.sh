#!/bin/bash
# 批量闭环评估脚本 - 基于最佳开环配置
# 用于评估指定配置的多个重复实验的修复后模型的闭环性能
# 通常用于评估开环评估中表现最好的配置的所有重复实验
#
# 使用方法:
#   bash leaderboard/scripts/batch_closed_loop_eval_based_on_best_open_loop_setting.sh \
#     <EXPERIMENT_PATH> \
#     [--fast] [--super-fast] [--gpu-rank GPU] [--port-base PORT] [--routes ROUTES]
#
# 示例:
#   # 方式1: 指定一个特定的实验文件夹
#   bash leaderboard/scripts/batch_closed_loop_eval_based_on_best_open_loop_setting.sh \
#     vad_base_Arachne_v2_DE_results/VAD_base_REP_VAL_t3s_Arachne_v2_DE_w26_p52_i50_es5_CONT_1 \
#     --fast
#
#   # 方式2: 指定一个父目录，自动查找所有包含.pth文件的子目录
#   bash leaderboard/scripts/batch_closed_loop_eval_based_on_best_open_loop_setting.sh \
#     vad_base_Arachne_v2_DE_results \
#     --fast
#
#   # 方式3: 使用通配符模式（向后兼容）
#   bash leaderboard/scripts/batch_closed_loop_eval_based_on_best_open_loop_setting.sh \
#     "vad_base_Arachne_v2_DE_results/VAD_base_REP_VAL_t3s_Arachne_v2_DE_w26_p52_i50_es5_CONT_*" \
#     --fast
#
# 参数说明:
#   EXPERIMENT_PATH: 实验路径，可以是：
#     - 单个实验文件夹路径（例如: vad_base_Arachne_v2_DE_results/VAD_base_REP_VAL_..._CONT_1）
#     - 父目录路径（例如: vad_base_Arachne_v2_DE_results），会自动查找所有包含.pth的子目录
#     - 通配符模式（例如: "vad_base_*_DE_results/VAD_base_REP_VAL_*_CONT_*"）
#   注意: 结果会保存在每个实验目录下的 closed_loop_eval/ 文件夹中（与 open_loop_eval/ 并列）
#   --fast: 使用 fast 版本（1280x720）
#   --super-fast: 使用 super-fast 版本（640x360）
#   --gpu-rank: GPU 设备ID（默认: 0）
#   --port-base: 端口起始值（默认: 30000，每个任务递增150）
#   --routes: routes XML 文件路径（默认: leaderboard/data/drivetransformer_bench2drive_dev10.xml）
#   --instance-id: 实例ID（用于并行运行，0-based，默认: 0）
#   --total-instances: 总实例数（用于并行运行，默认: 1，即串行运行）

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash leaderboard/scripts/batch_closed_loop_eval_based_on_best_open_loop_setting.sh \
    <EXPERIMENT_PATH> \
    [OPTIONS]

Arguments:
  EXPERIMENT_PATH  实验路径，可以是：
                    - 单个实验文件夹路径
                    - 父目录路径（自动查找所有包含.pth的子目录）
                    - 通配符模式（例如: "vad_base_*_DE_results/VAD_base_REP_VAL_*_CONT_*"）
                   注意: 结果会保存在每个实验目录下的 closed_loop_eval/ 文件夹中

Options:
  --fast           使用 fast 版本（1280x720）
  --super-fast     使用 super-fast 版本（640x360）
  --gpu-rank GPU   GPU 设备ID（默认: 0）
  --port-base PORT 端口起始值（默认: 30000，每个任务递增150）
  --routes ROUTES  routes XML 文件路径（默认: leaderboard/data/drivetransformer_bench2drive_dev10.xml）
  --instance-id ID  实例ID（用于并行运行，0-based，默认: 0）
  --total-instances N 总实例数（用于并行运行，默认: 1，即串行运行）

Examples:
  # 方式1: 指定单个实验文件夹
  bash leaderboard/scripts/batch_closed_loop_eval_based_on_best_open_loop_setting.sh \
    vad_base_Arachne_v2_DE_results/VAD_base_REP_VAL_t3s_Arachne_v2_DE_w26_p52_i50_es5_CONT_1 \
    --fast

  # 方式2: 指定父目录，自动查找所有实验（串行运行）
  bash leaderboard/scripts/batch_closed_loop_eval_based_on_best_open_loop_setting.sh \
    vad_base_Arachne_v2_DE_results \
    --fast

  # 方式3: 并行运行（N个实例，每个实例处理一部分实验）
  # 示例：3个实例并行（在3个不同的终端/tmux会话中分别运行）
  bash leaderboard/scripts/batch_closed_loop_eval_based_on_best_open_loop_setting.sh \
    vad_base_Arachne_v2_DE_results \
    --fast --instance-id 0 --total-instances 3
  
  bash leaderboard/scripts/batch_closed_loop_eval_based_on_best_open_loop_setting.sh \
    vad_base_Arachne_v2_DE_results \
    --fast --instance-id 1 --total-instances 3
  
  bash leaderboard/scripts/batch_closed_loop_eval_based_on_best_open_loop_setting.sh \
    vad_base_Arachne_v2_DE_results \
    --fast --instance-id 2 --total-instances 3
  
  # 超算环境示例：10个实例并行（使用SLURM作业数组）
  # sbatch --array=0-9 eval_job.sh
  # 其中 eval_job.sh 包含：
  #   bash leaderboard/scripts/batch_closed_loop_eval_based_on_best_open_loop_setting.sh \
  #     vad_base_Arachne_v2_DE_results \
  #     --fast --instance-id $SLURM_ARRAY_TASK_ID --total-instances 10
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ $# -lt 1 ]; then
  echo "ERROR: Expected at least 1 argument, got $#." 1>&2
  usage
  exit 2
fi

EXPERIMENT_PATH="$1"
shift 1

# 默认参数
USE_FAST_VERSION=0
USE_SUPER_FAST_VERSION=0
GPU_RANK=0
PORT_BASE=30000
ROUTES="leaderboard/data/drivetransformer_bench2drive_dev10.xml"
IS_BENCH2DRIVE="True"
PLANNER_TYPE="only_traj"
INSTANCE_ID=0
TOTAL_INSTANCES=1

# 解析可选参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --fast)
      USE_FAST_VERSION=1
      shift
      ;;
    --super-fast)
      USE_SUPER_FAST_VERSION=1
      shift
      ;;
    --gpu-rank)
      GPU_RANK="$2"
      shift 2
      ;;
    --port-base)
      PORT_BASE="$2"
      shift 2
      ;;
    --routes)
      ROUTES="$2"
      shift 2
      ;;
    --instance-id)
      INSTANCE_ID="$2"
      shift 2
      ;;
    --total-instances)
      TOTAL_INSTANCES="$2"
      shift 2
      ;;
    *)
      echo "ERROR: Unknown option: $1" 1>&2
      usage
      exit 2
      ;;
  esac
done

# 验证并行参数
if [ "${INSTANCE_ID}" -lt 0 ] || [ "${INSTANCE_ID}" -ge "${TOTAL_INSTANCES}" ]; then
  echo "ERROR: instance-id must be in range [0, total-instances-1]" 1>&2
  echo "  instance-id: ${INSTANCE_ID}" 1>&2
  echo "  total-instances: ${TOTAL_INSTANCES}" 1>&2
  exit 2
fi

# 检查互斥选项
if [ "${USE_FAST_VERSION}" = "1" ] && [ "${USE_SUPER_FAST_VERSION}" = "1" ]; then
  echo "ERROR: --fast and --super-fast are mutually exclusive" 1>&2
  exit 2
fi

# 确定 REPO_ROOT（需要在检查 CARLA_ROOT 之前定义）
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 检查 CARLA_ROOT，如果未设置，尝试使用默认路径
if [ -z "${CARLA_ROOT:-}" ]; then
  # 尝试使用默认路径：Bench2DriveZoo/carla
  DEFAULT_CARLA_ROOT="${REPO_ROOT}/Bench2DriveZoo/carla"
  if [ -f "${DEFAULT_CARLA_ROOT}/CarlaUE4.sh" ] && [ -d "${DEFAULT_CARLA_ROOT}/PythonAPI/carla/agents" ]; then
    export CARLA_ROOT="${DEFAULT_CARLA_ROOT}"
    echo "INFO: CARLA_ROOT not set, using default: ${CARLA_ROOT}"
  else
    echo "ERROR: CARLA_ROOT is not set and default path not found." 1>&2
    echo "" 1>&2
    echo "Please set CARLA_ROOT environment variable:" 1>&2
    echo "  export CARLA_ROOT=/path/to/carla" 1>&2
    echo "" 1>&2
    echo "Or ensure CARLA is installed at: ${DEFAULT_CARLA_ROOT}" 1>&2
    exit 1
  fi
fi

# 设置环境变量
export CARLA_SERVER="${CARLA_ROOT}/CarlaUE4.sh"
export PYTHONPATH=$PYTHONPATH:"${CARLA_ROOT}/PythonAPI"
export PYTHONPATH=$PYTHONPATH:"${CARLA_ROOT}/PythonAPI/carla"
export PYTHONPATH=$PYTHONPATH:"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export SCENARIO_RUNNER_ROOT=scenario_runner
export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS

# 确定使用的 agent 和 config（REPO_ROOT 已在前面定义）
if [ "${USE_SUPER_FAST_VERSION}" = "1" ]; then
  VERSION_NAME="super_fast"
  TEAM_AGENT="${REPO_ROOT}/Bench2DriveZoo/team_code/vad_b2d_agent_super_fast.py"
  BASE_CONFIG="${REPO_ROOT}/Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d_super_fast.py"
elif [ "${USE_FAST_VERSION}" = "1" ]; then
  VERSION_NAME="fast"
  TEAM_AGENT="${REPO_ROOT}/Bench2DriveZoo/team_code/vad_b2d_agent_fast.py"
  BASE_CONFIG="${REPO_ROOT}/Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d_fast.py"
else
  VERSION_NAME="normal"
  TEAM_AGENT="${REPO_ROOT}/Bench2DriveZoo/team_code/vad_b2d_agent.py"
  BASE_CONFIG="${REPO_ROOT}/Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py"
fi

# 检查文件是否存在
if [ ! -f "${TEAM_AGENT}" ]; then
  echo "ERROR: Team agent not found: ${TEAM_AGENT}" 1>&2
  exit 1
fi
if [ ! -f "${BASE_CONFIG}" ]; then
  echo "ERROR: Config file not found: ${BASE_CONFIG}" 1>&2
  exit 1
fi

# 输出目录会在每个实验目录下创建 closed_loop_eval/ 文件夹

# 解析实验路径并查找所有匹配的实验目录
echo "================================================================================"
echo "查找匹配的实验目录..."
echo "输入路径: ${EXPERIMENT_PATH}"
echo "================================================================================"

MATCHED_EXPERIMENTS=()

# 处理绝对路径或相对路径
if [[ "${EXPERIMENT_PATH}" = /* ]]; then
  # 绝对路径
  input_path="${EXPERIMENT_PATH}"
else
  # 相对路径，基于 REPO_ROOT
  input_path="${REPO_ROOT}/${EXPERIMENT_PATH}"
fi

# 情况1: 如果输入路径是一个文件（.pth文件），直接使用其所在目录
if [ -f "${input_path}" ] && [[ "${input_path}" == *.pth ]]; then
  exp_dir=$(dirname "$(dirname "$(dirname "${input_path}")")")
  pth_file="${input_path}"
  if [ -f "${pth_file}" ]; then
    MATCHED_EXPERIMENTS+=("${exp_dir}")
    echo "找到单个实验: ${exp_dir}"
  fi
# 情况2: 如果输入路径是一个目录，检查是否是单个实验目录
elif [ -d "${input_path}" ]; then
  # 检查是否是单个实验目录（包含 repair/repair_output/VAD_repaired_both_layers.pth）
  pth_file="${input_path}/repair/repair_output/VAD_repaired_both_layers.pth"
  if [ -f "${pth_file}" ]; then
    # 单个实验目录
    MATCHED_EXPERIMENTS+=("${input_path}")
    echo "找到单个实验目录: ${input_path}"
  else
    # 父目录，查找所有包含.pth文件的子目录
    echo "在父目录中查找所有包含.pth文件的实验..."
    while IFS= read -r -d '' exp_dir; do
      pth_file="${exp_dir}/repair/repair_output/VAD_repaired_both_layers.pth"
      if [ -f "${pth_file}" ]; then
        MATCHED_EXPERIMENTS+=("${exp_dir}")
      fi
    done < <(find "${input_path}" -type d -name "VAD_base_REP_VAL_*" -print0 2>/dev/null)
    
    if [ ${#MATCHED_EXPERIMENTS[@]} -eq 0 ]; then
      echo "在 ${input_path} 中未找到任何包含.pth文件的实验目录"
    fi
  fi
# 情况3: 如果输入路径包含通配符，使用 find 查找
elif [[ "${EXPERIMENT_PATH}" == *"*"* ]]; then
  # 提取目录部分和模式部分
  if [[ "${EXPERIMENT_PATH}" == *"/"* ]]; then
    # 包含路径分隔符
    dir_part=$(dirname "${EXPERIMENT_PATH}")
    pattern_part=$(basename "${EXPERIMENT_PATH}")
    
    if [[ "${dir_part}" = /* ]]; then
      search_base="${dir_part}"
    else
      search_base="${REPO_ROOT}/${dir_part}"
    fi
  else
    # 只有模式，在默认搜索目录中查找
    pattern_part="${EXPERIMENT_PATH}"
    search_base="${REPO_ROOT}"
  fi
  
  echo "使用通配符模式查找: ${pattern_part}"
  echo "搜索目录: ${search_base}"
  
  if [ ! -d "${search_base}" ]; then
    echo "ERROR: 搜索目录不存在: ${search_base}" 1>&2
    exit 1
  fi
  
  while IFS= read -r -d '' exp_dir; do
    pth_file="${exp_dir}/repair/repair_output/VAD_repaired_both_layers.pth"
    if [ -f "${pth_file}" ]; then
      MATCHED_EXPERIMENTS+=("${exp_dir}")
    fi
  done < <(find "${search_base}" -type d -name "${pattern_part}" -print0 2>/dev/null)
else
  echo "ERROR: 无效的输入路径: ${EXPERIMENT_PATH}" 1>&2
  echo "  请提供: 1) 单个实验目录路径" 1>&2
  echo "         2) 父目录路径（自动查找所有实验）" 1>&2
  echo "         3) 通配符模式（例如: vad_base_*_DE_results/VAD_base_REP_VAL_*_CONT_*）" 1>&2
  exit 1
fi

if [ ${#MATCHED_EXPERIMENTS[@]} -eq 0 ]; then
  echo "ERROR: 未找到任何匹配的实验" 1>&2
  echo "  输入路径: ${EXPERIMENT_PATH}" 1>&2
  echo "  请确保路径正确且包含修复后的.pth文件" 1>&2
  exit 1
fi

# 按名称排序
IFS=$'\n' MATCHED_EXPERIMENTS=($(sort <<<"${MATCHED_EXPERIMENTS[*]}"))
unset IFS

TOTAL_EXPERIMENTS=${#MATCHED_EXPERIMENTS[@]}
echo "找到 ${TOTAL_EXPERIMENTS} 个匹配的实验"

# 并行分配：根据 instance-id 和 total-instances 分配实验
if [ "${TOTAL_INSTANCES}" -gt 1 ]; then
  echo "并行模式: 实例 ${INSTANCE_ID}/${TOTAL_INSTANCES}"
  
  # 计算每个实例应该处理的实验索引
  ASSIGNED_EXPERIMENTS=()
  for i in "${!MATCHED_EXPERIMENTS[@]}"; do
    # 使用模运算分配：实验 i 分配给实例 (i % total_instances)
    assigned_instance=$((i % TOTAL_INSTANCES))
    if [ "${assigned_instance}" -eq "${INSTANCE_ID}" ]; then
      ASSIGNED_EXPERIMENTS+=("${MATCHED_EXPERIMENTS[$i]}")
    fi
  done
  
  MATCHED_EXPERIMENTS=("${ASSIGNED_EXPERIMENTS[@]}")
  echo "实例 ${INSTANCE_ID} 将处理 ${#MATCHED_EXPERIMENTS[@]} 个实验（共 ${TOTAL_EXPERIMENTS} 个）"
else
  echo "串行模式: 将处理所有 ${TOTAL_EXPERIMENTS} 个实验"
fi

echo ""
echo "本实例将评估的实验:"
for exp_dir in "${MATCHED_EXPERIMENTS[@]}"; do
  echo "  - ${exp_dir}"
done
echo ""

# 准备评估任务
TOTAL_TASKS=${#MATCHED_EXPERIMENTS[@]}
# 每个实例使用不同的端口范围，避免冲突
# 端口分配策略：
#   - 每个实例内的任务之间间隔150
#   - 不同实例的起始端口间隔100（确保不重叠）
#   - 实例0: 30000, 30150, 30300, 30450, ...
#   - 实例1: 30100, 30250, 30400, 30550, ...
#   - 实例2: 30200, 30350, 30500, 30650, ...
#   - 实例N: (30000 + N*100), (30000 + N*100 + 150), ...
# 注意：理论上支持任意数量的并行实例，但受限于：
#   - 可用端口范围（通常65535是上限）
#   - GPU内存（如果多个实例共享GPU）
#   - 系统资源（CPU、内存等）
INSTANCE_PORT_OFFSET=$((INSTANCE_ID * 100))  # 每个实例偏移100，避免重叠
CURRENT_PORT=$((PORT_BASE + INSTANCE_PORT_OFFSET))
CURRENT_TM_PORT=$((PORT_BASE + 20000 + INSTANCE_PORT_OFFSET))
PORT_INCREMENT=150  # 每个任务递增150

# 检查端口是否超出合理范围（警告，但不阻止）
MAX_REASONABLE_PORT=$((PORT_BASE + INSTANCE_PORT_OFFSET + (TOTAL_TASKS - 1) * PORT_INCREMENT))
if [ "${MAX_REASONABLE_PORT}" -gt 60000 ]; then
  echo "WARNING: 端口范围可能过大 (最大端口: ${MAX_REASONABLE_PORT})" 1>&2
  echo "  如果遇到端口冲突，请考虑减少并行实例数或增加端口间隔" 1>&2
fi

  echo "================================================================================"
echo "开始批量评估 (${VERSION_NAME} 版本)"
echo "================================================================================"
if [ "${TOTAL_INSTANCES}" -gt 1 ]; then
  echo "并行模式: 实例 ${INSTANCE_ID}/${TOTAL_INSTANCES}"
  echo "本实例任务数: ${TOTAL_TASKS} (总实验数: ${TOTAL_EXPERIMENTS})"
else
  echo "串行模式"
  echo "总任务数: ${TOTAL_TASKS}"
fi
echo "结果将保存在每个实验目录下的 closed_loop_eval/ 文件夹中"
echo "端口范围: ${CURRENT_PORT} (TM: ${CURRENT_TM_PORT}), 每个任务递增 ${PORT_INCREMENT}"
echo ""

SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_EXPERIMENTS=()

for i in "${!MATCHED_EXPERIMENTS[@]}"; do
  exp_dir="${MATCHED_EXPERIMENTS[$i]}"
  exp_name=$(basename "${exp_dir}")
  pth_file="${exp_dir}/repair/repair_output/VAD_repaired_both_layers.pth"
  
  task_num=$((i + 1))
  echo "[${task_num}/${TOTAL_TASKS}] 评估: ${exp_name}"
  echo "  PTH文件: ${pth_file}"
  
  # 构建输出路径：在每个实验目录下创建 closed_loop_eval/ 文件夹（与 open_loop_eval/ 并列）
  cl_dir="${exp_dir}/closed_loop_eval"
  checkpoint_json="${cl_dir}/closed_loop_eval.json"
  save_path="${cl_dir}"
  log_file="${cl_dir}/closed_loop_eval.log"
  
  mkdir -p "${cl_dir}"
  
  # 构建 team_config (config_path+model_path)
  team_config="${BASE_CONFIG}+${pth_file}"
  
  # 设置端口
  port=${CURRENT_PORT}
  tm_port=${CURRENT_TM_PORT}
  
  echo "  端口: ${port}, TM端口: ${tm_port}"
  echo "  输出: ${checkpoint_json}"
  
  # 调用评估脚本
  if bash "${REPO_ROOT}/leaderboard/scripts/run_closed_loop_eval.sh" \
    "${port}" \
    "${tm_port}" \
    "${IS_BENCH2DRIVE}" \
    "${ROUTES}" \
    "${TEAM_AGENT}" \
    "${team_config}" \
    "${checkpoint_json}" \
    "${save_path}" \
    "${PLANNER_TYPE}" \
    "${GPU_RANK}" \
    > "${log_file}" 2>&1; then
    echo "  ✓ 成功 (结果保存在: ${cl_dir})"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
    echo "  ✗ 失败 (查看日志: ${log_file})"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    FAILED_EXPERIMENTS+=("${exp_name}")
  fi
  
  # 递增端口
  CURRENT_PORT=$((CURRENT_PORT + PORT_INCREMENT))
  CURRENT_TM_PORT=$((CURRENT_TM_PORT + PORT_INCREMENT))
  
  echo ""
done

# 输出总结
echo "================================================================================"
echo "评估完成"
echo "================================================================================"
echo "成功: ${SUCCESS_COUNT}/${TOTAL_TASKS}"
echo "失败: ${FAILED_COUNT}/${TOTAL_TASKS}"

if [ ${FAILED_COUNT} -gt 0 ]; then
  echo ""
  echo "失败的实验:"
  for failed_exp in "${FAILED_EXPERIMENTS[@]}"; do
    echo "  - ${failed_exp}"
  done
fi

echo ""
echo "所有结果已保存在各个实验目录下的 closed_loop_eval/ 文件夹中"

