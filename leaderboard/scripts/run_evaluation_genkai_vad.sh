#!/bin/bash

# 单卡（例如超算上的单张 H100）多进程闭环评估脚本
# 思路：只申请 1 张 GPU，在同一块卡上起多个 CARLA + evaluator 进程，
#       通过不同的 PORT / TM_PORT 和拆分后的 routes 子 xml 来并行评估。

# ======================= 基本配置 =======================

# CARLA / TrafficManager 端口基准值（避免和别的作业冲突，必要时可以整体平移）
BASE_PORT=30000
BASE_TM_PORT=50000

# Bench2Drive 评估模式（固定为 True）
IS_BENCH2DRIVE=True

# routes 主文件（会被 split_xml.py 拆成多个子 xml）
BASE_ROUTES=leaderboard/data/bench2drive220

# 使用的 agent 与配置（正常分辨率版本）
# 如果你想用 fast / super-fast 版本，可以参考 run_evaluation_debug.sh 里的写法替换下两行
TEAM_AGENT=leaderboard/team_code/vad_b2d_agent.py
# 注意：这里的 YOUR_CKPT_PATH 需要你手动改成在超算上实际存放 ckpt 的路径前缀
TEAM_CONFIG=Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py+Bench2DriveZoo/ckpts/vad_b2d_base.pth

BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220
PLANNER_TYPE=traj
ALGO=vad

# 结果保存根目录（会在内部再按 route 名称等创建子目录）
SAVE_PATH=./eval_bench2drive220_${ALGO}_${PLANNER_TYPE}_genkai_single_gpu

# ======================= GPU 监控（可帮助你调 TASK_NUM） =======================

# 是否开启 GPU 监控（默认开启）。如不需要可以在外面 export ENABLE_GPU_MONITOR=0
ENABLE_GPU_MONITOR=${ENABLE_GPU_MONITOR:-1}
# 监控间隔（秒），默认 300 秒（5 分钟）
GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL:-300}
# 监控日志文件
GPU_MONITOR_LOG=${GPU_MONITOR_LOG:-gpu_monitor_genkai.log}

start_gpu_monitor() {
    if [ "$ENABLE_GPU_MONITOR" != "1" ]; then
        echo "GPU monitor disabled (ENABLE_GPU_MONITOR=${ENABLE_GPU_MONITOR})"
        return
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi 不可用，无法在脚本内监控 GPU。"
        return
    fi

    echo "启动 GPU 监控，间隔 ${GPU_MONITOR_INTERVAL}s，日志：${GPU_MONITOR_LOG}"
    echo "timestamp, index, name, util.gpu, util.mem, mem.used(MiB), mem.total(MiB), power.draw(W)" > "${GPU_MONITOR_LOG}"

    # 后台循环监控，直到主脚本结束
    (
        while true; do
            nvidia-smi \
              --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
              --format=csv,noheader >> "${GPU_MONITOR_LOG}"
            sleep "${GPU_MONITOR_INTERVAL}"
        done
    ) &

    GPU_MONITOR_PID=$!
}

stop_gpu_monitor() {
    if [ -n "${GPU_MONITOR_PID:-}" ]; then
        kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    fi
}

# ======================= routes 拆分 =======================

# 单卡建议的并行进程数（可以根据显存和稳定性调整，比如 2、4、6）
TASK_NUM=4

if [ ! -f "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag" ]; then
    echo -e "****************************\033[33m Attention \033[0m ****************************"
    echo -e "\033[33m Running split_xml.py (TASK_NUM=${TASK_NUM}) \033[0m"
    python tools/split_xml.py "$BASE_ROUTES" "$TASK_NUM" "$ALGO" "$PLANNER_TYPE"
    touch "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag"
    echo -e "\033[32m Splitting complete. Flag file created. \033[0m"
else
    echo -e "\033[32m Splitting already done. \033[0m"
fi

# ======================= 单卡多进程配置 =======================

echo -e "**************\033[36m Single-GPU multi-process eval on genkai \033[0m **************"

# 启动 GPU 监控（如果可用）
start_gpu_monitor

# 这里所有 GPU_RANK 都设为 0，表示所有进程共用同一张 GPU
# 如果你在作业脚本里设置了 CUDA_VISIBLE_DEVICES，通常这块卡在容器/作业内部就是 0
GPU_RANK_LIST=()
TASK_LIST=()
for ((i=0; i<${TASK_NUM}; i++)); do
    GPU_RANK_LIST+=("0")
    TASK_LIST+=("$i")
done

echo -e "\033[32m GPU_RANK_LIST: ${GPU_RANK_LIST[*]} \033[0m"
echo -e "\033[32m TASK_LIST: ${TASK_LIST[*]} \033[0m"
echo -e "***********************************************************************************"

length=${#GPU_RANK_LIST[@]}
for ((i=0; i<length; i++)); do
    PORT=$((BASE_PORT + i * 150))
    TM_PORT=$((BASE_TM_PORT + i * 150))
    ROUTES="${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.xml"
    CHECKPOINT_ENDPOINT="${ALGO}_b2d_${PLANNER_TYPE}/${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.json"
    GPU_RANK=${GPU_RANK_LIST[$i]}

    echo -e "\033[32m ALGO: $ALGO \033[0m"
    echo -e "\033[32m PLANNER_TYPE: $PLANNER_TYPE \033[0m"
    echo -e "\033[32m TASK_ID: $i \033[0m"
    echo -e "\033[32m PORT: $PORT \033[0m"
    echo -e "\033[32m TM_PORT: $TM_PORT \033[0m"
    echo -e "\033[32m ROUTES: $ROUTES \033[0m"
    echo -e "\033[32m CHECKPOINT_ENDPOINT: $CHECKPOINT_ENDPOINT \033[0m"
    echo -e "\033[32m GPU_RANK: $GPU_RANK \033[0m"
    echo -e "\033[32m bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK \033[0m"
    echo -e "***********************************************************************************"

    bash -e leaderboard/scripts/run_evaluation.sh \
        "$PORT" "$TM_PORT" "$IS_BENCH2DRIVE" \
        "$ROUTES" "$TEAM_AGENT" "$TEAM_CONFIG" \
        "$CHECKPOINT_ENDPOINT" "$SAVE_PATH" \
        "$PLANNER_TYPE" "$GPU_RANK" \
        2>&1 > "${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}_genkai_single_gpu.log" &

    # 给 CARLA 一点时间启动，避免一下子起太多导致崩溃
    sleep 5
done

wait

# 停止 GPU 监控
stop_gpu_monitor


