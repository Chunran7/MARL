#!/bin/sh
env="mujoco"
scenario="Ant-v2"
agent_conf="2x4"
agent_obsk=1
algo="mappo_lagr"
exp="rnn"
seed_max=1
seed_=50

# 异步训练设置 (可以修改这里来切换异步模式)
use_async=true                    # 是否使用异步训练 (true/false)
async_mode="progressive"          # 异步策略: hierarchical(分层)/importance(重要性采样)/progressive(渐进式)
selection_ratio=0.7               # 智能体选择比例 (0.1-1.0)
num_groups=4                      # 分层模式的组数
importance_window=10              # 重要性计算窗口大小

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
if [ "$use_async" = true ]; then
    echo "使用异步训练模式: ${async_mode}, 选择比例: ${selection_ratio}"
    algo="async_mappo_lagr"
    exp="${exp}_async_${async_mode}"
else
    echo "使用原始训练模式"
fi

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    
    # 构建基础命令参数 - 使用优化后的拉格朗日参数
    base_args="--env_name ${env} --experiment_name ${exp} --scenario ${scenario} --agent_conf ${agent_conf} --agent_obsk ${agent_obsk} --lr 9e-5 --critic_lr 5e-3 --std_x_coef 1 --std_y_coef 5e-1 --seed ${seed_} --n_training_threads 4 --n_rollout_threads 16 --num_mini_batch 40 --episode_length 1000 --num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --add_center_xy --use_state_agent --safety_bound 0.2 --lamda_lagr 0.1 --lagrangian_coef_rate 1e-4"
    
    if [ "$use_async" = true ]; then
        # 异步训练命令 - 使用统一脚本，传递所有异步参数
        CUDA_VISIBLE_DEVICES=0 python train/train_mujoco_unified.py ${base_args} --use_async --async_mode ${async_mode} --async_ratio ${selection_ratio} --num_groups ${num_groups} --importance_window ${importance_window}
    else
        # 同步训练命令 - 使用统一脚本
        CUDA_VISIBLE_DEVICES=0 python train/train_mujoco_unified.py ${base_args}
    fi
done
