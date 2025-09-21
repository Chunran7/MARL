#!/bin/bash
# 异步MAPPO-Lagrangian训练脚本 (Linux系统)
# 使用方法: chmod +x train_mujoco_async.sh && ./train_mujoco_async.sh

# 基础环境配置
env="mujoco"
scenario="Ant-v2"
agent_conf="2x4"
agent_obsk=1
algo="rmappo_lagr"
exp="async_training"
seed_max=1
seed_=50

# 异步训练配置
async_mode="hierarchical"  # 可选: hierarchical, importance, progressive
selection_ratio=0.5
hierarchy_levels=3

echo "=========================================="
echo "异步MAPPO-Lagrangian训练开始"
echo "环境: ${env}, 场景: ${scenario}, 算法: ${algo}"
echo "异步模式: ${async_mode}, 选择比例: ${selection_ratio}"
echo "最大种子数: ${seed_max}"
echo "=========================================="

for seed in `seq ${seed_max}`;
do
    echo "运行种子 ${seed}:"
    
    # 分层异步训练
    if [ "$async_mode" = "hierarchical" ]; then
        echo "  使用分层异步更新策略"
        CUDA_VISIBLE_DEVICES=0 python train_mujoco_async.py \
            --env_name ${env} \
            --algorithm_name ${algo} \
            --experiment_name "${exp}_${async_mode}" \
            --scenario ${scenario} \
            --agent_conf ${agent_conf} \
            --agent_obsk ${agent_obsk} \
            --lr 9e-5 \
            --critic_lr 5e-3 \
            --std_x_coef 1 \
            --std_y_coef 5e-1 \
            --seed ${seed_} \
            --n_training_threads 4 \
            --n_rollout_threads 16 \
            --num_mini_batch 40 \
            --episode_length 1000 \
            --num_env_steps 10000000 \
            --ppo_epoch 5 \
            --use_value_active_masks \
            --add_center_xy \
            --use_state_agent \
            --safety_bound 0.2 \
            --lamda_lagr 0.78 \
            --lagrangian_coef_rate 1e-7 \
            --async_mode ${async_mode} \
            --selection_ratio ${selection_ratio} \
            --hierarchy_levels ${hierarchy_levels} \
            --use_async_value_update
    
    # 重要性采样异步训练
    elif [ "$async_mode" = "importance" ]; then
        echo "  使用重要性采样异步更新策略"
        CUDA_VISIBLE_DEVICES=0 python train_mujoco_async.py \
            --env_name ${env} \
            --algorithm_name ${algo} \
            --experiment_name "${exp}_${async_mode}" \
            --scenario ${scenario} \
            --agent_conf ${agent_conf} \
            --agent_obsk ${agent_obsk} \
            --lr 9e-5 \
            --critic_lr 5e-3 \
            --std_x_coef 1 \
            --std_y_coef 5e-1 \
            --seed ${seed_} \
            --n_training_threads 4 \
            --n_rollout_threads 16 \
            --num_mini_batch 40 \
            --episode_length 1000 \
            --num_env_steps 10000000 \
            --ppo_epoch 5 \
            --use_value_active_masks \
            --add_center_xy \
            --use_state_agent \
            --safety_bound 0.2 \
            --lamda_lagr 0.78 \
            --lagrangian_coef_rate 1e-7 \
            --async_mode ${async_mode} \
            --selection_ratio ${selection_ratio} \
            --importance_threshold 0.1 \
            --importance_decay 0.95 \
            --use_async_value_update
    
    # 渐进式异步训练
    elif [ "$async_mode" = "progressive" ]; then
        echo "  使用渐进式异步更新策略"
        CUDA_VISIBLE_DEVICES=0 python train_mujoco_async.py \
            --env_name ${env} \
            --algorithm_name ${algo} \
            --experiment_name "${exp}_${async_mode}" \
            --scenario ${scenario} \
            --agent_conf ${agent_conf} \
            --agent_obsk ${agent_obsk} \
            --lr 9e-5 \
            --critic_lr 5e-3 \
            --std_x_coef 1 \
            --std_y_coef 5e-1 \
            --seed ${seed_} \
            --n_training_threads 4 \
            --n_rollout_threads 16 \
            --num_mini_batch 40 \
            --episode_length 1000 \
            --num_env_steps 10000000 \
            --ppo_epoch 5 \
            --use_value_active_masks \
            --add_center_xy \
            --use_state_agent \
            --safety_bound 0.2 \
            --lamda_lagr 0.78 \
            --lagrangian_coef_rate 1e-7 \
            --async_mode ${async_mode} \
            --min_selection_ratio 0.3 \
            --max_selection_ratio 1.0 \
            --use_async_value_update
    fi
    
    echo "  种子 ${seed} 训练完成"
    echo "----------------------------------------"
done

echo "所有训练任务完成！"