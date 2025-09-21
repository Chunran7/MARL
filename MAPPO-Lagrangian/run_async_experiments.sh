#!/bin/bash
# 异步MAPPO-Lagrangian对比实验脚本
# 运行不同异步策略的对比实验

# 实验配置
env="mujoco"
scenarios=("Ant-v2" "HalfCheetah-v2")
agent_confs=("2x4" "2x3")
algo="rmappo_lagr"
seed_max=3
base_seed=50

# 异步策略配置
async_modes=("hierarchical" "importance" "progressive")

echo "=========================================="
echo "异步MAPPO-Lagrangian对比实验"
echo "环境: ${scenarios[@]}"
echo "智能体配置: ${agent_confs[@]}"
echo "异步策略: ${async_modes[@]}"
echo "种子数量: ${seed_max}"
echo "=========================================="

# 创建实验结果目录
mkdir -p results/async_experiments
mkdir -p logs/async_experiments

# 遍历所有配置组合
for i in "${!scenarios[@]}"; do
    scenario=${scenarios[$i]}
    agent_conf=${agent_confs[$i]}
    
    echo "开始实验: ${scenario} with ${agent_conf}"
    echo "------------------------------------------"
    
    # 设置环境特定参数
    if [ "$scenario" = "Ant-v2" ]; then
        safety_bound=0.2
        lamda_lagr=0.78
        episode_length=1000
        num_env_steps=10000000
    elif [ "$scenario" = "HalfCheetah-v2" ]; then
        safety_bound=5.0
        lamda_lagr=0.5
        episode_length=1000
        num_env_steps=8000000
    fi
    
    # 运行不同异步策略
    for async_mode in "${async_modes[@]}"; do
        echo "  运行异步策略: ${async_mode}"
        
        for seed in $(seq 1 ${seed_max}); do
            current_seed=$((base_seed + seed))
            exp_name="async_${async_mode}_${scenario}_${agent_conf}_seed${seed}"
            
            echo "    种子 ${current_seed}: ${exp_name}"
            
            # 根据异步模式设置特定参数
            case $async_mode in
                "hierarchical")
                    python train_mujoco_async.py \
                        --env_name ${env} \
                        --algorithm_name ${algo} \
                        --experiment_name ${exp_name} \
                        --scenario ${scenario} \
                        --agent_conf ${agent_conf} \
                        --agent_obsk 1 \
                        --lr 9e-5 \
                        --critic_lr 5e-3 \
                        --std_x_coef 1 \
                        --std_y_coef 5e-1 \
                        --seed ${current_seed} \
                        --n_training_threads 4 \
                        --n_rollout_threads 16 \
                        --num_mini_batch 40 \
                        --episode_length ${episode_length} \
                        --num_env_steps ${num_env_steps} \
                        --ppo_epoch 5 \
                        --use_value_active_masks \
                        --add_center_xy \
                        --use_state_agent \
                        --safety_bound ${safety_bound} \
                        --lamda_lagr ${lamda_lagr} \
                        --lagrangian_coef_rate 1e-7 \
                        --async_mode ${async_mode} \
                        --selection_ratio 0.5 \
                        --hierarchy_levels 3 \
                        --use_async_value_update \
                        --use_wandb False \
                        > logs/async_experiments/${exp_name}.log 2>&1 &
                    ;;
                    
                "importance")
                    python train_mujoco_async.py \
                        --env_name ${env} \
                        --algorithm_name ${algo} \
                        --experiment_name ${exp_name} \
                        --scenario ${scenario} \
                        --agent_conf ${agent_conf} \
                        --agent_obsk 1 \
                        --lr 9e-5 \
                        --critic_lr 5e-3 \
                        --std_x_coef 1 \
                        --std_y_coef 5e-1 \
                        --seed ${current_seed} \
                        --n_training_threads 4 \
                        --n_rollout_threads 16 \
                        --num_mini_batch 40 \
                        --episode_length ${episode_length} \
                        --num_env_steps ${num_env_steps} \
                        --ppo_epoch 5 \
                        --use_value_active_masks \
                        --add_center_xy \
                        --use_state_agent \
                        --safety_bound ${safety_bound} \
                        --lamda_lagr ${lamda_lagr} \
                        --lagrangian_coef_rate 1e-7 \
                        --async_mode ${async_mode} \
                        --selection_ratio 0.6 \
                        --importance_threshold 0.1 \
                        --importance_decay 0.95 \
                        --use_async_value_update \
                        --use_wandb False \
                        > logs/async_experiments/${exp_name}.log 2>&1 &
                    ;;
                    
                "progressive")
                    python train_mujoco_async.py \
                        --env_name ${env} \
                        --algorithm_name ${algo} \
                        --experiment_name ${exp_name} \
                        --scenario ${scenario} \
                        --agent_conf ${agent_conf} \
                        --agent_obsk 1 \
                        --lr 9e-5 \
                        --critic_lr 5e-3 \
                        --std_x_coef 1 \
                        --std_y_coef 5e-1 \
                        --seed ${current_seed} \
                        --n_training_threads 4 \
                        --n_rollout_threads 16 \
                        --num_mini_batch 40 \
                        --episode_length ${episode_length} \
                        --num_env_steps ${num_env_steps} \
                        --ppo_epoch 5 \
                        --use_value_active_masks \
                        --add_center_xy \
                        --use_state_agent \
                        --safety_bound ${safety_bound} \
                        --lamda_lagr ${lamda_lagr} \
                        --lagrangian_coef_rate 1e-7 \
                        --async_mode ${async_mode} \
                        --min_selection_ratio 0.3 \
                        --max_selection_ratio 1.0 \
                        --use_async_value_update \
                        --use_wandb False \
                        > logs/async_experiments/${exp_name}.log 2>&1 &
                    ;;
            esac
            
            # 等待一段时间避免同时启动太多进程
            sleep 5
        done
        
        echo "    ${async_mode} 策略的所有种子已启动"
    done
    
    echo "  ${scenario} 实验已启动"
    echo ""
done

echo "所有实验已启动！"
echo "使用以下命令监控进程:"
echo "  ps aux | grep train_mujoco_async"
echo "使用以下命令查看日志:"
echo "  tail -f logs/async_experiments/*.log"
echo "使用以下命令等待所有任务完成:"
echo "  wait"