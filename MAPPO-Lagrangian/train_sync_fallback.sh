#!/bin/bash
env="mujoco"
scenario="Ant-v2"
algo="mappo_lagr"
exp="sync_fallback"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_mujoco.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario ${scenario} \
    --seed ${seed} \
    --n_training_threads 4 \
    --n_rollout_threads 16 \
    --num_mini_batch 40 \
    --episode_length 1000 \
    --num_env_steps 10000000 \
    --ppo_epoch 5 \
    --use_ReLU \
    --gain 0.01 \
    --lr 9e-5 \
    --critic_lr 5e-3 \
    --user_name "marl" \
    --use_centralized_V \
    --use_obs_instead_of_state \
    --agent_conf "2x4" \
    --agent_obsk 1 \
    --use_state_agent \
    --use_mustalive \
    --add_center_xy \
    --use_stacked_frames \
    --stacked_frames 1 \
    --hidden_size 64 \
    --layer_N 1 \
    --use_ReLU \
    --use_feature_normalization \
    --use_orthogonal \
    --use_gae \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --use_proper_time_limits \
    --use_huber_loss \
    --use_value_active_masks \
    --use_policy_active_masks \
    --huber_delta 10.0 \
    --use_clipped_value_loss \
    --clip_param 0.2 \
    --num_agents 2 \
    --share_policy \
    --save_interval 1 \
    --log_interval 5 \
    --eval_interval 25 \
    --safety_bound 0.2 \
    --lagrangian_coef 0.01 \
    --lamda_lagr 0.1 \
    --lagrangian_coef_rate 1e-4 \
    --use_async false
done