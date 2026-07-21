#!/bin/sh
# train_overcooked_cramped_room_policy.sh
#
# Modeled on mat/scripts/train_smac_3m_policy.sh.
# Place this file at: mat/scripts/train_overcooked_cramped_room_policy.sh
#
# BEFORE RUNNING: replace --preference_model_dir below with the actual path
# to your trained NMR checkpoint. It'll be somewhere like:
#   mat/scripts/results/pref_reward/overcooked/cramped_room/NMR/overcooked_cramped_room/models/reward_model_<epoch>.pt
# Run this to find the exact filename (it saves the checkpoint with the
# lowest eval loss, so pick the highest epoch number that got saved):
#   find mat/scripts/results/pref_reward/overcooked -iname "*.pt"

env="Overcooked"
layout="cramped_room"
model_type="NMR"
algo="mat"
exp="train_overcooked_cramped_room"
seed=1

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train_policy/train_overcooked.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--layout_name ${layout} --seed ${seed} \
--n_training_threads 8 --n_rollout_threads 4 --num_mini_batch 1 --episode_length 100 --num_env_steps 5000000 \
--lr 5e-4 --critic_lr 5e-4 --ppo_epoch 15 --clip_param 0.2 \
--save_interval 10 --use_value_active_masks --use_eval --log_interval 5 --eval_interval 20 --eval_episodes 16 \
--use_preference_reward --preference_model_type ${model_type} --preference_reward_std 0.1 \
--preference_model_dir "/mnt/c/Users/anjuv/OneDrive/Documents/GitHub/MAPT/mat/results/pref_reward/overcooked/cramped_room/NMR/overcooked_cramped_room/models/reward_model_95.pt" \
--preference_embd_dim 256 --preference_n_layer 1 --preference_n_head 4 \
--preference_traj_length 100
