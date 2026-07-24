#!/bin/sh
# train_overcooked_seeds.sh
#
# Runs the Overcooked policy training for seeds 1, 2, 3 SEQUENTIALLY
# (not in parallel — your 6GB GPU already OOM'd once running multiple
# JAX/CUDA contexts at once, so seeds run one after another instead).
#
# Uses the same fix as the working run: negative_rewards=False (already
# baked into Overcooked_Env.py) and entropy_coef=0.05.
#
# 1.5M steps per seed instead of the full 5M — reward peaked around
# step 500k and declined afterward in the first run, so this captures
# the meaningful part of the curve without 3x the full runtime.
#
# Place this file at: mat/scripts/train_overcooked_seeds.sh
#
# Run with: ./train_overcooked_seeds.sh
# (safe to background with nohup — see note at the bottom)

env="Overcooked"
layout="cramped_room"
model_type="NMR"
algo="mat"
num_steps=1500000

PREF_MODEL_DIR="/mnt/c/Users/anjuv/OneDrive/Documents/GitHub/MAPT/mat/results/pref_reward/overcooked/cramped_room/NMR/overcooked_cramped_room/models/reward_model_95.pt"

for seed in 1 2 3
do
    exp="overcooked_multiseed_s${seed}"
    echo ""
    echo "=========================================="
    echo "Starting seed ${seed} — experiment: ${exp}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=0 python train_policy/train_overcooked.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --layout_name ${layout} --seed ${seed} \
    --n_training_threads 8 --n_rollout_threads 4 --num_mini_batch 1 --episode_length 100 --num_env_steps ${num_steps} \
    --lr 5e-4 --critic_lr 5e-4 --ppo_epoch 15 --clip_param 0.2 --entropy_coef 0.05 \
    --save_interval 10 --use_value_active_masks --use_eval --log_interval 5 --eval_interval 20 --eval_episodes 16 \
    --use_preference_reward --preference_model_type ${model_type} --preference_reward_std 0.1 \
    --preference_model_dir "${PREF_MODEL_DIR}" \
    --preference_embd_dim 256 --preference_n_layer 1 --preference_n_head 4 \
    --preference_traj_length 100

    echo ""
    echo "Seed ${seed} finished."
done

echo ""
echo "All 3 seeds complete."

# To run this in the background so it survives closing the terminal:
#   export XLA_PYTHON_CLIENT_PREALLOCATE=false
#   nohup ./train_overcooked_seeds.sh > multiseed_log.txt 2>&1 &