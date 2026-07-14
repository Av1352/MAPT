"""
Generate synthetic preference data for MAPT sanity check.
Simulates HalfCheetah-v2 6x1 configuration:
- 6 agents, each controlling 1 joint
- obs_dim = 17 (standard HalfCheetah observation)
- act_dim = 1 (each agent controls 1 joint)
- T = 100 timesteps per trajectory
- 50,000 preference pairs total (matching MAPT paper)
"""

import pickle
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
N_AGENTS = 6
OBS_DIM = 17       # HalfCheetah-v2 observation dimension
ACT_DIM = 1        # each agent controls 1 joint
T = 100            # trajectory length
N_PAIRS = 5000     # number of preference pairs (start small for sanity check)
SAVE_PATH = "./synthetic_halfcheetah_6x1_pref_data.pkl"

# ── Helper: generate a trajectory with a given quality level ─────────────────
def generate_trajectory(quality=1.0):
    """
    Generate a synthetic trajectory.
    quality: float between 0 and 1. Higher = better trajectory.
    The 'reward' of a trajectory is correlated with quality.
    """
    # observations: random but scaled by quality
    obs = np.random.randn(T, N_AGENTS, OBS_DIM).astype(np.float32)
    obs *= quality  # higher quality = larger, more purposeful movements

    # actions: random continuous actions in [-1, 1]
    actions = np.random.uniform(-1, 1, (T, N_AGENTS, ACT_DIM)).astype(np.float32)
    actions *= quality  # higher quality = stronger actions

    return obs, actions

def compute_trajectory_return(obs, actions):
    """
    Synthetic ground truth return.
    In real MAMuJoCo this would be the actual environment reward.
    Here we use a simple proxy: mean magnitude of observations * actions.
    """
    return np.mean(np.abs(obs * actions))

# ── Generate preference pairs ─────────────────────────────────────────────────
print(f"Generating {N_PAIRS} preference pairs...")
print(f"Config: {N_AGENTS} agents, obs_dim={OBS_DIM}, act_dim={ACT_DIM}, T={T}")

dataset = []

for i in range(N_PAIRS):
    if i % 500 == 0:
        print(f"  {i}/{N_PAIRS} pairs generated...")

    # generate two trajectories with random quality levels
    q0 = np.random.uniform(0.1, 1.0)
    q1 = np.random.uniform(0.1, 1.0)

    obs0, act0 = generate_trajectory(quality=q0)
    obs1, act1 = generate_trajectory(quality=q1)

    # compute returns (scripted teacher uses ground truth return)
    r0 = compute_trajectory_return(obs0, act0)
    r1 = compute_trajectory_return(obs1, act1)

    # label: 1 if traj0 is better, -1 if traj1 is better, 0 if equal
    margin = abs(r0 - r1)
    if margin < 0.01:
        label = 0   # too close to call
    elif r0 > r1:
        label = 1   # traj0 is better
    else:
        label = -1  # traj1 is better

    dataset.append({
        'traj0': {
            'obs': obs0,        # shape: (T, N_agents, obs_dim)
            'actions': act0,    # shape: (T, N_agents, act_dim)
        },
        'traj1': {
            'obs': obs1,
            'actions': act1,
        },
        'label': label,
        'return0': r0,  # for debugging
        'return1': r1,  # for debugging
    })

# ── Save ─────────────────────────────────────────────────────────────────────
print(f"\nSaving dataset to {SAVE_PATH}...")
with open(SAVE_PATH, 'wb') as f:
    pickle.dump(dataset, f)

# ── Verify ───────────────────────────────────────────────────────────────────
print("\nVerifying saved dataset...")
with open(SAVE_PATH, 'rb') as f:
    loaded = pickle.load(f)

print(f"Number of pairs: {len(loaded)}")
print(f"traj0 obs shape: {loaded[0]['traj0']['obs'].shape}")
print(f"traj0 actions shape: {loaded[0]['traj0']['actions'].shape}")
print(f"traj1 obs shape: {loaded[0]['traj1']['obs'].shape}")
print(f"traj1 actions shape: {loaded[0]['traj1']['actions'].shape}")
print(f"Sample labels: {[d['label'] for d in loaded[:10]]}")

label_counts = {1: 0, -1: 0, 0: 0}
for d in loaded:
    label_counts[d['label']] += 1
print(f"Label distribution: traj0 better={label_counts[1]}, traj1 better={label_counts[-1]}, equal={label_counts[0]}")

print("\nDone! Dataset ready for MAPT reward model training.")
print(f"Use this path in the training script: {SAVE_PATH}")
