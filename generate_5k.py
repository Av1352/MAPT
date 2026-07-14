"""
Generate real MAMuJoCo HalfCheetah 6x1 preference data for MAPT.
Collects trajectories using random and slightly directed policies,
labels them using ground truth reward (scripted teacher),
and saves in MAPT's pickle format.
"""

import sys
import pickle
import numpy as np

sys.path.insert(0, '/home/anjuv/multiagent_mujoco')
from multiagent_mujoco.mujoco_multi import MujocoMulti

# ── Config ────────────────────────────────────────────────────────────────────
N_PAIRS = 5000          # number of preference pairs
T = 100                 # trajectory length
SAVE_PATH = './real_halfcheetah_5k_pref_data.pkl'

# ── Setup environment ─────────────────────────────────────────────────────────
print("Setting up MAMuJoCo HalfCheetah 6x1...")

def make_env():
    return MujocoMulti(env_args={
        'scenario': 'HalfCheetah-v2',
        'agent_conf': '6x1',
        'agent_obsk': 1,
        'episode_limit': 1000
    })

env = make_env()
env.reset()
obs = np.array(env.get_obs())
N_AGENTS = env.n_agents
OBS_DIM = obs.shape[1]
ACT_DIM = 1

print(f"N_AGENTS: {N_AGENTS}, OBS_DIM: {OBS_DIM}, ACT_DIM: {ACT_DIM}")

# ── Collect one trajectory ────────────────────────────────────────────────────
def collect_trajectory(env, policy_type='random', noise_scale=1.0):
    """
    Collect a trajectory of length T.
    policy_type: 'random' or 'forward' (biased toward forward motion)
    Returns obs array (T, N_agents, obs_dim), actions (T, N_agents, act_dim), total_reward
    """
    env.reset()
    
    all_obs = []
    all_actions = []
    total_reward = 0
    
    for t in range(T):
        obs = np.array(env.get_obs())  # (N_agents, obs_dim)
        all_obs.append(obs)
        
        if policy_type == 'random':
            actions = [np.random.uniform(-1, 1, ACT_DIM) * noise_scale 
                      for _ in range(N_AGENTS)]
        elif policy_type == 'forward':
            # bias actions toward positive values to encourage forward motion
            actions = [np.random.uniform(0.0, 1.0, ACT_DIM) * noise_scale 
                      for _ in range(N_AGENTS)]
        
        all_actions.append(np.array(actions))  # (N_agents, act_dim)
        
        reward, done, _ = env.step(actions)
        total_reward += reward
        
        if done:
            # pad remaining timesteps if episode ends early
            for _ in range(T - t - 1):
                all_obs.append(obs)
                all_actions.append(np.zeros((N_AGENTS, ACT_DIM)))
            break
    
    obs_array = np.array(all_obs)         # (T, N_agents, obs_dim)
    actions_array = np.array(all_actions)  # (T, N_agents, act_dim)
    
    return obs_array, actions_array, total_reward

# ── Generate preference pairs ─────────────────────────────────────────────────
print(f"\nGenerating {N_PAIRS} real preference pairs...")
print("This will take a while — each pair requires 2 episode rollouts...\n")

dataset = []

for i in range(N_PAIRS):
    if i % 100 == 0:
        print(f"  {i}/{N_PAIRS} pairs generated...")
    
    # randomly choose policy types for each trajectory
    # mix of random and forward-biased to create diverse quality levels
    policy_choices = ['random', 'forward']
    
    p0 = np.random.choice(policy_choices)
    p1 = np.random.choice(policy_choices)
    
    noise0 = np.random.uniform(0.5, 1.5)
    noise1 = np.random.uniform(0.5, 1.5)
    
    obs0, act0, r0 = collect_trajectory(env, policy_type=p0, noise_scale=noise0)
    obs1, act1, r1 = collect_trajectory(env, policy_type=p1, noise_scale=noise1)
    
    # scripted teacher labels based on ground truth return
    margin = abs(r0 - r1)
    if margin < 1.0:
        label = 0   # too close to call
    elif r0 > r1:
        label = 1   # traj0 better
    else:
        label = -1  # traj1 better
    
    dataset.append({
        'traj0': {
            'obs': obs0.astype(np.float32),      # (T, N_agents, obs_dim)
            'actions': act0.astype(np.float32),  # (T, N_agents, act_dim)
        },
        'traj1': {
            'obs': obs1.astype(np.float32),
            'actions': act1.astype(np.float32),
        },
        'label': label,
        'return0': r0,
        'return1': r1,
    })

# env.close()  # removed - raises NotImplementedError

# ── Save ─────────────────────────────────────────────────────────────────────
print(f"\nSaving to {SAVE_PATH}...")
with open(SAVE_PATH, 'wb') as f:
    pickle.dump(dataset, f)

# ── Verify ───────────────────────────────────────────────────────────────────
print("\nVerifying...")
with open(SAVE_PATH, 'rb') as f:
    loaded = pickle.load(f)

print(f"Number of pairs: {len(loaded)}")
print(f"traj0 obs shape: {loaded[0]['traj0']['obs'].shape}")
print(f"traj0 actions shape: {loaded[0]['traj0']['actions'].shape}")

label_counts = {1: 0, -1: 0, 0: 0}
returns = []
for d in loaded:
    label_counts[d['label']] += 1
    returns.extend([d['return0'], d['return1']])

print(f"Label distribution: traj0 better={label_counts[1]}, traj1 better={label_counts[-1]}, equal={label_counts[0]}")
print(f"Return range: min={min(returns):.1f}, max={max(returns):.1f}, mean={np.mean(returns):.1f}")
print(f"\nDone! Real preference data saved to {SAVE_PATH}")
