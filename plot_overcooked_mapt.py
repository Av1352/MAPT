"""
plot_overcooked_mapt.py

Generates two plots for the MAPT Overcooked v2 sanity check:
1. Reward model training curve (from tensorboard logs)
2. Trajectory reward distribution (from collected trajectories)

Usage:
    python plot_overcooked_mapt.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pickle
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MAPT_DIR     = os.path.join(SCRIPT_DIR, "mat")
LOG_BASE     = os.path.join(MAPT_DIR, "results", "pref_reward", "overcooked",
                            "cramped_room", "NMR", "overcooked_cramped_room", "logs")
TRAJ_PATH    = os.path.join(os.path.expanduser("~"), "JaxMARL", "baselines",
                            "IPPO", "overcooked_trajectories.pkl")

# ── Load training curves from tensorboard ─────────────────────────────────────
def load_tfevents(tag_subpath):
    """Load scalar values from a tensorboard events file."""
    import glob
    pattern = os.path.join(LOG_BASE, tag_subpath, "*.tfevents.*")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No tfevents found at {pattern}")
    
    from tensorflow.python.summary.summary_iterator import summary_iterator
    steps, values = [], []
    for e in summary_iterator(files[0]):
        for v in e.summary.value:
            steps.append(e.step)
            values.append(v.simple_value)
    return np.array(steps), np.array(values)

print("Loading tensorboard logs...")
train_steps, train_loss = load_tfevents("reward/lstm_loss/reward/lstm_loss")
eval_steps,  eval_loss  = load_tfevents("reward/eval_lstm_loss/reward/eval_lstm_loss")
print(f"  Train: {len(train_steps)} epochs, final loss = {train_loss[-1]:.4f}")
print(f"  Eval:  {len(eval_steps)} epochs, final loss = {eval_loss[-1]:.4f}")

# ── Load trajectory rewards ───────────────────────────────────────────────────
print(f"Loading trajectories from {TRAJ_PATH}...")
with open(TRAJ_PATH, "rb") as f:
    trajectories = pickle.load(f)

expert_rewards = np.array([t["total_reward"] for t in trajectories if t["policy"] == "expert"])
random_rewards = np.array([t["total_reward"] for t in trajectories if t["policy"] == "random"])
print(f"  Expert: n={len(expert_rewards)}, mean={expert_rewards.mean():.1f}, min={expert_rewards.min():.0f}, max={expert_rewards.max():.0f}")
print(f"  Random: n={len(random_rewards)}, mean={random_rewards.mean():.1f}, min={random_rewards.min():.0f}, max={random_rewards.max():.0f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# ── Plot 1: Training Curve ────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])

# raw train (faded)
ax1.plot(train_steps, train_loss, color='steelblue', alpha=0.25, linewidth=1.0)
# smoothed train
window = 5
smooth = np.convolve(train_loss, np.ones(window)/window, mode='valid')
ax1.plot(train_steps[window-1:], smooth, color='steelblue', linewidth=2.5, label='Train loss')
# eval
ax1.plot(eval_steps, eval_loss, color='tomato', linewidth=2.5,
         linestyle='--', marker='o', markersize=4, label='Eval loss')

# annotations
ax1.annotate(f'Final: {train_loss[-1]:.3f}',
             xy=(train_steps[-1], train_loss[-1]),
             xytext=(train_steps[-1]-25, train_loss[-1]+0.02),
             fontsize=9, color='steelblue',
             arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2))
ax1.annotate(f'Final: {eval_loss[-1]:.3f}',
             xy=(eval_steps[-1], eval_loss[-1]),
             xytext=(eval_steps[-1]-30, eval_loss[-1]-0.025),
             fontsize=9, color='tomato',
             arrowprops=dict(arrowstyle='->', color='tomato', lw=1.2))

ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax1.set_title('MAPT NMR Reward Model Training\nOvercooked v2 — Cramped Room', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, train_steps[-1])

# ── Plot 2: Reward Distribution ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])

bins = np.linspace(-10, max(expert_rewards.max(), random_rewards.max()) + 20, 40)
ax2.hist(expert_rewards, bins=bins, color='steelblue', alpha=0.7,
         label=f'Expert (n={len(expert_rewards)})\nmean={expert_rewards.mean():.0f}')
ax2.hist(random_rewards, bins=bins, color='tomato', alpha=0.7,
         label=f'Random (n={len(random_rewards)})\nmean={random_rewards.mean():.1f}')

ax2.axvline(expert_rewards.mean(), color='steelblue', linewidth=2.0, linestyle='--')
ax2.axvline(random_rewards.mean(), color='tomato', linewidth=2.0, linestyle='--')

ax2.set_xlabel('Episode Return', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Trajectory Reward Distribution\nExpert vs Random Policies', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# ── Save ─────────────────────────────────────────────────────────────────────
plt.suptitle('MAPT Baseline — Overcooked v2 Sanity Check', fontsize=14, fontweight='bold', y=1.02)
plt.savefig(os.path.join(SCRIPT_DIR, 'mapt_overcooked_plots.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(SCRIPT_DIR, 'mapt_overcooked_plots.pdf'), dpi=300, bbox_inches='tight')
print("\nSaved mapt_overcooked_plots.png and .pdf")