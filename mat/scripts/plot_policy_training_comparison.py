"""
plot_policy_training_comparison.py

Reads the tensorboard logs directly (no live server needed) and generates
saved comparison plots between the original run (negative_rewards=True,
ground-truth reward collapsed to 0) and the fixed run
(negative_rewards=False, entropy_coef=0.05, agent actually cooks).

Place this file at: mat/scripts/plot_policy_training_comparison.py

Usage:
    python plot_policy_training_comparison.py

Output:
    policy_training_comparison.png
    policy_training_comparison.pdf
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE = os.path.join(SCRIPT_DIR, "results", "Overcooked", "cramped_room", "mat")

# Adjust these run folder names if yours differ
RUN_BEFORE_FIX = os.path.join(RESULTS_BASE, "train_overcooked_cramped_room", "run8", "logs")
RUN_AFTER_FIX = os.path.join(RESULTS_BASE, "train_overcooked_cramped_room_v2_noneg_reward", "run2", "logs")

LABEL_BEFORE = "Before fix (negative_rewards=True)"
LABEL_AFTER = "After fix (negative_rewards=False, higher entropy)"


def load_scalar(logdir, tag_subpath):
    """Load a scalar series from a tensorboard logdir subfolder."""
    ea = EventAccumulator(os.path.join(logdir, tag_subpath))
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    if not tags:
        return np.array([]), np.array([])
    # there's usually exactly one scalar tag matching the folder name
    events = ea.Scalars(tags[0])
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth(values, window=5):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode='valid')


print("Loading tensorboard logs...")
before_steps, before_rewards = load_scalar(RUN_BEFORE_FIX, "train_episode_rewards/aver_rewards")
after_steps, after_rewards = load_scalar(RUN_AFTER_FIX, "train_episode_rewards/aver_rewards")

before_entropy_steps, before_entropy = load_scalar(RUN_BEFORE_FIX, "dist_entropy/dist_entropy")
after_entropy_steps, after_entropy = load_scalar(RUN_AFTER_FIX, "dist_entropy/dist_entropy")

before_eval_steps, before_eval = load_scalar(RUN_BEFORE_FIX, "eval_average_episode_rewards/aver_rewards")
after_eval_steps, after_eval = load_scalar(RUN_AFTER_FIX, "eval_average_episode_rewards/aver_rewards")

print(f"  Before fix: {len(before_rewards)} train reward points, {len(before_eval)} eval points")
print(f"  After fix:  {len(after_rewards)} train reward points, {len(after_eval)} eval points")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.35)

# ── Plot 1: Train ground-truth episode reward ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
if len(before_rewards) > 0:
    ax1.plot(before_steps, before_rewards, color='tomato', alpha=0.3, linewidth=1)
    if len(before_rewards) >= 5:
        sm = smooth(before_rewards)
        ax1.plot(before_steps[4:], sm, color='tomato', linewidth=2.5, label=LABEL_BEFORE)
if len(after_rewards) > 0:
    ax1.plot(after_steps, after_rewards, color='steelblue', alpha=0.3, linewidth=1)
    if len(after_rewards) >= 5:
        sm = smooth(after_rewards)
        ax1.plot(after_steps[4:], sm, color='steelblue', linewidth=2.5, label=LABEL_AFTER)
ax1.set_xlabel('Env Steps')
ax1.set_ylabel('Ground-Truth Episode Reward')
ax1.set_title('Training: Ground-Truth Reward', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── Plot 2: Eval ground-truth episode reward ──────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
if len(before_eval) > 0:
    ax2.plot(before_eval_steps, before_eval, color='tomato', linewidth=2,
             marker='o', markersize=3, label=LABEL_BEFORE)
if len(after_eval) > 0:
    ax2.plot(after_eval_steps, after_eval, color='steelblue', linewidth=2,
             marker='o', markersize=3, label=LABEL_AFTER)
ax2.set_xlabel('Env Steps')
ax2.set_ylabel('Eval Ground-Truth Episode Reward')
ax2.set_title('Evaluation: Ground-Truth Reward', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── Plot 3: Policy entropy ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
if len(before_entropy) > 0:
    ax3.plot(before_entropy_steps, before_entropy, color='tomato', alpha=0.4, linewidth=1, label=LABEL_BEFORE)
if len(after_entropy) > 0:
    ax3.plot(after_entropy_steps, after_entropy, color='steelblue', alpha=0.4, linewidth=1, label=LABEL_AFTER)
ax3.set_xlabel('Env Steps')
ax3.set_ylabel('Policy Entropy')
ax3.set_title('Policy Entropy (exploration vs. collapse)', fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ── Plot 4: Summary text ───────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')
summary_lines = ["Summary", ""]
if len(before_rewards) > 0:
    summary_lines.append(f"Before fix — final train reward: {before_rewards[-1]:.2f}")
    summary_lines.append(f"Before fix — mean of last 20%: {np.mean(before_rewards[-max(1,len(before_rewards)//5):]):.2f}")
if len(after_rewards) > 0:
    summary_lines.append(f"After fix — final train reward: {after_rewards[-1]:.2f}")
    summary_lines.append(f"After fix — mean of last 20%: {np.mean(after_rewards[-max(1,len(after_rewards)//5):]):.2f}")
summary_lines.append("")
summary_lines.append("Fix applied: negative_rewards=False,")
summary_lines.append("entropy_coef 0.01 -> 0.05")
ax4.text(0.05, 0.95, "\n".join(summary_lines), transform=ax4.transAxes,
         fontsize=12, verticalalignment='top', fontfamily='monospace')

plt.suptitle('MAPT Policy Training on Overcooked v2 — Before vs. After Fix',
            fontsize=15, fontweight='bold', y=1.00)

out_png = os.path.join(SCRIPT_DIR, 'policy_training_comparison.png')
out_pdf = os.path.join(SCRIPT_DIR, 'policy_training_comparison.pdf')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
print(f"\nSaved {out_png} and {out_pdf}")