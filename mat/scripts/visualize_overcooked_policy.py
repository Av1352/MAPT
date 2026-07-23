"""
visualize_overcooked_policy.py

Loads your trained MAT checkpoint (transformer.pt) and plays out a few
deterministic episodes in the real Overcooked v2 environment, saving them
as an animated gif so you can literally watch what the agent is doing.

This exists to answer one specific question: is the policy actually stuck
doing something degenerate (standing still, wandering aimlessly, bumping
into a wall on repeat), or is it doing something plausible-but-imperfect
that just hasn't produced a full delivery yet?

Place this file at:
    mat/scripts/visualize_overcooked_policy.py
(next to train_overcooked_cramped_room_policy.sh, so the relative imports
sys.path.append("../") line below resolves correctly)

Usage:
    python visualize_overcooked_policy.py --model_dir <path_to_run>/models --n_episodes 3

Output:
    overcooked_rollout_ep0.gif, overcooked_rollout_ep1.gif, ... in the
    current directory
"""
import sys
import argparse
import numpy as np
import torch
import jax
import jax.numpy as jnp

sys.path.append("../../")
from mat.config import get_config
from mat.envs.overcooked.Overcooked_Env import OvercookedEnv
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer


def _t2n(x):
    return x.detach().cpu().numpy()


def _unwrap_to_grid_state(state):
    """Walk down through wrapper layers until we find the actual game
    state (the one with a .grid attribute), which is what the visualizer
    needs. Avoids another round of manual attribute-name debugging."""
    seen = 0
    while not hasattr(state, "grid") and seen < 5:
        if hasattr(state, "env_state"):
            state = state.env_state
        elif hasattr(state, "state"):
            state = state.state
        else:
            raise AttributeError(
                f"Could not find a .grid attribute by unwrapping. "
                f"Stuck at type {type(state)} with fields "
                f"{[f.name for f in __import__('dataclasses').fields(state)]}. "
                f"Please paste this error back."
            )
        seen += 1
    return state


def main():
    # ── Build the same all_args the training run used ──────────────────
    # We reuse get_config() + the same extra flags train_overcooked.py
    # defines, so the TransformerPolicy is constructed with IDENTICAL
    # hyperparameters to the ones the checkpoint was actually trained
    # with. Mismatched hidden_size/n_block/n_embd/etc. would silently
    # produce wrong results or a shape-mismatch error on load.
    parser = get_config()
    parser.add_argument('--layout_name', type=str, default='cramped_room')
    parser.add_argument('--num_agents', type=int, default=2)
    parser.add_argument('--reward_shaping_horizon', type=int, default=2_500_000)
    parser.add_argument('--model_dir_to_load', type=str, required=True,
                        help="Path to the 'models' folder containing transformer.pt "
                             "(e.g. .../train_overcooked_cramped_room/run8/models)")
    parser.add_argument('--n_episodes', type=int, default=3)
    parser.add_argument('--out_prefix', type=str, default='overcooked_rollout')

    all_args = parser.parse_known_args(sys.argv[1:])[0]

    # These two flags need to match what training used, since they affect
    # network architecture and observation handling
    all_args.algorithm_name = "mat"
    all_args.env_name = "Overcooked"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Build a single (non-vectorized) env instance ────────────────────
    env = OvercookedEnv(all_args)

    # ── Load the trained policy ──────────────────────────────────────────
    policy = Policy(
        all_args,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        env.n_agents,
        device=device,
    )
    print(f"Restoring checkpoint from {all_args.model_dir_to_load} ...")
    transformer_path = all_args.model_dir_to_load.rstrip("/") + "/transformer.pt"
    policy.restore(transformer_path, None)
    print("Checkpoint loaded.")

    visualizer = OvercookedV2Visualizer()

    for ep in range(all_args.n_episodes):
        print(f"\n--- Running episode {ep} ---")
        obs, share_obs, available_actions = env.reset()

        # state_seq collects the RAW jaxmarl env state at every timestep,
        # which is what the visualizer actually needs to render frames
        # (not our flattened obs vectors). env.state is wrapped by
        # OvercookedV2LogWrapper, so we unwrap .env_state to get the
        # actual game state (the one with .grid) that the visualizer expects.
        state_seq = [_unwrap_to_grid_state(env.state)]

        rnn_states = np.zeros(
            (env.n_agents, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32
        )
        masks = np.ones((env.n_agents, 1), dtype=np.float32)

        episode_ground_truth_reward = 0.0

        for step in range(all_args.episode_length):
            with torch.no_grad():
                actions, rnn_states = policy.act(
                    share_obs,
                    obs,
                    rnn_states,
                    masks,
                    available_actions,
                    deterministic=True,
                )
            actions = _t2n(actions)
            rnn_states = _t2n(rnn_states)

            obs, share_obs, rewards, dones, infos, available_actions = env.step(actions)
            episode_ground_truth_reward += float(np.sum(rewards))

            state_seq.append(_unwrap_to_grid_state(env.state))

            if bool(np.all(dones)):
                break

        print(f"Episode {ep} total ground-truth reward: {episode_ground_truth_reward:.2f}")
        print(f"Episode {ep} length: {len(state_seq)} frames")

        # animate() vmaps over axis 0, so it needs ONE stacked pytree with
        # a leading time dimension, not a Python list of separate states.
        import jax
        import jax.numpy as jnp
        stacked_state_seq = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *state_seq)

        out_path = f"{all_args.out_prefix}_ep{ep}.gif"
        print(f"Saving animation to {out_path} ...")
        visualizer.animate(stacked_state_seq, filename=out_path, agent_view_size=2)
        print(f"Saved {out_path}")

    env.close()
    print("\nDone. Open the .gif files to see what the agent is actually doing.")


if __name__ == "__main__":
    main()