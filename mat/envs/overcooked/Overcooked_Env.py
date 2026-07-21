"""
Overcooked_Env.py

Wraps JaxMARL's Overcooked v2 environment to match the interface that MAPT's
ShareSubprocVecEnv / ShareDummyVecEnv expect (same pattern as
mat/envs/starcraft2/StarCraft2_Env.py).

Place this file at:
    mat/envs/overcooked/Overcooked_Env.py
(create the `overcooked` folder if it doesn't exist, with an empty __init__.py
inside it, same as mat/envs/starcraft2/__init__.py)

Required interface (confirmed from mat/envs/env_wrappers.py -> shareworker):
    env.n_agents                                        (int)
    env.observation_space[i]       -> gym.spaces.Box     (per agent)
    env.share_observation_space[i] -> gym.spaces.Box     (per agent)
    env.action_space[i]            -> gym.spaces.Discrete (per agent)
    env.seed(seed)
    env.reset()  -> (obs, share_obs, available_actions)
    env.step(actions) -> (obs, share_obs, rewards, dones, infos, available_actions)
    env.close()

Shapes (n_agents = 2 for cramped_room):
    obs:               (n_agents, obs_dim)
    share_obs:         (n_agents, n_agents * obs_dim)   -- concat of all agents' obs
    available_actions: (n_agents, n_actions)            -- all ones, no action masking
    rewards:           (n_agents, 1)
    dones:             (n_agents,)  bool
    infos:             list of n_agents dicts, each must contain 'bad_transition'
"""

import numpy as np
import jax
import jaxmarl
from gym.spaces import Discrete, Box
from jaxmarl.environments.overcooked_v2 import overcooked_v2_layouts
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper


class OvercookedEnv(object):
    def __init__(self, args):
        self.layout_name = args.layout_name

        self.env = jaxmarl.make(
            "overcooked_v2",
            layout=overcooked_v2_layouts[self.layout_name],
            agent_view_size=2,
            negative_rewards=True,
            sample_recipe_on_delivery=True,
            random_agent_positions=True,
        )
        self.env = OvercookedV2LogWrapper(self.env, replace_info=False)

        self.agents = self.env.agents
        self.n_agents = self.env.num_agents

        # will be overwritten by .seed(), this is just a default
        self.key = jax.random.PRNGKey(getattr(args, "seed", 0))
        self.state = None

        # ── Reward shaping ──────────────────────────────────────────────
        # IPPO's original successful run (reward 0 -> 498) used dense
        # partial-credit shaped reward (onion pickup, pot placement,
        # plating, etc.) on top of the sparse "dish delivered" reward,
        # annealed to 0 over training so the agent ends up optimizing the
        # true sparse objective. Without this, ground-truth reward stays
        # at 0 for a very long time since full deliveries are rare for an
        # untrained policy. We mirror that pattern here.
        self.reward_shaping_horizon = getattr(args, "reward_shaping_horizon", 2_500_000)
        self.local_step_count = 0  # steps seen by THIS subprocess's env instance

        obs_shape = self.env.observation_space().shape
        self.obs_dim = int(np.prod(obs_shape))
        self.n_actions = self.env.action_space(self.agents[0]).n

        self.observation_space = [
            Box(low=-1e6, high=1e6, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            Box(low=-1e6, high=1e6, shape=(self.obs_dim * self.n_agents,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.action_space = [Discrete(self.n_actions) for _ in range(self.n_agents)]

    def seed(self, seed):
        self.key = jax.random.PRNGKey(seed)

    def _flatten_obs(self, obs_dict):
        # (n_agents, obs_dim)
        return np.stack(
            [np.array(obs_dict[a]).flatten().astype(np.float32) for a in self.agents],
            axis=0,
        )

    def _make_share_obs(self, obs):
        # concat every agent's obs into one vector, tiled per agent -> (n_agents, n_agents*obs_dim)
        flat = obs.reshape(1, -1)
        return np.tile(flat, (self.n_agents, 1)).astype(np.float32)

    def reset(self):
        self.key, subkey = jax.random.split(self.key)
        obs_dict, self.state = self.env.reset(subkey)
        obs = self._flatten_obs(obs_dict)
        share_obs = self._make_share_obs(obs)
        available_actions = np.ones((self.n_agents, self.n_actions), dtype=np.float32)
        return obs, share_obs, available_actions

    def step(self, actions):
        # actions arrives as shape (n_agents, 1) or (n_agents,) — squeeze to scalars
        self.key, subkey = jax.random.split(self.key)
        action_dict = {
            a: int(np.array(actions[i]).squeeze()) for i, a in enumerate(self.agents)
        }
        obs_dict, self.state, reward_dict, done_dict, info = self.env.step(
            subkey, self.state, action_dict
        )

        obs = self._flatten_obs(obs_dict)
        share_obs = self._make_share_obs(obs)

        # ── Apply annealed shaped reward, same pattern as ippo_rnn_overcooked_v2.py ──
        self.local_step_count += 1
        anneal_factor = max(0.0, 1.0 - self.local_step_count / self.reward_shaping_horizon)
        shaped_reward = info.get("shaped_reward", None)
        if shaped_reward is not None:
            combined_reward = {
                a: float(reward_dict[a]) + float(shaped_reward[a]) * anneal_factor
                for a in self.agents
            }
        else:
            combined_reward = {a: float(reward_dict[a]) for a in self.agents}

        rewards = np.array([[combined_reward[a]] for a in self.agents], dtype=np.float32)

        done_env = bool(done_dict["__all__"])
        dones = np.array([done_env] * self.n_agents, dtype=bool)

        # 'bad_transition' is required by base_runner.insert() for bad_masks;
        # Overcooked v2 episodes end on the timestep limit, not on a failure
        # state, so this is always False.
        infos = [{"bad_transition": False} for _ in range(self.n_agents)]

        available_actions = np.ones((self.n_agents, self.n_actions), dtype=np.float32)

        return obs, share_obs, rewards, dones, infos, available_actions

    def close(self):
        pass