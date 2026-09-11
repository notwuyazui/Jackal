from copy import deepcopy

from environment.jackal_env import JackalEnv


class JackalMultiAgentEnv:
    """Thin wrapper that exposes a PyMARL-like multi-agent env interface."""

    def __init__(self, env_args):
        self.env_args = deepcopy(env_args)
        self.env_name = self.env_args.pop("name", "jackal")

        self.env = JackalEnv(**self.env_args)
        obs, state = self.env.reset()

        self.n_agents = self.env.n_agents
        self.n_actions = self.env.n_actions
        self.obs_shape = obs[0].shape[0]
        self.state_shape = state.shape[0]
        self.episode_limit = self.env.max_steps

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        next_obs, next_state, reward, done, info = self.env.step(actions)
        return reward, done, info, next_obs, next_state

    def get_obs(self):
        return self.env.get_obs()

    def get_state(self):
        return self.env.get_state()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def get_env_info(self):
        return {
            "n_agents": self.n_agents,
            "n_actions": self.n_actions,
            "state_shape": self.state_shape,
            "obs_shape": self.obs_shape,
            "episode_limit": self.episode_limit,
            "env_name": self.env_name,
            "unit_type_dim": getattr(self.env, "unit_type_dim", 0),
            "obs_map_dim": self.env.observation_manager.observation_map_dim(),
            "state_map_dim": self.env.observation_manager.state_map_dim(),
        }

    def close(self):
        self.env.close()
