from copy import deepcopy

from environment.jackal_env import JackalEnv


class JackalMultiAgentEnv:
    """Thin wrapper that exposes a PyMARL-like multi-agent env interface."""

    def __init__(self, env_args):
        self.env_args = deepcopy(env_args)
        self.env_name = self.env_args.pop("name", "jackal")

        self.env = JackalEnv(**self.env_args)
        env_info = self.env.get_env_info()

        self.n_agents = int(env_info["n_agents"])
        self.n_actions = int(env_info["n_actions"])
        self.obs_shape = int(env_info["obs_shape"])
        self.state_shape = int(env_info["state_shape"])
        self.episode_limit = int(env_info["episode_limit"])
        self._reset_shapes_validated = False

    def reset(self):
        obs, state = self.env.reset()
        if not self._reset_shapes_validated:
            if obs[0].shape[0] != self.obs_shape or state.shape[0] != self.state_shape:
                raise RuntimeError("Environment metadata does not match reset output shapes")
            self._reset_shapes_validated = True
        return obs, state

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
        info = self.env.get_env_info()
        info["env_name"] = self.env_name
        return info

    def close(self):
        self.env.close()
