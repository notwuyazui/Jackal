import numpy as np


class EpisodeRunner:
    def __init__(self, env, mac):
        self.env = env
        self.mac = mac
        env_info = env.get_env_info()
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        self.obs_shape = env_info["obs_shape"]
        self.state_shape = env_info["state_shape"]
        self.episode_limit = env_info["episode_limit"]

    def run(self, test_mode=False, epsilon=0.0):
        obs, state = self.env.reset()
        self.mac.init_hidden(batch_size=1)

        episode_batch = {
            "obs": np.zeros((self.episode_limit + 1, self.n_agents, self.obs_shape), dtype=np.float32),
            "state": np.zeros((self.episode_limit + 1, self.state_shape), dtype=np.float32),
            "avail_actions": np.zeros((self.episode_limit + 1, self.n_agents, self.n_actions), dtype=np.float32),
            "actions": np.zeros((self.episode_limit, self.n_agents, 1), dtype=np.int64),
            "reward": np.zeros((self.episode_limit, 1), dtype=np.float32),
            "terminated": np.zeros((self.episode_limit, 1), dtype=np.float32),
            "filled": np.zeros((self.episode_limit, 1), dtype=np.float32),
        }

        episode_return = 0.0
        final_info = {}
        t_episode = 0
        final_terminated = False

        for t in range(self.episode_limit):
            avail_actions = self.env.get_avail_actions()
            episode_batch["obs"][t] = np.array(obs, dtype=np.float32)
            episode_batch["state"][t] = np.array(state, dtype=np.float32)
            episode_batch["avail_actions"][t] = np.array(avail_actions, dtype=np.float32)

            chosen_actions = self.mac.select_actions(
                obs=obs,
                avail_actions=avail_actions,
                epsilon=epsilon,
                test_mode=test_mode,
            )

            reward, terminated, info, next_obs, next_state = self.env.step(chosen_actions)
            episode_batch["actions"][t, :, 0] = np.array(chosen_actions, dtype=np.int64)
            episode_batch["reward"][t, 0] = float(reward)
            episode_batch["terminated"][t, 0] = float(terminated)
            episode_batch["filled"][t, 0] = 1.0

            episode_return += float(reward)
            final_info = info
            obs = next_obs
            state = next_state
            t_episode = t + 1
            final_terminated = bool(terminated)

            if terminated:
                break

        # Store terminal timestep features for bootstrap.
        if final_terminated:
            avail_actions = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
            avail_actions[:, 0] = 1.0
        else:
            avail_actions = self.env.get_avail_actions()
        episode_batch["obs"][t_episode] = np.array(obs, dtype=np.float32)
        episode_batch["state"][t_episode] = np.array(state, dtype=np.float32)
        episode_batch["avail_actions"][t_episode] = np.array(avail_actions, dtype=np.float32)

        stats = {
            "episode_return": episode_return,
            "episode_length": t_episode,
            "battle_won": bool(final_info.get("battle_won", False)),
            "episode_limit": bool(final_info.get("episode_limit", False)),
            "no_kill_timeout": bool(final_info.get("no_kill_timeout", False)),
        }
        return episode_batch, stats

    def close(self):
        self.env.close()
