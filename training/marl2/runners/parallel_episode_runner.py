import multiprocessing as mp
import random
from typing import Any

import numpy as np
import torch


def _env_worker(remote, parent_remote, env_cls, env_args, seed):
    parent_remote.close()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    env = env_cls(env_args)

    try:
        while True:
            try:
                cmd, data = remote.recv()
            except EOFError:
                break

            if cmd == "reset":
                obs, state = env.reset()
                remote.send((obs, state))
            elif cmd == "step":
                reward, terminated, info, next_obs, next_state = env.step(data)
                remote.send((reward, terminated, info, next_obs, next_state))
            elif cmd == "get_avail_actions":
                remote.send(env.get_avail_actions())
            elif cmd == "get_env_info":
                remote.send(env.get_env_info())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise ValueError(f"Unknown worker command: {cmd}")
    finally:
        try:
            env.close()
        except Exception:
            pass


class ParallelEpisodeRunner:
    """Collect one episode per environment in parallel subprocesses."""

    def __init__(self, env_cls, env_args, mac, n_envs, seed=None, start_method="spawn"):
        self.env_cls = env_cls
        self.env_args = dict(env_args)
        self.mac = mac
        self.n_envs = int(n_envs)

        if self.n_envs < 2:
            raise ValueError("ParallelEpisodeRunner requires n_envs >= 2")

        self._ctx = mp.get_context(start_method)
        self.parent_conns = []
        self.processes = []
        self.closed = False

        for idx in range(self.n_envs):
            parent_conn, worker_conn = self._ctx.Pipe()
            worker_seed = None if seed is None else int(seed) + idx + 1
            process = self._ctx.Process(
                target=_env_worker,
                args=(worker_conn, parent_conn, self.env_cls, self.env_args, worker_seed),
                daemon=True,
            )
            process.start()
            worker_conn.close()

            self.parent_conns.append(parent_conn)
            self.processes.append(process)

        self.parent_conns[0].send(("get_env_info", None))
        env_info = self.parent_conns[0].recv()

        self.n_agents = int(env_info["n_agents"])
        self.n_actions = int(env_info["n_actions"])
        self.obs_shape = int(env_info["obs_shape"])
        self.state_shape = int(env_info["state_shape"])
        self.episode_limit = int(env_info["episode_limit"])

    def _new_episode_batch(self):
        return {
            "obs": np.zeros((self.episode_limit + 1, self.n_agents, self.obs_shape), dtype=np.float32),
            "state": np.zeros((self.episode_limit + 1, self.state_shape), dtype=np.float32),
            "avail_actions": np.zeros((self.episode_limit + 1, self.n_agents, self.n_actions), dtype=np.float32),
            "actions": np.zeros((self.episode_limit, self.n_agents, 1), dtype=np.int64),
            "reward": np.zeros((self.episode_limit, 1), dtype=np.float32),
            "terminated": np.zeros((self.episode_limit, 1), dtype=np.float32),
            "filled": np.zeros((self.episode_limit, 1), dtype=np.float32),
        }

    def run(self, test_mode=False, epsilon=0.0):
        if self.closed:
            raise RuntimeError("ParallelEpisodeRunner has already been closed")

        for conn in self.parent_conns:
            conn.send(("reset", None))
        reset_results = [conn.recv() for conn in self.parent_conns]

        obs_list = [res[0] for res in reset_results]
        state_list = [res[1] for res in reset_results]

        self.mac.init_hidden(batch_size=self.n_envs)

        episode_batches = [self._new_episode_batch() for _ in range(self.n_envs)]
        episode_returns = [0.0 for _ in range(self.n_envs)]
        episode_lengths = [0 for _ in range(self.n_envs)]
        final_infos: list[dict[str, Any]] = [{} for _ in range(self.n_envs)]
        terminated = [False for _ in range(self.n_envs)]

        for t in range(self.episode_limit):
            active_envs = [idx for idx in range(self.n_envs) if not terminated[idx]]
            if not active_envs:
                break

            active_avail_map = {}
            for env_idx in active_envs:
                self.parent_conns[env_idx].send(("get_avail_actions", None))
            for env_idx in active_envs:
                active_avail_map[env_idx] = self.parent_conns[env_idx].recv()

            obs_batch = np.array(obs_list, dtype=np.float32)
            avail_batch: np.ndarray = np.zeros(
                (self.n_envs, self.n_agents, self.n_actions), dtype=np.float32
            )
            for env_idx in active_envs:
                avail_batch[env_idx] = np.array(active_avail_map[env_idx], dtype=np.float32)

            chosen_actions_batch = self.mac.select_actions_batch(
                obs_batch=obs_batch,
                avail_actions_batch=avail_batch,
                epsilon=epsilon,
                test_mode=test_mode,
                active_envs=active_envs,
            )

            for env_idx in active_envs:
                episode_batches[env_idx]["obs"][t] = np.array(obs_list[env_idx], dtype=np.float32)
                episode_batches[env_idx]["state"][t] = np.array(state_list[env_idx], dtype=np.float32)
                episode_batches[env_idx]["avail_actions"][t] = np.array(active_avail_map[env_idx], dtype=np.float32)

                self.parent_conns[env_idx].send(("step", chosen_actions_batch[env_idx].tolist()))

            for env_idx in active_envs:
                reward, done, info, next_obs, next_state = self.parent_conns[env_idx].recv()
                episode_batches[env_idx]["actions"][t, :, 0] = np.array(chosen_actions_batch[env_idx], dtype=np.int64)
                episode_batches[env_idx]["reward"][t, 0] = float(reward)
                episode_batches[env_idx]["terminated"][t, 0] = float(done)
                episode_batches[env_idx]["filled"][t, 0] = 1.0

                episode_returns[env_idx] += float(reward)
                episode_lengths[env_idx] = t + 1
                final_infos[env_idx] = info

                obs_list[env_idx] = next_obs
                state_list[env_idx] = next_state
                terminated[env_idx] = bool(done)

        avail_actions_last: list[np.ndarray] = [
            np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
            for _ in range(self.n_envs)
        ]
        active_last_envs = [idx for idx in range(self.n_envs) if not terminated[idx]]
        for env_idx in active_last_envs:
            self.parent_conns[env_idx].send(("get_avail_actions", None))
        for env_idx in active_last_envs:
            avail_actions_last[env_idx] = self.parent_conns[env_idx].recv()
        for env_idx in range(self.n_envs):
            if terminated[env_idx]:
                terminal_avail: np.ndarray = np.zeros(
                    (self.n_agents, self.n_actions), dtype=np.float32
                )
                terminal_avail[:, 0] = 1.0
                avail_actions_last[env_idx] = terminal_avail

        for env_idx in range(self.n_envs):
            t_ep = episode_lengths[env_idx]
            episode_batches[env_idx]["obs"][t_ep] = np.array(obs_list[env_idx], dtype=np.float32)
            episode_batches[env_idx]["state"][t_ep] = np.array(state_list[env_idx], dtype=np.float32)
            episode_batches[env_idx]["avail_actions"][t_ep] = np.array(avail_actions_last[env_idx], dtype=np.float32)

        stats_list = [
            {
                "episode_return": episode_returns[env_idx],
                "episode_length": episode_lengths[env_idx],
                "battle_won": bool(final_infos[env_idx].get("battle_won", False)),
                "episode_limit": bool(final_infos[env_idx].get("episode_limit", False)),
                "no_kill_timeout": bool(final_infos[env_idx].get("no_kill_timeout", False)),
            }
            for env_idx in range(self.n_envs)
        ]

        return episode_batches, stats_list

    def close(self):
        if self.closed:
            return

        for conn in self.parent_conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass

        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()

        for conn in self.parent_conns:
            try:
                conn.close()
            except Exception:
                pass

        self.closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
