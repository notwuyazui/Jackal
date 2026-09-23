import multiprocessing as mp
import random
from typing import Any

import numpy as np
import torch


def _derive_worker_seed(
    base_seed: int | None,
    worker_index: int,
) -> int | None:
    if base_seed is None:
        return None
    return (int(base_seed) + int(worker_index)) % (2**32)


def _env_worker(remote, parent_remote, env_cls, env_args, seed):
    parent_remote.close()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    worker_env_args = dict(env_args)
    if seed is not None:
        worker_env_args["seed"] = int(seed)
    env = env_cls(worker_env_args)

    try:
        while True:
            try:
                cmd, data = remote.recv()
            except EOFError:
                break

            if cmd == "reset":
                episode_seed = None
                if data is not None:
                    episode_seed = int(data)
                    random.seed(episode_seed)
                    np.random.seed(episode_seed % (2**32))
                    torch.manual_seed(episode_seed)
                obs, state = env.reset(seed=episode_seed)
                remote.send((obs, state, env.get_avail_actions()))
            elif cmd == "step":
                reward, terminated, info, next_obs, next_state = env.step(data)
                next_avail_actions = None if terminated else env.get_avail_actions()
                remote.send(
                    (
                        reward,
                        terminated,
                        info,
                        next_obs,
                        next_state,
                        next_avail_actions,
                    )
                )
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
        base_seed = seed if seed is not None else self.env_args.get("seed")

        for idx in range(self.n_envs):
            parent_conn, worker_conn = self._ctx.Pipe()
            worker_seed = _derive_worker_seed(base_seed, idx)
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

    def run(
        self,
        test_mode=False,
        epsilon=0.0,
        *,
        n_episodes=None,
        episode_seeds=None,
        collect_batches=True,
    ):
        if self.closed:
            raise RuntimeError("ParallelEpisodeRunner has already been closed")

        run_count = self.n_envs if n_episodes is None else int(n_episodes)
        if run_count < 1 or run_count > self.n_envs:
            raise ValueError(f"n_episodes must be in [1, {self.n_envs}], got {run_count}")
        if episode_seeds is not None and len(episode_seeds) != run_count:
            raise ValueError("episode_seeds length must match n_episodes")

        active_conns = self.parent_conns[:run_count]
        for env_idx, conn in enumerate(active_conns):
            episode_seed = None if episode_seeds is None else episode_seeds[env_idx]
            conn.send(("reset", episode_seed))
        reset_results = [conn.recv() for conn in active_conns]

        obs_list = [res[0] for res in reset_results]
        state_list = [res[1] for res in reset_results]
        avail_list = [res[2] for res in reset_results]

        self.mac.init_hidden(batch_size=run_count)

        episode_batches = (
            [self._new_episode_batch() for _ in range(run_count)]
            if collect_batches
            else None
        )
        episode_returns = [0.0 for _ in range(run_count)]
        episode_lengths = [0 for _ in range(run_count)]
        final_infos: list[dict[str, Any]] = [{} for _ in range(run_count)]
        terminated = [False for _ in range(run_count)]

        for t in range(self.episode_limit):
            active_envs = [idx for idx in range(run_count) if not terminated[idx]]
            if not active_envs:
                break

            obs_batch = np.array(obs_list, dtype=np.float32)
            avail_batch: np.ndarray = np.zeros(
                (run_count, self.n_agents, self.n_actions), dtype=np.float32
            )
            for env_idx in active_envs:
                avail_batch[env_idx] = np.array(avail_list[env_idx], dtype=np.float32)

            chosen_actions_batch = self.mac.select_actions_batch(
                obs_batch=obs_batch,
                avail_actions_batch=avail_batch,
                epsilon=epsilon,
                test_mode=test_mode,
                active_envs=active_envs,
            )

            for env_idx in active_envs:
                if episode_batches is not None:
                    episode_batches[env_idx]["obs"][t] = np.array(obs_list[env_idx], dtype=np.float32)
                    episode_batches[env_idx]["state"][t] = np.array(state_list[env_idx], dtype=np.float32)
                    episode_batches[env_idx]["avail_actions"][t] = np.array(
                        avail_list[env_idx], dtype=np.float32
                    )

                self.parent_conns[env_idx].send(("step", chosen_actions_batch[env_idx].tolist()))

            for env_idx in active_envs:
                reward, done, info, next_obs, next_state, next_avail_actions = (
                    self.parent_conns[env_idx].recv()
                )
                if episode_batches is not None:
                    episode_batches[env_idx]["actions"][t, :, 0] = np.array(
                        chosen_actions_batch[env_idx], dtype=np.int64
                    )
                    episode_batches[env_idx]["reward"][t, 0] = float(reward)
                    episode_batches[env_idx]["terminated"][t, 0] = float(done)
                    episode_batches[env_idx]["filled"][t, 0] = 1.0

                episode_returns[env_idx] += float(reward)
                episode_lengths[env_idx] = t + 1
                final_infos[env_idx] = info

                obs_list[env_idx] = next_obs
                state_list[env_idx] = next_state
                terminated[env_idx] = bool(done)
                if not done:
                    avail_list[env_idx] = next_avail_actions

        avail_actions_last: list[np.ndarray] = [
            np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
            for _ in range(run_count)
        ]
        for env_idx in range(run_count):
            if terminated[env_idx]:
                terminal_avail: np.ndarray = np.zeros(
                    (self.n_agents, self.n_actions), dtype=np.float32
                )
                terminal_avail[:, 0] = 1.0
                avail_actions_last[env_idx] = terminal_avail
            else:
                avail_actions_last[env_idx] = np.array(
                    avail_list[env_idx], dtype=np.float32
                )

        if episode_batches is not None:
            for env_idx in range(run_count):
                t_ep = episode_lengths[env_idx]
                episode_batches[env_idx]["obs"][t_ep] = np.array(
                    obs_list[env_idx], dtype=np.float32
                )
                episode_batches[env_idx]["state"][t_ep] = np.array(
                    state_list[env_idx], dtype=np.float32
                )
                episode_batches[env_idx]["avail_actions"][t_ep] = np.array(
                    avail_actions_last[env_idx], dtype=np.float32
                )

        stats_list = [
            {
                "episode_return": episode_returns[env_idx],
                "episode_length": episode_lengths[env_idx],
                "battle_won": bool(final_infos[env_idx].get("battle_won", False)),
                "episode_limit": bool(final_infos[env_idx].get("episode_limit", False)),
                "no_kill_timeout": bool(final_infos[env_idx].get("no_kill_timeout", False)),
                "episode_seed": final_infos[env_idx].get("episode_seed"),
            }
            for env_idx in range(run_count)
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
