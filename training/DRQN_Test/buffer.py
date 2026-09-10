import numpy as np
import random
from collections import deque
from typing import Any, Deque

class EpisodeBuffer:
    def __init__(self, capacity=2000):
        """
        容量限制：这里存放的是 2000 '局' 完整的游戏轨迹，
        而不是 2000 '帧' 散乱的画面。
        """
        self.buffer: Deque[Any] = deque(maxlen=capacity)

    def push_episode(self, episode):
        """
        存入一整条轨迹。
        episode 是一个列表，包含了从开局到结束按时间顺序排列的 transition 组：
        [(state, action, reward, next_state, done), ...]
        """
        self.buffer.append(episode)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        """
        核心方法：抽取批量轨迹并进行对齐 (Padding) 与掩码生成 (Masking)
        返回的所有 numpy 数组形状均为 3D: (Batch_Size, Max_Sequence_Length, Feature_Dim)
        """
        # 1. 随机采样 batch_size 条完整的轨迹
        sampled_episodes = random.sample(self.buffer, batch_size)
        
        # 2. 探明该批次数据中的“最长轨迹”，以此作为对齐标尺
        seq_lens = [len(ep) for ep in sampled_episodes]
        max_len = max(seq_lens)
        
        # 3. 动态获取状态维度 (探测第一条轨迹的第一帧状态)
        state_dim = len(sampled_episodes[0][0][0])
        
        # 4. 初始化全零的 Numpy 矩阵 (这天然就完成了 Padding 补零的动作)
        padded_states = np.zeros((batch_size, max_len, state_dim), dtype=np.float32)
        padded_actions = np.zeros((batch_size, max_len, 1), dtype=np.int64)
        padded_rewards = np.zeros((batch_size, max_len, 1), dtype=np.float32)
        padded_next_states = np.zeros((batch_size, max_len, state_dim), dtype=np.float32)
        padded_dones = np.zeros((batch_size, max_len, 1), dtype=np.float32)
        
        # 【极其重要】：掩码矩阵 Mask
        # 用于在后续计算 Loss 时，将那些“为了凑齐长度而补零的无效步”屏蔽掉
        masks = np.zeros((batch_size, max_len, 1), dtype=np.float32)
        
        # 5. 遍历采样出的轨迹，将真实的经验填入对应的槽位中
        for b, ep in enumerate(sampled_episodes):
            ep_len = len(ep) # 这条轨迹的真实长度
            for t in range(ep_len):
                state, action, reward, next_state, done = ep[t]
                
                padded_states[b, t] = state
                padded_actions[b, t] = action
                padded_rewards[b, t] = reward
                padded_next_states[b, t] = next_state
                padded_dones[b, t] = done
                
                # 真实的步数为 1.0；超过 ep_len 的部分由于初始化是 zeros，自然就是 0.0
                masks[b, t] = 1.0 
                
        return padded_states, padded_actions, padded_rewards, padded_next_states, padded_dones, masks

    def sample_sequence_batch(self, batch_size, burn_in, learn_len):
        """
        Truncated BPTT 采样：
        - 每条轨迹随机截取长度为 (burn_in + learn_len) 的窗口
        - 窗口不足长度时尾部补零并用 mask 标记无效步
        """
        total_len = burn_in + learn_len
        sampled_episodes = random.sample(self.buffer, batch_size)

        state_dim = len(sampled_episodes[0][0][0])

        padded_states = np.zeros((batch_size, total_len, state_dim), dtype=np.float32)
        padded_actions = np.zeros((batch_size, total_len, 1), dtype=np.int64)
        padded_rewards = np.zeros((batch_size, total_len, 1), dtype=np.float32)
        padded_next_states = np.zeros((batch_size, total_len, state_dim), dtype=np.float32)
        padded_dones = np.zeros((batch_size, total_len, 1), dtype=np.float32)
        masks = np.zeros((batch_size, total_len, 1), dtype=np.float32)

        for batch_index, episode in enumerate(sampled_episodes):
            episode_len = len(episode)
            if episode_len <= 0:
                continue

            max_start = max(0, episode_len - total_len)
            start = random.randint(0, max_start)
            end = min(start + total_len, episode_len)
            window = episode[start:end]

            for time_index, transition in enumerate(window):
                state, action, reward, next_state, done = transition
                padded_states[batch_index, time_index] = state
                padded_actions[batch_index, time_index] = action
                padded_rewards[batch_index, time_index] = reward
                padded_next_states[batch_index, time_index] = next_state
                padded_dones[batch_index, time_index] = done
                masks[batch_index, time_index] = 1.0

        return padded_states, padded_actions, padded_rewards, padded_next_states, padded_dones, masks
