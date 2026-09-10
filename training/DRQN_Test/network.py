import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from training.DRQN_Test.action_factorization import CHASSIS_DIM, TURRET_DIM, FIRE_DIM

class DRQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, chassis_dim=CHASSIS_DIM, turret_dim=TURRET_DIM, fire_dim=FIRE_DIM):
        super(DRQNNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.chassis_dim = chassis_dim
        self.turret_dim = turret_dim
        self.fire_dim = fire_dim

        # 1. 特征提取层 (将几十维的状态升维到隐藏层空间)
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        # 2. 核心记忆层：GRUCell 
        # 它接收 fc1 的输出特征，以及上一个时间步的隐状态 (Hidden State)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

        # 3. 动作因子化三头输出
        self.chassis_head = nn.Linear(hidden_dim, self.chassis_dim)
        self.turret_head = nn.Linear(hidden_dim, self.turret_dim)
        self.fire_head = nn.Linear(hidden_dim, self.fire_dim)

    def init_hidden(
        self,
        batch_size: int = 1,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        初始化全零的隐状态 (Hidden State)。
        在每次新开一局游戏 (Episode) 时，或者在 Learner 开始处理一个新 Batch 时调用。
        """
        # 返回形状为 (Batch_Size, Hidden_Dim) 的零张量
        return torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32, device=device)

    def forward(self, x, hidden_state):
        """
        前向传播
        :param x: 当前的输入状态，形状 (Batch_Size, State_Dim)
        :param hidden_state: 上一刻的隐状态，形状 (Batch_Size, Hidden_Dim)
        :return: 三个头的 Q 值和更新后的隐状态
        """
        # 1. 提取当前状态的特征
        x = F.relu(self.fc1(x))
        
        # 2. 将特征和历史记忆送入 GRU 单元，得到新的记忆
        # 注意：GRUCell 的输入必须是 2D 张量 (Batch_Size, Feature_Dim)
        h_out = self.rnn(x, hidden_state)
        
        q_chassis = self.chassis_head(h_out)
        q_turret = self.turret_head(h_out)
        q_fire = self.fire_head(h_out)

        return q_chassis, q_turret, q_fire, h_out
