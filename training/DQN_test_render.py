import argparse
import torch
import os
import json
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from environment.jackal_env import JackalEnv
from training.DQN_Test.network import QNetwork
from training.utils.device import get_device, print_device_info

def parse_args():
    parser = argparse.ArgumentParser(description="DQN render evaluation")

    parser.add_argument("--model-path", type=str, default="artifacts/checkpoints/dqn/dqn_model_final.pth")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="选择计算设备")

    return parser.parse_args()

def test_model(model_path, episodes=3, device_name="auto", video_dir="artifacts/videos/dqn_eval"):
    print(f"正在加载模型并准备录制视频: {model_path}")

    device = get_device(device_name)
    print_device_info(device)

    config_path = os.path.join(os.path.dirname(model_path), "dqn_run_config.json")
    run_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            run_config = json.load(f)
        print(f"检测到训练配置: {config_path}")
    
    # ==========================================
    # 1. 初始化环境 (开启视频录制)
    # ==========================================
    # 注意：这里的 auto_aim 必须与你训练时保存的模型保持一致！
    # 如果你在没有 GUI 的服务器上跑，保持 headless=True；
    # use_video=True 会自动将每一帧渲染并保存为 mp4 文件。
    env = JackalEnv(
        headless=True,
        use_video=True,
        video_dir=video_dir,
        auto_aim=run_config.get("auto_aim", True)
    )
    
    _, initial_state = env.reset()
    state_dim = initial_state.shape[0]
    action_dim = env.n_actions

    if run_config:
        expected_state_dim = run_config.get("state_dim")
        expected_action_dim = run_config.get("action_dim")
        if expected_state_dim is not None and state_dim != expected_state_dim:
            print(f"状态维度不匹配: env={state_dim}, model={expected_state_dim}")
            env.close()
            return
        if expected_action_dim is not None and action_dim != expected_action_dim:
            print(f"动作维度不匹配: env={action_dim}, model={expected_action_dim}")
            env.close()
            return
    
    # ==========================================
    # 2. 实例化网络并加载权重
    # ==========================================
    policy_net = QNetwork(state_dim, action_dim).to(device)
    
    # 使用 map_location 确保即使在没有 GPU 的机器上也能加载 GPU 训练出的模型
    if os.path.exists(model_path):
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        print("模型权重加载成功！")
    else:
        print(f"找不到模型文件: {model_path}，请检查路径。")
        return
        
    policy_net.eval()  # 设置为评估模式
    
    # ==========================================
    # 3. 开启测试循环
    # ==========================================
    for episode in range(1, episodes + 1):
        _, state = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        info = {}
        
        while not done:
            # 【核心差异】：完全抛弃探索，执行纯粹的最优策略 (Greedy)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()  # 永远选择 Q 值最大的动作
                
            _, next_state, reward, done, info = env.step([action])
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
        battle_result = "Win" if info.get("battle_won", False) else "Loss"
        print(f"测试局 {episode}/{episodes} | 存活步数: {step_count} | 总奖励: {episode_reward:.1f} | 结果: {battle_result}")

    # ==========================================
    # 4. 释放资源并保存视频
    # ==========================================
    env.close()
    print(f"\n测试完成！录制的视频已保存在 {os.path.abspath('artifacts/videos/dqn_eval')} 目录下。")

if __name__ == "__main__":
    args = parse_args()

    test_model(
        args.model_path,
        episodes=args.episodes,
        device_name=args.device
    )
