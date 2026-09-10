import torch
import os
import json
import argparse
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from environment.jackal_env import JackalEnv

# 导入 DRQN 网络
from training.DRQN_Test.network import DRQNNetwork
from training.DRQN_Test.action_factorization import compose_action

from training.utils.device import get_device, print_device_info


def parse_args():
    parser = argparse.ArgumentParser(description="DRQN render evaluation")
    parser.add_argument("--model-path", type=str, default="artifacts/checkpoints/drqn/drqn/model_final.pth")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--video-dir", type=str, default="artifacts/videos/drqn_eval")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="覆盖测试环境最大步数；若不传则优先读取 run_config.json 的 max_steps",
    )
    parser.add_argument(
        "--fixed-delta-time",
        type=float,
        default=None,
        help="覆盖测试环境 fixed_delta_time；若不传则优先读取 run_config.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="选择计算设备: auto/cpu/cuda/mps"
    )
    return parser.parse_args()


def resolve_run_config_path(model_path):
    # 新布局: artifacts/checkpoints/drqn/<run_name>/model_final.pth + run_config.json
    same_dir_config = os.path.join(os.path.dirname(model_path), "run_config.json")
    if os.path.exists(same_dir_config):
        return same_dir_config

    # 旧布局兼容: artifacts/checkpoints/drqn/<prefix>_model_final.pth + <prefix>_run_config.json
    base_name = os.path.basename(model_path)
    if base_name.endswith("_model_final.pth"):
        legacy_config_name = base_name.replace("_model_final.pth", "_run_config.json")
        legacy_config_path = os.path.join(os.path.dirname(model_path), legacy_config_name)
        if os.path.exists(legacy_config_path):
            return legacy_config_path

    if "_model_ep" in base_name:
        prefix = base_name.split("_model_ep", 1)[0]
        legacy_config_name = f"{prefix}_run_config.json"
        legacy_config_path = os.path.join(os.path.dirname(model_path), legacy_config_name)
        if os.path.exists(legacy_config_path):
            return legacy_config_path

    # 最后回退（历史脚本写死文件名）
    fallback_config = os.path.join(os.path.dirname(model_path), "drqn_run_config.json")
    if os.path.exists(fallback_config):
        return fallback_config
    return None

def test_drqn_model(
    model_path,
    episodes=3,
    video_dir="artifacts/videos/drqn_eval",
    max_steps=None,
    fixed_delta_time=None,
    device_name="auto"
):
    print(f"正在加载 DRQN 模型并准备录制视频: {model_path}")

    device = get_device(device_name)
    print_device_info(device)

    config_path = resolve_run_config_path(model_path)
    run_config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            run_config = json.load(f)
        print(f"检测到训练配置: {config_path}")
    
    # ==========================================
    # 1. 初始化环境 (开启视频录制)
    # ==========================================
    # 注意：这里的 auto_aim 和 n_enemies 必须与你训练该模型时完全一致！
    config_delta_time = run_config.get("fixed_delta_time", None)
    env_delta_time = fixed_delta_time if fixed_delta_time is not None else config_delta_time

    env_kwargs = {
        "headless": True,
        "use_video": True,
        "video_dir": video_dir,
        "auto_aim": run_config.get("auto_aim", False),
    }
    if env_delta_time is not None:
        env_kwargs["fixed_delta_time"] = env_delta_time

    env = JackalEnv(**env_kwargs)

    # 默认读取训练配置中的 max_steps，确保测试时长与训练一致。
    if max_steps is not None:
        env.max_steps = int(max_steps)
    elif run_config.get("max_steps") is not None:
        env.max_steps = int(run_config["max_steps"])

    print(f"测试环境配置: max_steps={env.max_steps}, delta_time={env.delta_time}")
    
    _, initial_state = env.reset()
    state_dim = initial_state.shape[0]
    action_dim = env.n_actions
    hidden_dim = run_config.get("hidden_dim", 128)
    chassis_dim = run_config.get("chassis_dim", 9)
    turret_dim = run_config.get("turret_dim", 3)
    fire_dim = run_config.get("fire_dim", 2)
    fire_action_id = run_config.get("fire_action_id", 27)

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
    # hidden_dim 必须与训练时保持一致（优先读取配置）
    policy_net = DRQNNetwork(
        state_dim,
        action_dim,
        hidden_dim=hidden_dim,
        chassis_dim=chassis_dim,
        turret_dim=turret_dim,
        fire_dim=fire_dim,
    ).to(device)
    
    if os.path.exists(model_path):
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        print("--> 模型权重加载成功！")
    else:
        print(f"--> 找不到模型文件: {model_path}，请检查路径。")
        return
        
    policy_net.eval()  # 设置为评估模式，关闭 Dropout/BatchNorm 等
    
    # ==========================================
    # 3. 开启测试循环 (带记忆流转的纯贪婪策略)
    # ==========================================
    for episode in range(1, episodes + 1):
        _, state = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        info = {}
        
        # 【DRQN 测试核心】：开局获取全零的空白记忆
        hidden_state = policy_net.init_hidden(batch_size=1, device=device)
        
        while not done:
            # 执行纯粹的最优策略 (Greedy)，不带任何 Epsilon 随机探索
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                q_chassis, q_turret, q_fire, hidden_state = policy_net(state_tensor, hidden_state)

                if run_config.get("factorized_action", True):
                    chassis_action = q_chassis.argmax(dim=1).item()
                    turret_action = q_turret.argmax(dim=1).item()
                    fire_action = q_fire.argmax(dim=1).item()
                    action = compose_action(
                        chassis_action,
                        turret_action,
                        fire_action,
                        chassis_dim=chassis_dim,
                        fire_action_id=fire_action_id,
                    )
                else:
                    raise RuntimeError("当前测试脚本仅支持 factorized_action=True 的 DRQN 权重。")
                
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
    print(f"\n测试完成！录制的视频已保存在 {os.path.abspath(video_dir)} 目录下。")

if __name__ == "__main__":
    args = parse_args()
    test_drqn_model(
        args.model_path,
        episodes=args.episodes,
        video_dir=args.video_dir,
        max_steps=args.max_steps,
        fixed_delta_time=args.fixed_delta_time,
        device_name=args.device,
    )
