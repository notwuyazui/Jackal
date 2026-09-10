import torch
import json
import os
import argparse
import sys
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from environment.jackal_env import JackalEnv

# 导入我们刚刚写好的 DRQN 四大金刚
from training.DRQN_Test.network import DRQNNetwork
from training.DRQN_Test.agent import DRQNAgent
from training.DRQN_Test.buffer import EpisodeBuffer
from training.DRQN_Test.learner import DRQNLearner


def parse_args():
    parser = argparse.ArgumentParser(description="DRQN 1v1 training")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=6000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-buffer-episodes", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--burn-in", type=int, default=20)
    parser.add_argument("--learn-len", type=int, default=40)
    parser.add_argument("--train-every-steps", type=int, default=10)
    parser.add_argument("--updates-per-train", type=int, default=2)
    parser.add_argument("--target-update-freq", type=int, default=10)
    parser.add_argument("--fire-explore-bias", type=float, default=0.25)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", action="store_false", dest="headless")
    parser.add_argument("--auto-aim", action="store_true", default=False)
    parser.add_argument("--save-prefix", type=str, default="drqn")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()

def main():
    args = parse_args()
    from training.utils.device import (
        get_device,
        print_device_info
    )
    device = get_device(args.device)
    print_device_info(device)
    
    # ==========================================
    # 1. 实例化环境 (你可以随时把 n_enemies 改为 2 开启 1v2 挑战)
    # ==========================================
    env = JackalEnv(headless=args.headless, use_video=False, auto_aim=args.auto_aim)
    env.max_steps = args.max_steps
    _, initial_state = env.reset()
    state_dim = initial_state.shape[0]
    action_dim = env.n_actions
    hidden_dim = args.hidden_dim
    chassis_dim = 9
    turret_dim = 1 if env.auto_aim else 3
    fire_dim = 2
    fire_action_id = 9 if env.auto_aim else 27

    run_config = {
        "algo": "drqn",
        "auto_aim": env.auto_aim,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "factorized_action": True,
        "chassis_dim": chassis_dim,
        "turret_dim": turret_dim,
        "fire_dim": fire_dim,
        "hidden_dim": hidden_dim,
        "n_agents": env.n_agents,
        "n_enemies": env.n_enemies,
        "burn_in": args.burn_in,
        "learn_len": args.learn_len,
        "train_every_steps": args.train_every_steps,
        "updates_per_train": args.updates_per_train,
        "fire_action_id": fire_action_id,
        "fire_explore_bias": args.fire_explore_bias,
        "batch_size": args.batch_size,
        "episodes": args.episodes,
        "lr": args.lr,
        "tau": args.tau,
        "target_update_freq": args.target_update_freq,
        "gpu_id": args.gpu_id,
        "max_steps": args.max_steps,
        "min_buffer_episodes": args.min_buffer_episodes,
    }
    model_root_dir = "artifacts/checkpoints/drqn"
    run_dir = os.path.join(model_root_dir, args.save_prefix)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    run_config_path = os.path.join(run_dir, "run_config.json")
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    print(f"运行配置已保存: {run_config_path}")
    
    # ==========================================
    # 2. 实例化带 GRU 记忆的神经网络
    # ==========================================
    policy_net = DRQNNetwork(
        state_dim,
        action_dim,
        hidden_dim=hidden_dim,
        chassis_dim=chassis_dim,
        turret_dim=turret_dim,
        fire_dim=fire_dim,
    ).to(device)
    target_net = DRQNNetwork(
        state_dim,
        action_dim,
        hidden_dim=hidden_dim,
        chassis_dim=chassis_dim,
        turret_dim=turret_dim,
        fire_dim=fire_dim,
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # ==========================================
    # 3. 实例化 DRQN 组件 
    # ==========================================
    # 容量 2000 局 (如果显存吃紧，可以适当调小)
    buffer = EpisodeBuffer(capacity=2000) 
    agent = DRQNAgent(
        action_dim,
        policy_net,
        device,
        chassis_dim=run_config["chassis_dim"],
        turret_dim=run_config["turret_dim"],
        fire_action_id=run_config["fire_action_id"],
        fire_explore_bias=run_config["fire_explore_bias"],
    )
    learner = DRQNLearner(
        policy_net,
        target_net,
        device,
        lr=args.lr,
        tau=args.tau,
        chassis_dim=run_config["chassis_dim"],
        fire_action_id=run_config["fire_action_id"],
    )
    
    # DRQN 超参数：
    # 因为一条轨迹可能长达 500 步，Batch Size 不能像 DQN 那样设为 128
    # 设为 32 条轨迹 (32 * 500 = 16000 帧)，对显存和 BPTT 来说比较健康
    batch_size = args.batch_size
    num_episodes = args.episodes
    target_update_freq = args.target_update_freq
    burn_in = run_config["burn_in"]
    learn_len = run_config["learn_len"]
    train_every_steps = run_config["train_every_steps"]
    updates_per_train = run_config["updates_per_train"]
    
    # ==========================================
    # 4. 主干交互循环 
    # ==========================================
    pbar = tqdm(range(1, num_episodes + 1), desc="DRQN 训练进度", unit="ep")
    total_step_count = 0
    
    for episode in pbar:
        _, state = env.reset()
        episode_reward = 0
        episode_loss = 0
        step_count = 0
        done = False
        info = {}
        
        current_episode_trajectory = [] 
        
        # 【DRQN 核心 1】：开局获取全零的空白记忆
        hidden_state = agent.init_hidden()
        
        while not done:
            # 【DRQN 核心 2】：带着历史记忆做决策，并接收新记忆
            action, next_hidden_state = agent.select_action(state, hidden_state)
            
            _, next_state, reward, done, info = env.step([action])
            
            # 记录这一步的经验
            current_episode_trajectory.append((state, action, reward, next_state, float(done)))
            
            # 状态转移 & 记忆流转
            state = next_state
            hidden_state = next_hidden_state
            
            episode_reward += reward
            step_count += 1
            total_step_count += 1

            min_ready_episodes = max(args.min_buffer_episodes, batch_size)
            if len(buffer) >= min_ready_episodes and total_step_count % train_every_steps == 0:
                for _ in range(updates_per_train):
                    batch_data = buffer.sample_sequence_batch(batch_size, burn_in=burn_in, learn_len=learn_len)
                    loss = learner.train_step(batch_data, burn_in=burn_in)
                    episode_loss += loss
            
        # 【DRQN 核心 3】：游戏结束，将一整条完整的时序轨迹压入 Buffer
        buffer.push_episode(current_episode_trajectory)
        
        # 回合结束处理
        agent.decay_epsilon()
        if episode % target_update_freq == 0:
            learner.update_target_network()
            
        # ==========================================
        # 5. 动态更新进度条
        # ==========================================
        # 由于我们每个 Episode 只在最后统一 train 一次，Loss 就是那一批的 Loss
        avg_loss = episode_loss 
        battle_result = "Win" if info.get("battle_won", False) else "Loss"
        
        pbar.set_postfix({
            'Step': step_count,
            'Rwd': f"{episode_reward:.1f}",
            'Res': battle_result,
            'Eps': f"{agent.epsilon:.3f}",
            'Loss': f"{avg_loss:.4f}"
        })
        
        # 定期保存模型权重
        if episode % 100 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"ep{episode}.pth")
            torch.save(policy_net.state_dict(), ckpt_path)
            pbar.write(f"--> [检查点] 模型已保存至 {ckpt_path}")

    final_path = os.path.join(run_dir, "model_final.pth")
    torch.save(policy_net.state_dict(), final_path)
    print(f"\nDRQN 训练结束！最终模型已保存至 {final_path}")
    env.close()

if __name__ == "__main__":
    main()
