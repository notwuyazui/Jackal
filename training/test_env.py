"""Short headless smoke run for the multi-agent environment."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.jackal_env import JackalEnv


def test_marl_env():
    print("初始化 MARL 标准环境测试...")
    env = JackalEnv(headless=True, fixed_delta_time=0.1, use_video=False,
                   video_dir="artifacts/videos/jackal_records")
    try:
        obs, state = env.reset()
        print(f"动作空间大小: {env.n_actions}")
        print(f"局部观测(obs[0])维度: {obs[0].shape[0]} 维")
        print(f"全局状态(state)维度: {state.shape[0]} 维")
        print("开始模拟运行 20 步...\n")

        for step in range(20):
            # Every controlled agent receives the same idle action.
            obs, state, _, done, _ = env.step([0] * env.n_agents)
            if done:
                print(f"\n>>> [Step {step}] Episode 结束！<<<")
                break
            if step % 10 == 0:
                active_bullets = env.get_runtime_info()["active_bullets"]
                print(f"\n--- [Step {step}] 状态报告 (场上子弹: {active_bullets}) ---")
                print(f"局部观测 (obs[0]): {obs[0]}")
                print(f"全局状态 (state): {state}")
        print("\n测试完成。")
    finally:
        env.close()


if __name__ == "__main__":
    test_marl_env()
