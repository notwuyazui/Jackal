import os
import sys
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import random
import numpy as np
from environment.jackal_env import JackalEnv

def test_marl_env():
    print("初始化 MARL 标准环境测试 (附带视频录制)...")
    
    env = JackalEnv(headless=True, fixed_delta_time=0.1, use_video=False, video_dir="artifacts/videos/jackal_records")
    obs, state = env.reset()
    
    np.set_printoptions(precision=3, suppress=True)
    
    print(f"动作空间大小: {env.n_actions}")
    print(f"局部观测(obs[0])维度: {obs[0].shape[0]} 维")
    print(f"全局状态(state)维度:  {state.shape[0]} 维")
    print("开始模拟运行 300 步...\n")
    
    for step in range(20):
        actions = [0]  # 默认动作：不移动也不转向，挨打测试

        obs, state, reward, done, info = env.step(actions)
        if done:
            print(f"\n>>> [Step {step}] Episode 结束！<<<")
            break
            
        if step % 10 == 0:
            print(f"\n--- [Step {step}] 状态报告 (场上子弹: {len(env.bullet_manager.bullets)}) ---")
            print(f"局部观测 (obs[0]): {obs[0]}")
            print(f"全局状态 (state): {state}")
            # for agent in env.agents:
            #     info_agent = agent.get_info()
            #     print(f"  [玩家] 位置: ({info_agent['position'][0]:.1f}, {info_agent['position'][1]:.1f}) | 朝向: {info_agent['direction']:>5.1f} | 血量: {info_agent['health']}")
                
            # for enemy in env.enemies:
            #     info_enemy = enemy.get_info()
            #     print(f"  [敌军] 位置: ({info_enemy['position'][0]:.1f}, {info_enemy['position'][1]:.1f}) | 血量: {info_enemy['health']}")
    
    print("\n测试完成。正在保存录像文件...")
    env.close()  
    print("录像保存成功！请查看 'artifacts/videos/jackal_records' 文件夹。")

if __name__ == "__main__":
    test_marl_env()
