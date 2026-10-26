### 快速开始
1. 通过键鼠控制运行一局游戏
    python training/run_game_keyboard.py
    目前键鼠控制游戏仅支持2D渲染模式。
2. 通过模型/AI控制运行一局游戏
    python training/run_game_ai.py
    参数：
        --render：渲染方式；可选2D/3D/2D3D，默认为2D3D
        --config：配置名称或JSON路径；使用游戏内置AI控制时从engine/scenarios/game_states读取，使用模型控制时从training/configs/marl2读取；默认为border_map_3v3
        --checkpoint：模型路径artifacts/checkpoints/[model_path]/[checkpoint_name].pt；该值为空时使用游戏内置5级AI控制
        --seed：随机种子；模型控制时默认沿用训练配置，AI控制时默认为42
        --max-steps：单局步数上限；模型控制时默认沿用训练配置，AI控制时默认为6000
        --ai-intelligence：AI控制时使用的AI等级，可选1-9；模型控制时默认为5
        --enemy-intelligence：覆盖敌方AI等级；模型控制时默认沿用训练配置，AI对战默认与己方相同
        --speed：倍速播放；默认为1
        --follow-id：在3D渲染模式下跟随的玩家ID；默认为2
        --port：本地数据服务窗口；默认自动选择
        --view-port：在3D渲染模式下时浏览器服务端口；默认自动选择
    目前只有border_map_3v3支持3D渲染模式
3. 训练模型
    python train_qmix_marl2.py
    python DQN_train.py
    python DRQN_train.py
    参数：
        --device：训练设备；可选cpu/cuda/mps/npu/auto；默认自动选择，目前不支持多卡训练
        ...
4. 测试模型
    python test_model.py --checkpoint artifacts/checkpoints/[model_path]/[checkpoint_name].pt
    参数：
        --checkpoint：模型路径；根据权重结构自动选择网络，该参数为必填项
        --device：训练设备；可选cpu/cuda/mps/npu/auto；默认自动选择，目前不支持多卡训练
        --config：训练配置JSON；默认查找检查点旁的运行配置，QMIX回退到标准3v3配置
        --episodes：测试局数；默认为10
        --max-steps：单局步数上限；默认沿用训练配置中的评估步数上限
        --seed-base：逐局种子起点，默认配置seed+100000
    
