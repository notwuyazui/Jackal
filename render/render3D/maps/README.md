# 地图目录

地图的物理布局属于 `engine/scenarios`；本目录只存放对应的视觉描述和构建器。
`catalog/*.json` 是 Python 与浏览器共用的注册表。Python 按实际 `tile_size`、`tiles`、地图尺寸和车型匹配场景，不依赖配置文件名、队伍规模或固定的六个单位 ID。

当前场景：

| ID | 引擎状态 | 场景构建器 |
| --- | --- | --- |
| `border-verdant` | `border_map_3t_vs_3t`，960×640 | `verdant` |
| `border-small-verdant` | `border_small_2t_vs_2t`，640×512 | `verdant` |

外部程序只需从 `render.api` 调用 `list_maps()`、`load_map()`、`apply_map()` 或 `create_viewer()`。

## 增加地图

1. 在 `engine/scenarios/game_states` 增加并验证物理存档。物理障碍与出生位置由 engine 负责。
2. 在 `catalog` 添加 JSON：`id`、`name`、`builder`、`layout`（真实 tile_size/tiles）、`meters_per_pixel`、`unit_models`、可选 `default_state`。不要把坦克位置重复写进地图描述。
3. 复用已有布景时填写 `builder: "verdant"`；新地形样式在 `scenes/<builder>.ts` 导出 `prepare(context)`。浏览器自动发现构建器，无需修改测试入口。
4. 构建器调用 `../../src` 的公用材质、模型与布景模块，也可调用 `@cot` 资源。`build(frame)` 返回 `MapScene`，地图内部可有独立植被/道路算法；战斗区域内的视觉地面不能改变 engine 的平面移动。
5. 同一布局允许不同主题，但调用方必须明确传 `scene_id` 或 CLI 的 `--scene`，避免不确定地选取第一张图。
6. 添加地图匹配与出生位置回归，并检查实际 3D 显示。未知地图明确报错，不能自动替换成 Border 场景。

当前 `verdant` 构建器服务于平坦 Border 地图。带内部山体、河流或建筑物的战斗地图需要制作相应构建器，才能在视觉上与其物理布局对应。

## 坦克部署与坐标

`apply_map(world)` 根据世界的实时初始快照生成部署信息：

```python
from render.api import apply_map, load_map

world = load_map("border-small-verdant")
scene = apply_map(world)
print(scene["initial_units"])  # id/model/team/x/y/z/hull_yaw/turret_yaw
```

像素平面 `(x, y)` 映射为 `((x-width/2)*scale, 0, (y-height/2)*scale)`，朝向映射为 `pi-radians(angle)`。
随后每帧的实际姿态由 FrameSnapshot 提供。模型整体等比缩放，碰撞箱仍来自引擎；炮弹在炮塔高度的战斗平面显示。

新增车型在 `../models/catalog/<model>.json` 声明 `id` 和 `adapter`，并在 `../models/adapters/<adapter>.ts` 导出 `prepare()` 与 `create(parameters, team, manifest)`。适配器返回的对象需满足 `src/models.ts` 的 `Vehicle` 接口，负责原版模型的姿态与特效；在场景 JSON 的 `unit_models` 中声明 unit_type → 模型 ID。Python 与浏览器自动发现同一份注册表，不需要修改测试入口或 `src/models.ts`。`render.api.list_models()` 可查询已注册车型。

地图引用不存在的车型、车型缺少适配器等配置错误会在创建窗口前报错。浏览器场景加载或 WebGL 出错时，错误也会回传 Python，立即释放同步等待；不再一直停留在“正在连接战斗画面”。

## 资源所有权

`PreparedMap` 持有可复用的纹理和资源工厂；`MapScene` 持有本局的地面、草、布景与道路实体。
换局先调用 `MapScene.dispose()`；更换 builder 或关闭视图时再调用 `PreparedMap.dispose()`。
不要通过遍历整个世界再次释放已由组件释放的 geometry/material，也不要释放仍由同一构建器缓存的纹理。
