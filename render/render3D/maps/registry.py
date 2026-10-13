"""One JSON catalog shared by Python validation and the browser scene registry.

Scene selection depends on the actual tile layout, never a configuration filename
or the number of tanks. Applying a scene only computes visual placements; it does
not spawn units, move them or change engine collision/weapon parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING

from engine.contracts.render import BattleManifest, FrameSnapshot
from render.render3D.models.registry import list_models

if TYPE_CHECKING:
    from engine.battle_world import BattleWorld

CATALOG = Path(__file__).with_name("catalog")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SceneDefinition:
    id: str
    name: str
    builder: str
    tile_size: int
    tiles: tuple[str, ...]
    meters_per_pixel: float
    unit_models: tuple[tuple[str, str], ...]
    default_state: str | None

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "builder": self.builder,
                "meters_per_pixel": self.meters_per_pixel,
                "unit_models": dict(self.unit_models)}


def list_maps(catalog: Path = CATALOG) -> tuple[SceneDefinition, ...]:
    """Read scene descriptors. Adding a descriptor needs no CLI changes."""
    scenes = []
    ids = set()
    available_models = set(list_models())
    for path in sorted(Path(catalog).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            layout = data["layout"]
            tile_size, tiles = layout["tile_size"], layout["tiles"]
            scale = float(data.get("meters_per_pixel", .25))
            models = data["unit_models"]
            if (type(tile_size) is not int or tile_size <= 0
                    or not isinstance(tiles, list) or not tiles
                    or any(not isinstance(row, str) or not row for row in tiles)
                    or len({len(row) for row in tiles}) != 1
                    or not math.isfinite(scale) or scale <= 0
                    or not isinstance(models, dict) or not models
                    or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
                           for k, v in models.items())):
                raise ValueError("invalid layout, scale or unit_models")
            missing_models = set(models.values()) - available_models
            if missing_models:
                raise ValueError(f"未注册的 3D 模型：{sorted(missing_models)}")
            if not isinstance(data['name'], str) or not data['name'].strip():
                raise ValueError('name must be a nonempty string')
            if data.get('default_state') is not None and not isinstance(data['default_state'], str):
                raise ValueError('default_state must be a path string')
            for key in ("id", "builder"):
                if not re.fullmatch(r"[a-z][a-z0-9_-]*", data[key]):
                    raise ValueError(f"invalid {key}")
            if not (CATALOG.parent / "scenes" / f"{data['builder']}.ts").is_file():
                raise ValueError(f"scene builder does not exist: {data['builder']}")
            if data["id"] in ids:
                raise ValueError(f"duplicate scene id: {data['id']}")
            ids.add(data["id"])
            scenes.append(SceneDefinition(data["id"], data["name"], data["builder"],
                tile_size, tuple(tiles), scale, tuple(models.items()), data.get("default_state")))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"3D 地图描述无效：{path}: {exc}") from exc
    return tuple(scenes)


def resolve_scene(manifest: BattleManifest, scene_id: str | None = None,
                  *, catalog: Path = CATALOG) -> SceneDefinition:
    """Validate a scene against the immutable engine manifest."""
    scenes = list_maps(catalog)
    matches = [s for s in scenes if (scene_id is None or s.id == scene_id)
               and s.tile_size == manifest.map.tile_size and s.tiles == manifest.map.tiles]
    if not matches:
        available = ", ".join(s.id for s in scenes) or "无"
        raise ValueError(f"当前地图还没有匹配的 3D 场景（已注册：{available}）。"
                         "请添加地图描述和场景构建器，或使用 --render 2D")
    if len(matches) > 1:
        raise ValueError("该地图匹配多个 3D 场景，请通过 scene_id 明确选择")
    scene = matches[0]
    expected_size = (len(scene.tiles[0]) * scene.tile_size, len(scene.tiles) * scene.tile_size)
    if manifest.map.size != expected_size:
        raise ValueError("地图尺寸与 tile 布局不一致")
    missing = {u.unit_type for u in manifest.units} - dict(scene.unit_models).keys()
    if missing:
        raise ValueError(f"场景 {scene.id} 未配置车型 {sorted(missing)} 的 3D 模型；请使用 --render 2D")
    if not manifest.units:
        raise ValueError("3D 场景需要至少一个单位")
    if len({u.id for u in manifest.units}) != len(manifest.units):
        raise ValueError("单位 ID 必须唯一")
    return scene


def apply_map(world: BattleWorld, scene_id: str | None = None, *, dt: float = .03) -> dict:
    """Apply a registered visual map and place every tank at its engine pose.

    Returns scene metadata and initial unit placements in metres/radians. The
    original BattleWorld is unchanged; subsequent frames remain authoritative.
    """
    manifest = world.render_manifest(dt)
    return deployment(manifest, world.render_snapshot(), scene_id)


def deployment(manifest: BattleManifest, frame: FrameSnapshot,
               scene_id: str | None = None) -> dict:
    scene = resolve_scene(manifest, scene_id)
    if (frame.map != manifest.map or len(frame.units) != len(manifest.units)
            or {(u.id, u.unit_type) for u in frame.units} != {(u.id, u.unit_type) for u in manifest.units}):
        raise ValueError("地图部署的快照与 manifest 不一致")
    width, height = manifest.map.size
    scale = scene.meters_per_pixel
    models = dict(scene.unit_models)
    placements = [{"id": u.id, "model": models[u.unit_type], "team": u.team,
                   "x": (u.position[0] - width / 2) * scale,
                   "y": 0., "z": (u.position[1] - height / 2) * scale,
                   "hull_yaw": math.pi - math.radians(u.heading),
                   "turret_yaw": math.pi - math.radians(u.turret_heading)} for u in frame.units]
    return {**scene.to_dict(), "initial_units": placements}


def load_map(scene_id: str, *, state_file: str | Path | None = None, **load_options) -> BattleWorld:
    """Load the scene's default engine state (or an explicit state) and validate it."""
    from engine.battle_world import BattleWorld
    scene = next((s for s in list_maps() if s.id == scene_id), None)
    if scene is None:
        raise ValueError(f"未注册的 3D 地图：{scene_id}")
    if state_file is None:
        if not scene.default_state:
            raise ValueError(f"地图 {scene_id} 未设置 default_state，请提供 state_file")
        state_file = (PROJECT_ROOT / scene.default_state).resolve()
        if not state_file.is_relative_to(PROJECT_ROOT / "engine/scenarios"):
            raise ValueError("default_state 必须位于 engine/scenarios 中")
    world = BattleWorld()
    world.load_game_state(state_file, **load_options)
    apply_map(world, scene_id)
    return world
