"""Verify embedded Jackal Pygame frames and the CoT browser render receipts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def verify(directory=ROOT / "artifacts/demo"):
    frames = [json.loads(line) for line in (directory / "last-run.jsonl").read_text().splitlines()]
    receipts = [json.loads(line) for line in (directory / "view-receipts.jsonl").read_text().splitlines()]
    jackal_images = {item["tick"]: item["png_sha256"] for item in receipts if item["view"] == "2d"}
    report = json.loads((directory / "last-result.json").read_text())
    by_tick = {frame["tick"]: frame for frame in frames}
    expected_effects = {}
    previous_alive = {unit["id"]: unit["alive"] for unit in frames[0]["units"]}
    total_shots = total_deaths = 0
    shots_by_unit = {unit["id"]: [] for unit in frames[0]["units"]}
    cooldowns = {unit["id"]: unit["cooldown"] for unit in report["manifest"]["units"]}
    effect_captures = {}
    for frame in frames:
        total_shots += sum(event["kind"] == "fire" for event in frame["events"])
        for event in frame["events"]:
            if event["kind"] == "fire": shots_by_unit[event["source_id"]].append(frame["time"])
        for unit in frame["units"]:
            assert unit["cooldown_duration"] == cooldowns[unit["id"]], "manifest cooldown differs from actual unit timing"
        total_deaths += sum(previous_alive[unit["id"]] and not unit["alive"]
                            for unit in frame["units"])
        previous_alive = {unit["id"]: unit["alive"] for unit in frame["units"]}
        expected_effects[frame["tick"]] = (total_shots, total_deaths)
    assert list(by_tick) == list(range(frames[0]["tick"], frames[-1]["tick"] + 1)), "missing physics step"
    for frame in frames:
        original = {key: value for key, value in frame.items() if key != "hash"}
        digest = hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()[:16]
        assert frame["hash"] == digest, ("corrupted frame", frame["tick"])
    seen = {"2d": set(), "3d": set()}
    followed_id = report["manifest"]["follow_id"]
    maxima = {"position_px": 0., "heading_deg": 0., "turret_deg": 0.,
              "collider_px": 0., "model_scale": 0., "model_aspect": 0.,
              "combat_plane_m": 0., "vertical_speed_mps": 0.,
              "bullet_position_px": 0., "camera_anchor_px": 0.,
              "camera_heading_deg": 0., "camera_distance_m": 0.,
              "camera_pitch_deg": 0., "camera_fov_deg": 0.}
    embedded_frames = 0
    divergent_camera_ticks = set()
    plane_heights = []
    shell_frames = set()
    visual_sizes = {}
    max_start_bob = max_fire_pitch = 0.0
    max_crushed_props = 0
    max_native_recoil = 0.0
    max_animation_frames = 0

    def error(name, value):
        assert math.isfinite(value) and value <= 1e-4, (name, value)
        maxima[name] = max(maxima[name], value)

    for receipt in receipts:
        frame = by_tick[receipt["tick"]]
        assert receipt["hash"] == frame["hash"], ("wrong frame displayed", receipt["view"], receipt["tick"])
        seen[receipt["view"]].add(frame["tick"])
        if receipt["view"] == "2d":
            assert receipt["display_mode"] == "pygame-offscreen"
            assert receipt["window_open"] is False
            assert receipt["surface_size"] == report["manifest"]["map"]["size"]
            assert len(receipt["png_sha256"]) == 64
            continue
        inset = receipt["jackal_inset"]
        assert inset["mode"] == "original-pygame-frame"
        assert inset["tick"] == frame["tick"]
        assert inset["size"] == report["manifest"]["map"]["size"]
        assert len(inset["png_sha256"]) == 64
        assert inset["png_sha256"] == jackal_images[frame["tick"]]
        assert inset["presentation"] == "atomic-canvas-copy"
        embedded_frames += 1
        assert receipt["scenery"]["model"] == "cot-native-m1a1"
        assert receipt["scenery"]["scene_revision"] == "verdant-native-continuous-v7"
        assert receipt["scenery"]["ground"] == "cot-grass004"
        assert receipt["scenery"]["grass"]["near"]["count"] > 0
        assert receipt["scenery"]["shell"] == "cot-native-he-tracer"
        assert receipt["scenery"]["shell_textures"] == "cot-original-particle-atlases"
        assert receipt["scenery"]["collider"] == "jackal-axis-aligned-cuboid"
        assert receipt["scenery"]["boundary"] == "transparent-air-barrier"
        assert all(receipt["scenery"]["exterior"].get(key) == value for key, value in {"trees": 3450, "hillside_trees": 1600,
            "terrain_layers": 5, "rocks": 210, "houses": 12,
            "poles": 94, "hay": 46, "mountains": 5, "barrier_panels": 4,
            "boundary_fences": 89, "boundary_rocks": 26, "boundary_houses": 2,
            "mountain_style": "cot-verdant-native-vista", "mountain_relief_bake": True, "far_range": True}.items())
        assert receipt["scenery"]["exterior"]["mountain_rows"] >= 18
        assert receipt["scenery"]["exterior"]["mountain_forest"] >= 1000
        assert receipt["scenery"]["camouflage"] == {"PLAYER": "merdc", "ENEMY": "desert"}
        assert receipt["scenery"]["sky"] == "cot-baked-cumulus-verdant"
        assert receipt["scenery"]["lighting"] == "diffuse-soil-sun-entity-shadows"
        assert receipt["scenery"]["ground_dressing"]["paths"] == 3
        assert receipt["scenery"]["ground_dressing"]["patches"] >= 500
        assert receipt["scenery"]["ground_dressing"]["tufts"] >= 37000
        assert receipt["scenery"]["ground_dressing"]["bushes"] == 420
        assert receipt["scenery"]["ground_dressing"]["road_style"] == "cot-verdant-ground071-cut-ruts-v5"
        assert receipt["scenery"]["ground_dressing"]["road_grass_clearance_m"] == 0.55
        assert receipt["scenery"]["roadside"]["cars"] >= 5
        assert receipt["scenery"]["roadside"]["fuel_drums"] >= 8
        assert receipt["scenery"]["roadside"]["crushed"] >= max_crushed_props
        max_crushed_props = receipt["scenery"]["roadside"]["crushed"]
        assert receipt["scenery"]["effects"]["muzzle_style"] == "cot-native-continuous-200mm-visual"
        assert receipt["scenery"]["effects"]["wreck_fade"] == "cot-burn-then-alpha-3.55s"
        effects = receipt["scenery"]["effects"]
        assert effects["soft_depth_bound"] is True
        assert receipt["performance"]["world_passes_per_frame"] == 1
        max_animation_frames = max(max_animation_frames, effects["animation_frames"])
        max_native_recoil = max(max_native_recoil, effects["animation_maxima"]["recoil_m"])
        for capture in receipt.get("effect_captures", []): effect_captures[capture["name"]] = capture
        assert (effects["muzzle_flashes"], effects["destructions"]) == expected_effects[frame["tick"]]
        plane = receipt["scenery"]["battle_plane_height_m"]
        assert math.isfinite(plane) and plane > 0
        plane_heights.append(plane)
        rendered = {unit["id"]: unit for unit in receipt["rendered"]}
        assert set(rendered) == {unit["id"] for unit in frame["units"]}
        for unit in frame["units"]:
            actual = rendered[unit["id"]]
            error("position_px", math.dist((actual["x"], actual["y"]), unit["position"]))
            error("heading_deg", abs((actual["heading"] - unit["heading"] + 180) % 360 - 180))
            error("turret_deg", abs((actual["turret_heading"] - unit["turret_heading"] + 180) % 360 - 180))
            for key in ("hitbox", "collision_box"):
                assert len(actual[key]) == 4
                error("collider_px", max(abs(a-b) for a, b in zip(actual[key], unit[key])))
            assert len(actual["model_scale"]) == len(actual["native_body_size_m"]) == len(actual["body_size_m"]) == 3
            assert min(actual["model_scale"]) > 0
            error("model_scale", max(actual["model_scale"])-min(actual["model_scale"]))
            ratios = [a/b for a, b in zip(actual["body_size_m"], actual["native_body_size_m"])]
            error("model_aspect", max(ratios)-min(ratios))
            error("combat_plane_m", abs(actual["muzzle_height_m"]-plane))
            error("vertical_speed_mps", abs(actual["gun_vertical_component"]))
            assert actual["collider_y"][0] < plane < actual["collider_y"][1]
            visual_sizes[unit["id"]] = {"scale": actual["model_scale"], "body_size_m": actual["body_size_m"]}
            assert (actual["health"], actual["alive"]) == (unit["health"], unit["alive"])
            assert actual["collider_active"] == unit["alive"], ("dead collider", unit["id"])
            assert 0 <= actual["visual_opacity"] <= 1
            assert math.isfinite(actual["cosmetic_bob_m"]) and abs(actual["cosmetic_bob_m"]) <= 0.25
            assert math.isfinite(actual["cosmetic_pitch_deg"]) and abs(actual["cosmetic_pitch_deg"]) <= 8
            if actual["start_bounce_age_s"] is not None and 0 < actual["start_bounce_age_s"] < 0.3:
                max_start_bob = max(max_start_bob, abs(actual["cosmetic_bob_m"]))
            if actual["recoil_age_s"] is not None and actual["recoil_age_s"] < 0.3:
                max_fire_pitch = max(max_fire_pitch, abs(actual["cosmetic_pitch_deg"]))
            if frame["result"]["finished"] and not unit["alive"]:
                assert not actual["visible"] and actual["visual_opacity"] == 0
        bullets = {bullet["id"]: bullet for bullet in receipt["bullets"]}
        assert set(bullets) == {bullet["id"] for bullet in frame["bullets"]}
        for bullet in frame["bullets"]:
            actual = bullets[bullet["id"]]
            error("bullet_position_px", math.dist((actual["x"], actual["y"]), bullet["position"]))
            assert actual["tracer"] == "HE"
            error("combat_plane_m", max(abs(actual["height_m"]-plane), abs(actual["tail_height_m"]-plane)))
            error("vertical_speed_mps", abs(actual["vertical_speed_mps"]))
            shell_frames.add(frame["tick"])
        camera = receipt["camera"]
        assert camera["follow_id"] == followed_id
        assert camera["mode"] == "turret-follow"
        error("camera_distance_m", abs(camera["orbit_distance_m"] - 13))
        error("camera_pitch_deg", abs(camera["orbit_pitch_deg"] - 10))
        error("camera_fov_deg", abs(camera["fov_deg"] - 60))
        unit = next(unit for unit in frame["units"] if unit["id"] == followed_id)
        error("camera_anchor_px", math.dist((camera["anchor_x"], camera["anchor_y"]), unit["position"]))
        error("camera_heading_deg", abs((camera["heading"] - unit["turret_heading"] + 180) % 360 - 180))
        if abs((unit["heading"] - unit["turret_heading"] + 180) % 360 - 180) > 1:
            divergent_camera_ticks.add(frame["tick"])
    for view, ticks in seen.items():
        assert ticks == set(by_tick), ("unconfirmed steps", view, sorted(set(by_tick) - ticks))
    assert report["both_views_confirmed_final"]
    assert not report["validation"]["errors"]
    assert frames[-1]["result"]["finished"]
    assert report["manifest"]["max_steps"] == 6000
    assert frames[-1]["result"]["reason"] != "timeout", "fight still ended at the short training horizon"
    assert report["final"]["hash"] == frames[-1]["hash"]
    error("combat_plane_m", max(plane_heights)-min(plane_heights))
    assert shell_frames, "no original CoT shell instances verified"
    assert max_start_bob > 0.03, "startup chassis bounce was not visible"
    assert max_fire_pitch > 2.0, "firing chassis shudder was not visible"
    assert max_crushed_props >= 2, "roadside car and drum were not driven over"
    assert max_native_recoil > .25, "original 3D barrel did not recoil"
    assert max_animation_frames > frames[-1]["tick"], "continuous visual animation did not run"
    assert set(effect_captures) == {"muzzle-peak", "muzzle-recovery", "muzzle-smoke"}
    assert effect_captures["muzzle-peak"]["flash_particles"] > 0
    assert effect_captures["muzzle-peak"]["recoil_m"] > .2
    assert effect_captures["muzzle-smoke"]["smoke_particles"] > 0
    assert all(capture["soft_depth_bound"] for capture in effect_captures.values())
    for unit_id, times in shots_by_unit.items():
        assert len(times) > 1
        assert min(b-a for a,b in zip(times,times[1:])) >= cooldowns[unit_id] - 1e-8, unit_id
    for name in effect_captures:
        assert (directory / (name + "-3d.png")).read_bytes().startswith(b"\x89PNG")
    state_file = ROOT.parent / report["manifest"]["state_file"]
    assert hashlib.sha256(state_file.read_bytes()).hexdigest() == report["manifest"]["state_sha256"]
    summary = {
        "passed": True, "confirmed_steps": {view: len(ticks) for view, ticks in seen.items()},
        "final_tick": frames[-1]["tick"], "simulation_seconds": round(frames[-1]["time"], 6),
        "final_hash": frames[-1]["hash"], "result": frames[-1]["result"],
        "followed_id": followed_id, "max_errors": maxima,
        "presentation": report["manifest"]["presentation"],
        "embedded_jackal_frames": embedded_frames,
        "turret_camera_frames_diverging_from_hull": len(divergent_camera_ticks),
        "combat_plane_height_m": plane_heights[0], "original_he_tracer_frames": len(shell_frames),
        "muzzle_flashes": total_shots, "destructions": total_deaths,
        "max_start_bob_m": max_start_bob, "max_fire_pitch_deg": max_fire_pitch,
        "crushed_roadside_props": max_crushed_props,
        "continuous_animation_frames": max_animation_frames, "max_native_barrel_recoil_m": max_native_recoil,
        "fire_cooldown_s": cooldowns, "shots_by_unit": {k:len(v) for k,v in shots_by_unit.items()},
        "firing_captures": effect_captures, "mountains": receipts[-1]["scenery"]["exterior"],
        "visual_models": visual_sizes,
        "performance": receipts[-1]["performance"],
        "source_state_sha256": report["manifest"]["state_sha256"],
    }
    (directory / "consistency-summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
