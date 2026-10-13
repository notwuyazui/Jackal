"""Optional strict synchronized renderer diagnostics. No simulation or draw calls."""
import base64
import math
import json

class RenderDiagnostics:
    def acknowledge(self, payload):
        view = payload.get("view")
        if view not in ("2d", "3d"):
            raise ValueError("unknown view")
        # A Safari window from an earlier demo can reconnect to a new local
        # server. Ignore its stale visual receipt and screenshot; only the
        # current scene revision may advance this run.
        if view == "3d" and (payload.get("scenery", {}).get("scene_revision") != "verdant-native-continuous-v7"
                             or payload.get("view_token") != self.view_token):
            self.validation["ignored_legacy_3d"] = self.validation.get("ignored_legacy_3d", 0) + 1
            self.validation["last_ignored_3d"] = {
                "matching_token": payload.get("view_token") == self.view_token,
                "revision": payload.get("scenery", {}).get("scene_revision")}
            return
        with self.condition:
            frame = self.latest
            if payload.get("version") != self.version:
                return  # A reset/reconnect can legitimately leave an old response in flight.
            errors = []
            if payload.get("hash") != frame["hash"] or payload.get("tick") != frame["tick"]:
                errors.append(f"{view}: state hash/tick mismatch at {frame['tick']}")
            if view == "2d" and (payload.get("display_mode") != "pygame-offscreen"
                                  or payload.get("window_open") is not False
                                  or payload.get("png_sha256") != frame["jackal_png_sha256"]
                                  or payload.get("surface_size") != self.info["map"]["size"]):
                errors.append("2d: original Jackal offscreen frame not confirmed")
            if view == "3d":
                inset = payload.get("jackal_inset", {})
                if (inset.get("mode") != "original-pygame-frame"
                        or inset.get("tick") != frame["tick"]
                        or inset.get("png_sha256") != frame["jackal_png_sha256"]
                        or inset.get("size") != self.info["map"]["size"]):
                    errors.append("3d: embedded Jackal frame mismatch")
                scenery = payload.get("scenery", {})
                plane = scenery.get("battle_plane_height_m")
                if (scenery.get("collider") != "jackal-axis-aligned-cuboid"
                        or scenery.get("shell") != "cot-native-he-tracer"
                        or scenery.get("shell_textures") != "cot-original-particle-atlases"
                        or not isinstance(plane, (int, float)) or not math.isfinite(plane) or plane <= 0):
                    errors.append("3d: invalid original CoT shell / combat-plane presentation")
                    plane = 0
                if (scenery.get("boundary") != "transparent-air-barrier"
                        or any(scenery.get("exterior", {}).get(key) != expected for key, expected in {"trees": 3450, "hillside_trees": 1600,
                            "terrain_layers": 5, "rocks": 210, "houses": 12,
                            "poles": 94, "hay": 46, "mountains": 5, "barrier_panels": 4,
                            "boundary_fences": 89, "boundary_rocks": 26, "boundary_houses": 2,
                            "mountain_style": "cot-verdant-native-vista", "mountain_relief_bake": True,
                            "far_range": True}.items())
                        or scenery.get("exterior", {}).get("mountain_rows", 0) < 18
                        or scenery.get("exterior", {}).get("mountain_forest", 0) < 1000
                        or scenery.get("camouflage") != {"PLAYER": "merdc", "ENEMY": "desert"}
                        or scenery.get("sky") != "cot-baked-cumulus-verdant"
                        or scenery.get("lighting") != "diffuse-soil-sun-entity-shadows"
                        or scenery.get("ground_dressing", {}).get("paths") != 3
                        or scenery.get("ground_dressing", {}).get("patches", 0) < 500
                        or scenery.get("ground_dressing", {}).get("tufts", 0) < 37000
                        or scenery.get("ground_dressing", {}).get("bushes") != 420
                        or scenery.get("ground_dressing", {}).get("road_style") != "cot-verdant-ground071-cut-ruts-v5"
                        or scenery.get("ground_dressing", {}).get("road_grass_clearance_m") != 0.55
                        or scenery.get("roadside", {}).get("cars", 0) < 5
                        or scenery.get("roadside", {}).get("fuel_drums", 0) < 8
                        or not isinstance(scenery.get("roadside", {}).get("crushed"), int)
                        or not 0 <= scenery["roadside"]["crushed"] <=
                            scenery["roadside"]["cars"] + scenery["roadside"]["fuel_drums"]
                        or scenery.get("effects", {}).get("muzzle_style") != "cot-native-continuous-200mm-visual"
                        or scenery.get("effects", {}).get("wreck_fade") != "cot-burn-then-alpha-3.55s"
                        or scenery.get("effects", {}).get("soft_depth_bound") is not True
                        or inset.get("presentation") != "atomic-canvas-copy"):
                    errors.append("3d: exterior or air barrier is missing")

                def check(name, value):
                    if not math.isfinite(value):
                        errors.append(f"3d: non-finite {name}")
                        return
                    key = "max_" + name
                    self.validation[key] = max(value, self.validation[key])
                    if value > 1e-4:
                        errors.append(f"3d: {name}={value} at tick {frame['tick']}")

                def numeric_vector(actual, key, length):
                    vector = actual.get(key, [])
                    if not isinstance(vector, list) or len(vector) != length or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector):
                        errors.append(f"3d: invalid {key}")
                        return None
                    return vector

                if self.battle_plane_height is None:
                    self.battle_plane_height = plane
                check("combat_plane_error_m", abs(plane-self.battle_plane_height))
                rendered = {u["id"]: u for u in payload.get("rendered", [])}
                if set(rendered) != {u["id"] for u in frame["units"]}:
                    errors.append("3d: incomplete rendered entity IDs")
                for unit in frame["units"]:
                    actual = rendered.get(unit["id"])
                    if actual is None:
                        continue
                    if not all(isinstance(actual.get(k), (int, float)) and math.isfinite(actual[k])
                               for k in ("x", "y", "heading", "turret_heading", "health", "body_width", "body_length",
                                         "muzzle_height_m", "gun_vertical_component")):
                        errors.append("3d: non-finite transform")
                        continue
                    pos_error = math.hypot(actual["x"] - unit["position"][0], actual["y"] - unit["position"][1])
                    heading_error = abs((actual["heading"] - unit["heading"] + 180) % 360 - 180)
                    turret_error = abs((actual["turret_heading"] - unit["turret_heading"] + 180) % 360 - 180)
                    for name, value in (("position_error_px", pos_error), ("heading_error_deg", heading_error),
                                        ("turret_error_deg", turret_error)):
                        check(name, value)
                    for key in ("hitbox", "collision_box"):
                        rectangle = numeric_vector(actual, key, 4)
                        if rectangle is not None:
                            check("collider_error_px", max(abs(a-b) for a, b in zip(rectangle, unit[key])))
                    scaling = numeric_vector(actual, "model_scale", 3)
                    native_size = numeric_vector(actual, "native_body_size_m", 3)
                    body_size = numeric_vector(actual, "body_size_m", 3)
                    if scaling is not None:
                        check("model_scale_error", max(scaling)-min(scaling))
                    if native_size is not None and body_size is not None and min(native_size) > 0:
                        ratios = [a/b for a, b in zip(body_size, native_size)]
                        check("model_aspect_error", max(ratios)-min(ratios))
                    y_range = numeric_vector(actual, "collider_y", 2)
                    if y_range is not None and not y_range[0] < plane < y_range[1]:
                        errors.append("3d: combat plane outside collision cuboid")
                    check("combat_plane_error_m", abs(actual["muzzle_height_m"]-plane))
                    check("vertical_speed_mps", abs(actual["gun_vertical_component"]))
                    if actual["health"] != unit["health"] or actual.get("alive") != unit["alive"]:
                        errors.append("3d: life-state mismatch")
                    if actual.get("collider_active") != unit["alive"]:
                        errors.append("3d: dead tank retained a collider")
                    fade = actual.get("visual_opacity")
                    if not isinstance(fade, (int, float)) or not math.isfinite(fade) or not 0 <= fade <= 1:
                        errors.append("3d: invalid wreck fade")
                    for key, limit in (("cosmetic_bob_m", 0.25), ("cosmetic_pitch_deg", 8),
                                       ("native_recoil_m", 0.4), ("cosmetic_retreat_m", 0.25)):
                        value = actual.get(key)
                        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > limit:
                            errors.append(f"3d: invalid {key}")
                    if frame["result"]["finished"] and not unit["alive"] and actual.get("visible"):
                        errors.append("3d: finished wreck did not disappear")
                bullets = {b["id"]: b for b in payload.get("bullets", [])}
                if set(bullets) != {b["id"] for b in frame["bullets"]}:
                    errors.append("3d: projectile IDs mismatch")
                for bullet in frame["bullets"]:
                    actual = bullets.get(bullet["id"], {})
                    if not all(isinstance(actual.get(k), (int, float)) and math.isfinite(actual[k])
                               for k in ("x", "y", "height_m", "tail_height_m", "vertical_speed_mps")):
                        errors.append("3d: invalid projectile transform")
                        continue
                    error = math.dist((actual["x"], actual["y"]), bullet["position"])
                    check("bullet_position_error_px", error)
                    check("combat_plane_error_m", max(abs(actual["height_m"]-plane), abs(actual["tail_height_m"]-plane)))
                    check("vertical_speed_mps", abs(actual["vertical_speed_mps"]))
                    if actual.get("tracer") != "HE":
                        errors.append("3d: wrong projectile visual type")
                camera = payload.get("camera", {})
                for key, expected in (("orbit_distance_m", 13), ("orbit_pitch_deg", 10), ("fov_deg", 60)):
                    value = camera.get(key)
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value - expected) > 1e-4:
                        errors.append(f"3d: CoT arcade camera {key} mismatch")
                followed = next(u for u in frame["units"] if u["id"] == self.info["follow_id"])
                if camera.get("follow_id") != self.info["follow_id"]:
                    errors.append("3d: wrong followed entity")
                if not all(isinstance(camera.get(k), (int, float)) and math.isfinite(camera[k])
                           for k in ("anchor_x", "anchor_y")):
                    errors.append("3d: invalid camera anchor")
                else:
                    error = math.dist((camera["anchor_x"], camera["anchor_y"]), followed["position"])
                    self.validation["max_camera_anchor_error_px"] = max(
                        self.validation["max_camera_anchor_error_px"], error)
                    if error > 1e-4:
                        errors.append("3d: camera anchor mismatch")
                bearing = camera.get("heading")
                if camera.get("mode") != "turret-follow" or not isinstance(bearing, (int, float)) or not math.isfinite(bearing):
                    errors.append("3d: invalid turret-follow camera")
                else:
                    error = abs((bearing - followed["turret_heading"] + 180) % 360 - 180)
                    self.validation["max_camera_heading_error_deg"] = max(
                        self.validation["max_camera_heading_error_deg"], error)
                    if error > 1e-4:
                        errors.append("3d: camera does not follow turret bearing")
            if errors:
                self.validation["errors"].extend(errors)
                self.paused = True
                print("Render validation failed: " + "; ".join(errors), flush=True)
                raise ValueError("; ".join(errors))
            self.acks[view] = {"version": self.version, "tick": frame["tick"], "hash": frame["hash"]}
            self.validation["ack_count"][view] += 1
            receipt = {k: v for k, v in payload.items() if k not in ("screenshot", "effect_screenshot", "effect_captures")}
            captures = payload.get("effect_captures", [])
            receipt["effect_captures"] = [{key: value for key, value in capture.items() if key != "png"}
                                           for capture in captures]
            for capture in captures:
                if capture.get("name") not in ("muzzle-peak", "muzzle-recovery", "muzzle-smoke"):
                    raise ValueError("unknown effect capture")
                png_uri = capture.get("png", "")
                if not png_uri.startswith("data:image/png;base64,"):
                    raise ValueError("invalid effect PNG")
                png_bytes = base64.b64decode(png_uri.split(",", 1)[1], validate=True)
                if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("invalid effect PNG")
                (self.artifacts / (capture["name"] + "-3d.png")).write_bytes(png_bytes)
                if capture["name"] == "muzzle-peak":
                    (self.artifacts / "muzzle-3d.png").write_bytes(png_bytes)
            self.receipts.write(json.dumps(receipt, separators=(",", ":")) + "\n")
            self.receipts.flush()
            screenshot = payload.get("screenshot")
            if view == "3d" and screenshot:
                if not isinstance(screenshot, str) or not screenshot.startswith("data:image/png;base64,"):
                    raise ValueError("invalid screenshot")
                stem = "final" if frame["result"]["finished"] else "initial" if frame["tick"] == 0 else "contact"
                if stem != "contact" or not self.contact_saved:
                    png = base64.b64decode(screenshot.split(",", 1)[1], validate=True)
                    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                        raise ValueError("invalid PNG")
                    (self.artifacts / (stem + "-3d.png")).write_bytes(png)
                    (self.artifacts / (stem + "-2d.png")).write_bytes(base64.b64decode(self.png))
                    if stem == "contact":
                        self.contact_saved = True
            effect_screenshot = payload.get("effect_screenshot")
            if view == "3d" and effect_screenshot:
                if not isinstance(effect_screenshot, str) or not effect_screenshot.startswith("data:image/png;base64,"):
                    raise ValueError("invalid muzzle screenshot")
                effect_png = base64.b64decode(effect_screenshot.split(",", 1)[1], validate=True)
                if not effect_png.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("invalid muzzle PNG")
                (self.artifacts / "muzzle-3d.png").write_bytes(effect_png)
        if frame["result"]["finished"]:
            self.write_report()

