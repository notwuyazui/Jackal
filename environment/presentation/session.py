"""Evaluation episode driver: Jackal owns simulation, renderers consume snapshots."""
import base64
import hashlib
import json
import secrets
import threading
import time
from environment.presentation.adapter import ROOT, build_world, export_frame, manifest, step_world
from environment.presentation.validation import RenderDiagnostics
from render.api import apply_map, create_inset_renderer

class Demo(RenderDiagnostics):
    supports_restart = True
    def __init__(self, args):
        self.args = args
        self.view_token = ""
        self.condition = threading.Condition()
        self.restart_requested = False
        self.acks = {}
        self.version = -1
        self.run_id = 0
        self.paused = True
        self.started = False
        self.speed = 1.0
        self.latest = None
        self.png = ""
        self.renderer = None
        self.fx = []
        self.records = None
        self.receipts = None
        self.artifacts = ROOT / "artifacts" / "demo"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self):
        self.view_token = secrets.token_urlsafe(12)
        if self.records:
            self.records.close()
        if self.receipts:
            self.receipts.close()
        self.world = build_world(self.args.state, self.args.seed)
        self.info = manifest(self.world, self.args.state, self.args.dt, self.args.seed, self.args.max_steps)
        self.info["scene"] = apply_map(self.world, dt=self.args.dt)
        if self.renderer:
            self.renderer.close()
        self.renderer = create_inset_renderer(*self.world.game_map.get_map_size())
        self.renderer.initialize(self.world.render_manifest(self.args.dt))
        self.run_id += 1
        self.paused = True
        self.started = False
        self.fx = []
        self.acks.clear()
        self.validation = {"ack_count": {"2d": 0, "3d": 0}, "errors": [],
                           "max_position_error_px": 0.0, "max_heading_error_deg": 0.0,
                           "max_turret_error_deg": 0.0, "max_collider_error_px": 0.0,
                           "max_model_scale_error": 0.0, "max_model_aspect_error": 0.0,
                           "max_combat_plane_error_m": 0.0, "max_vertical_speed_mps": 0.0,
                           "max_bullet_position_error_px": 0.0, "max_camera_anchor_error_px": 0.0,
                           "max_camera_heading_error_deg": 0.0}
        self.records = (self.artifacts / "last-run.jsonl").open("w")
        self.receipts = (self.artifacts / "view-receipts.jsonl").open("w")
        (self.artifacts / "manifest.json").write_text(json.dumps(self.info, indent=2))
        self.contact_saved = False
        self.battle_plane_height = None
        self.publish(record=True)

    def publish(self, record=False):
        frame = export_frame(self.world, self.args.max_steps)
        if record:
            self.records.write(json.dumps(frame, separators=(",", ":")) + "\n")
            self.records.flush()
        png = self.renderer.png(self.world.render_snapshot(), frame)
        png_sha256 = hashlib.sha256(png).hexdigest()
        with self.condition:
            self.version += 1
            self.latest = {**frame, "version": self.version, "run_id": self.run_id,
                           "paused": self.paused, "speed": self.speed, "manifest": self.info,
                           "jackal_png_sha256": png_sha256}
            self.png = base64.b64encode(png).decode()
            self.condition.notify_all()
        # The original Jackal renderer produced this frame without an SDL window.
        # The browser confirms that it decoded and displayed the same PNG.
        self.acknowledge({"view": "2d", "version": self.version,
                          "tick": frame["tick"], "hash": frame["hash"],
                          "display_mode": "pygame-offscreen", "window_open": False,
                          "png_sha256": png_sha256,
                          "surface_size": list(self.renderer.surface.get_size())})
        if frame["result"]["finished"]:
            self.write_report()

    def write_report(self):
        with self.condition:
            report = {"manifest": self.info, "final": self.latest,
                      "validation": self.validation, "view_receipts": dict(self.acks)}
            report["both_views_confirmed_final"] = all(
                self.acks.get(view, {}).get("version") == self.version for view in ("2d", "3d"))
            (self.artifacts / "last-result.json").write_text(json.dumps(report, indent=2))


    def loop(self):
        next_step = time.monotonic()
        while True:
            with self.condition:
                if self.restart_requested:
                    self.restart_requested = False
                    self.reset()
                    next_step = time.monotonic()
                    continue
            ready = all(self.acks.get(v, {}).get("version") == self.version for v in ("2d", "3d"))
            if ready and not self.started:
                self.started = True
                self.paused = False
                next_step = time.monotonic()
                print("CoT 3D and embedded Jackal 2D ready; battle started automatically", flush=True)
            now = time.monotonic()
            if not self.paused and ready and not self.latest["result"]["finished"] and now >= next_step:
                step_world(self.world, self.args.dt)
                if export_frame(self.world, self.args.max_steps)["result"]["finished"]:
                    self.paused = True
                self.publish(record=True)
                next_step = now + self.args.dt
            time.sleep(0.002)
