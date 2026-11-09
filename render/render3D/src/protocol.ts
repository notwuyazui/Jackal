import type { BattleManifest, FrameSnapshot, BattleEvent } from './generated/engine.ts';
import type { SceneDefinition } from '../maps/registry.ts';
export type { UnitFrame, BulletFrame, BattleEvent, Parameters } from './generated/engine.ts';

export interface Manifest extends BattleManifest {
  scene: SceneDefinition;
  seed: number; max_steps: number; follow_id: number; meters_per_pixel: number;
  mode?: 'async' | 'lockstep';
  show_inset?: boolean; controller?: string;
}
export interface Frame extends FrameSnapshot {
  hash: string; version: number; run_id: number;
  jackal_png: string; jackal_png_sha256: string;
  paused: boolean; speed: number; manifest: Manifest;
  pending_events?: BattleEvent[];
  result: { finished: boolean; winner: string | null; reason: string | null; alive: Record<string, number> };
}
export interface RenderReceipt {
  id: number; x: number; y: number; heading: number; turret_heading: number;
  health: number; alive: boolean; body_width: number; body_length: number;
  model_scale: number[]; native_body_size_m: number[]; body_size_m: number[];
  muzzle_height_m: number; gun_vertical_component: number;
  hitbox: number[]; collision_box: number[]; collider_y: number[];
  visual_opacity?: number; wreck_age_s?: number; recoil_age_s?: number | null;
  cosmetic_bob_m?: number; cosmetic_pitch_deg?: number; start_bounce_age_s?: number | null;
  chassis_roll_deg?: number; cosmetic_retreat_m?: number; native_recoil_m?: number;
  visible?: boolean; collider_active?: boolean;
  locomotion_mode?: string; gait_phase_rad?: number; gait_blend?: number;
  step_rate_hz?: number; stride_length_m?: number;
  left_foot_contact?: boolean; right_foot_contact?: boolean;
  nozzle_yaw_error_deg?: number;
  left_foot_world?: number[]; right_foot_world?: number[];
  foot_ik_error_m?: number; pelvis_yaw_deg?: number; torso_twist_deg?: number;
}
