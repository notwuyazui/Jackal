/** Invisible, inert aim points on the burning house; no former tank/leaf mesh. */
import * as THREE from 'three';
import { effectiveParameters, toWorld, toHeading, toYaw, worldSize } from './mapping.ts';
import type { Manifest, Parameters, RenderReceipt, UnitFrame } from './protocol.ts';

export function createFireSite(parameters: Parameters, _team: string, manifest: Manifest) {
  const root = new THREE.Group(), body = new THREE.Group(), turret = new THREE.Group();
  root.name = `House aim point ${parameters.id}`;
  root.add(body, turret);
  const gameplaySpec = effectiveParameters(parameters, manifest.meters_per_pixel);
  let lastReceipt: RenderReceipt | null = null;
  return {
    root, body, turret, gameplaySpec, modelHeight: 7.2,
    get muzzleHeight() { return 7.2; },
    muzzleWorld(out: THREE.Vector3) { return out.copy(root.position).setY(7.2); },
    fire(_time: number) {},
    pose(unit: UnitFrame, _time: number, _physicsDt = 0): RenderReceipt {
      const p = toWorld(...unit.position, ...worldSize(manifest), manifest.meters_per_pixel);
      root.position.set(p.x, 0, p.z);
      root.rotation.y = toYaw(unit.heading);
      root.visible = false;
      lastReceipt = { id: unit.id, x: unit.position[0], y: unit.position[1],
        heading: toHeading(root.rotation.y), turret_heading: unit.turret_heading,
        health: unit.health, alive: unit.alive, body_width: 0, body_length: 0,
        model_scale: [1, 1, 1], native_body_size_m: [0, 0, 0], body_size_m: [0, 0, 0],
        muzzle_height_m: 7.2, gun_vertical_component: 0, hitbox: [], collision_box: [], collider_y: [],
        visual_opacity: 0, wreck_age_s: 0, recoil_age_s: null,
        cosmetic_bob_m: 0, cosmetic_pitch_deg: 0, chassis_roll_deg: 0,
        cosmetic_retreat_m: 0, native_recoil_m: 0, start_bounce_age_s: null,
        visible: false, collider_active: unit.alive };
      return lastReceipt;
    },
    presentationState() { return lastReceipt; },
    labelAnchor(out: THREE.Vector3) { return out.copy(root.position).setY(7.2); },
    cameraPivot(out: THREE.Vector3) { return out.copy(root.position).setY(7.2); },
    dispose() { root.removeFromParent(); },
  };
}
