/** Original CoT M1A1 visual; all live gameplay state comes from Jackal. */
import * as THREE from 'three';
import { createTank, ensureTankBuilder } from '@cot/vehicles/fleetFactory.ts';
import { worldSize, effectiveParameters, relativeTurretYaw, toHeading, toWorld, toYaw } from './mapping.ts';
import type { Manifest, Parameters, RenderReceipt, UnitFrame } from './protocol.ts';

export const prepareM1A1 = () => ensureTankBuilder('m1a1');
export const TEAM_CAMO = { PLAYER: 'fire-engine-red', ENEMY: 'burning-house-aim-points' } as const;

export function createM1A1(parameters: Parameters, team: string, manifest: Manifest) {
  const visual = createTank('m1a1', { anisotropy: 8 }, {
    quality: 'high', geometryQuality: 'high', batchStatic: true,
    battleDetailLod: false, staticPreview: true, camoSeed: manifest.seed + parameters.id,
    camoPattern: 'desert',
  });
  const root = new THREE.Group();
  root.name = `Jackal #${parameters.id} / CoT native M1A1 Abrams`;
  root.add(visual.root);
  visual.root.traverse(object => {
    if (!(object instanceof THREE.Mesh)) return;
    if (object.userData.shadowOnly) object.castShadow = false;
    else {
      object.castShadow = true;
      object.receiveShadow = true;
    }
  });
  const body = visual.root.getObjectByName('rig_hull')!;
  const turret = visual.root.getObjectByName('rig_turret')!;
  const gun = visual.root.getObjectByName('rig_gun')!;
  const recoil = visual.root.getObjectByName('rig_recoil')!;
  if (!body || !turret || !gun || !recoil) throw new Error('CoT M1A1 articulation nodes are missing');
  root.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(body);
  const dimensions = bounds.getSize(new THREE.Vector3());
  const center = bounds.getCenter(new THREE.Vector3());
  const scale = manifest.meters_per_pixel;
  // One scale for the entire original model, including all articulated parts.
  // Jackal's rectangular colliders are independent of the visual footprint.
  const modelScale = Math.min(parameters.size[0] * scale / dimensions.x,
    parameters.size[1] * scale / dimensions.z);
  visual.root.scale.setScalar(modelScale);
  const nativePosition = new THREE.Vector3(-center.x, -bounds.min.y, -center.z).multiplyScalar(modelScale);
  visual.root.position.copy(nativePosition);
  const turretSeat = turret.position.clone();
  const turretRestRotation = turret.rotation.clone();
  const gunSeat = gun.position.clone(), gunRestRotation = gun.rotation.clone();
  const recoilSeat = recoil.position.clone(), recoilRestRotation = recoil.rotation.clone();
  root.updateMatrixWorld(true);
  const restMuzzleHeight = visual.gunMuzzleWorld(new THREE.Vector3()).y;
  const cameraAnchorHeight = visual.turretTopWorld(new THREE.Vector3()).y + 2.5;
  const bodySize = new THREE.Box3().setFromObject(body).getSize(new THREE.Vector3());
  const footprint = [bodySize.x / scale, bodySize.z / scale];
  const modelHeight = new THREE.Box3().setFromObject(visual.root).max.y;
  const gameplaySpec = effectiveParameters(parameters, scale);
  root.userData = { unitId: parameters.id, modelId: 'm1a1', team, gameplaySpec, physicsOwner: 'Jackal' };
  const trackScroll = { l: 0, r: 0 };
  let previousTime = 0;
  let lastReceipt: RenderReceipt | null = null;
  const painted = new Map<THREE.Material, THREE.MeshStandardMaterial>();
  const paint = (material: THREE.Material) => {
    const role = material.userData.appearanceRole ?? material.userData.vehicleMaterialRole;
    if (!(material instanceof THREE.MeshStandardMaterial) ||
        !['armorPaint', 'wheelPaint', 'fittingPaint'].includes(role)) return material;
    let red = painted.get(material);
    if (!red) {
      red = material.clone();
      red.map = null;
      red.color.setHex(role === 'fittingPaint' ? 0x9e1419 : 0xd91e26);
      red.roughness = 0.42;
      red.metalness = 0.25;
      painted.set(material, red);
    }
    return red;
  };
  visual.root.traverse(object => {
    if (object instanceof THREE.Mesh) object.material = Array.isArray(object.material)
      ? object.material.map(paint) : paint(object.material);
  });
  // White rescue stripes, water tank and a light bar make the silhouette read
  // as a tracked fire appliance while keeping CoT's original articulated rig.
  const rescue = new THREE.Group();
  const white = new THREE.MeshStandardMaterial({ color: 0xf4f2df, roughness: 0.42 });
  const silver = new THREE.MeshStandardMaterial({ color: 0xc5d5da, metalness: 0.7, roughness: 0.3 });
  const blue = new THREE.MeshStandardMaterial({ color: 0x238eff, emissive: 0x1258db, emissiveIntensity: 1.3 });
  function box(w: number, h: number, d: number, x: number, y: number, z: number, material: THREE.Material) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
    mesh.position.set(x, y, z); mesh.castShadow = true; rescue.add(mesh);
  }
  const w = bodySize.x, l = bodySize.z;
  const roof = visual.turretTopWorld(new THREE.Vector3());
  for (const side of [-1, 1]) box(0.04, 0.16, l * 0.7, side * w * 0.49, modelHeight * 0.38, 0, white);
  box(w * 0.55, 0.6, l * 0.25, 0, modelHeight * 0.67, -l * 0.3, silver);
  box(w * 0.48, 0.09, 0.26, roof.x, roof.y + 0.05, roof.z, white);
  for (const side of [-1, 1]) box(0.42, 0.17, 0.27, roof.x + side * w * 0.18, roof.y + 0.17, roof.z, blue);
  root.add(rescue);

  return {
    root, body, turret, gameplaySpec, modelHeight,
    get muzzleHeight() { return restMuzzleHeight; },
    muzzleWorld(out: THREE.Vector3) { root.updateMatrixWorld(true); return visual.gunMuzzleWorld(out); },
    fire(_time: number) { /* Water cannon: no recoil or muzzle flash. */ },
    pose(unit: UnitFrame, time: number, physicsDt = 0): RenderReceipt {
      const dt = Math.max(0, time - previousTime);
      previousTime = time;
      const position = toWorld(...unit.position, ...worldSize(manifest), scale);
      root.position.set(position.x, 0, position.z);
      root.rotation.set(0, toYaw(unit.heading), 0);
      const yawRate = -unit.angular_speed * Math.PI / 180;
      trackScroll.l += (unit.speed * scale + yawRate * parameters.size[0] * scale / 2) * physicsDt / modelScale;
      trackScroll.r += (unit.speed * scale - yawRate * parameters.size[0] * scale / 2) * physicsDt / modelScale;
      root.visible = unit.alive;
      const turretYaw = relativeTurretYaw(unit.heading, unit.turret_heading);
      visual.syncFromState({ pos: nativePosition, yaw: 0, visualPitch: 0, visualRoll: 0,
        yawRate: 0, speed: unit.speed * scale, turretYaw, gunPitch: 0, trackScroll,
        _susp: { p: 0, r: 0 }, _swayEst: 0 }, dt);
      // Suppress CoT's internal suspension, idle sway and recoil as well.
      visual.root.position.copy(nativePosition);
      visual.root.rotation.set(0, 0, 0);
      body.rotation.x = body.rotation.z = 0;
      gun.position.copy(gunSeat);
      gun.rotation.copy(gunRestRotation);
      recoil.position.copy(recoilSeat);
      recoil.rotation.copy(recoilRestRotation);
      turret.position.copy(turretSeat);
      turret.rotation.set(turretRestRotation.x, turretYaw, turretRestRotation.z, turretRestRotation.order);
      root.updateMatrixWorld(true);
      lastReceipt = { id: unit.id, x: root.position.x / scale + worldSize(manifest)[0] / 2,
        y: root.position.z / scale + worldSize(manifest)[1] / 2,
        heading: toHeading(root.rotation.y), turret_heading: toHeading(toYaw(unit.turret_heading)),
        health: unit.health, alive: unit.alive, body_width: footprint[0], body_length: footprint[1],
        model_scale: visual.root.scale.toArray(), native_body_size_m: dimensions.toArray(),
        body_size_m: bodySize.toArray(), muzzle_height_m: restMuzzleHeight,
        gun_vertical_component: 0, hitbox: [], collision_box: [], collider_y: [],
        visual_opacity: unit.alive ? 1 : 0, wreck_age_s: 0, recoil_age_s: null,
        cosmetic_bob_m: 0, cosmetic_pitch_deg: 0, chassis_roll_deg: 0,
        cosmetic_retreat_m: 0, native_recoil_m: 0, start_bounce_age_s: null,
        visible: root.visible, collider_active: unit.alive };
      return lastReceipt;
    },
    presentationState() { return lastReceipt; },
    labelAnchor(out: THREE.Vector3) { visual.turretTopWorld(out); out.y += 0.6; return out; },
    // Keep the chase camera anchored to the physics pose.
    cameraPivot(out: THREE.Vector3) { out.copy(root.position); out.y += cameraAnchorHeight; return out; },
    dispose() {
      for (const material of painted.values()) material.dispose();
      for (const material of [white, silver, blue]) material.dispose();
      rescue.traverse(object => { if (object instanceof THREE.Mesh) object.geometry.dispose(); });
      visual.dispose(); root.removeFromParent();
    },
  };
}
