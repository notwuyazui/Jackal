/** Original CoT M1A1 visual; all live gameplay state comes from Jackal. */
import * as THREE from 'three';
import { createTank, ensureTankBuilder } from '@cot/vehicles/fleetFactory.ts';
import { effectiveParameters, relativeTurretYaw, toHeading, toWorld, toYaw } from './mapping.ts';
import type { Manifest, Parameters, RenderReceipt, UnitFrame } from './protocol.ts';

export const prepareM1A1 = () => ensureTankBuilder('m1a1');
export const TEAM_CAMO = { PLAYER: 'merdc', ENEMY: 'desert' } as const;

export function createM1A1(parameters: Parameters, team: string, manifest: Manifest) {
  const visual = createTank('m1a1', { anisotropy: 8 }, {
    quality: 'high', geometryQuality: 'high', batchStatic: true,
    battleDetailLod: false, staticPreview: true, camoSeed: manifest.seed + parameters.id,
    camoPattern: team === 'ENEMY' ? TEAM_CAMO.ENEMY : TEAM_CAMO.PLAYER,
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
  let alive = true;
  let deathAt: number | null = null;
  let lastFireAt = -Infinity;
  let startBounceAt = -Infinity;
  let previousSpeed = 0;
  let lastReceipt: RenderReceipt | null = null;
  let fadeMaterials: Array<{ object: THREE.Mesh; original: THREE.Material | THREE.Material[] }> = [];
  const ownedFadeMaterials = new Set<THREE.Material>();

  function startFade() {
    if (fadeMaterials.length) return;
    const clones = new Map<THREE.Material, THREE.Material>();
    visual.root.traverse(object => {
      if (!(object instanceof THREE.Mesh)) return;
      const original = object.material;
      const clone = (material: THREE.Material) => {
        if (material.colorWrite === false) return material;
        let faded = clones.get(material);
        if (!faded) {
          faded = material.clone();
          faded.transparent = true;
          faded.depthWrite = false;
          faded.opacity = 1;
          faded.alphaTest = 0;
          faded.needsUpdate = true;
          clones.set(material, faded);
          ownedFadeMaterials.add(faded);
        }
        return faded;
      };
      object.material = Array.isArray(original) ? original.map(clone) : clone(original);
      object.castShadow = false;
      fadeMaterials.push({ object, original });
    });
  }

  function setFade(opacity: number) {
    for (const material of ownedFadeMaterials) material.opacity = opacity;
    root.visible = opacity > 0.001;
  }

  return {
    root, body, turret, gameplaySpec, modelHeight,
    get muzzleHeight() { return restMuzzleHeight; },
    muzzleWorld(out: THREE.Vector3) { root.updateMatrixWorld(true); return visual.gunMuzzleWorld(out); },
    fire(time: number) { lastFireAt = time; visual.recoilKick(0, 1); },
    pose(unit: UnitFrame, time: number, physicsDt = 0): RenderReceipt {
      const dt = Math.max(0, time - previousTime);
      previousTime = time;
      const position = toWorld(...unit.position, ...manifest.map.size, scale);
      root.position.set(position.x, 0, position.z);
      root.rotation.set(0, toYaw(unit.heading), 0);
      if (unit.alive && Math.abs(previousSpeed) < 1.5 && Math.abs(unit.speed) >= 2) {
        startBounceAt = time;
      }
      previousSpeed = unit.speed;
      const yawRate = -unit.angular_speed * Math.PI / 180;
      trackScroll.l += (unit.speed * scale + yawRate * parameters.size[0] * scale / 2) * physicsDt / modelScale;
      trackScroll.r += (unit.speed * scale - yawRate * parameters.size[0] * scale / 2) * physicsDt / modelScale;
      if (alive !== unit.alive) {
        alive = unit.alive;
        if (alive) { visual.resetDestroyed(); deathAt = null; root.visible = true; }
        else { deathAt = time; visual.setDestroyed({ pop: false, ageS: 0 }); }
      }
      const turretYaw = relativeTurretYaw(unit.heading, unit.turret_heading);
      visual.syncFromState({ pos: nativePosition, yaw: 0, visualPitch: 0, visualRoll: 0,
        yawRate: 0, speed: unit.speed * scale, turretYaw, gunPitch: 0, trackScroll,
        _susp: { p: 0, r: 0 }, _swayEst: 0 }, dt);
      // The outer root is Jackal's authoritative pose. These damped motions
      // move only CoT's inner model, never its collision cuboid or shell plane.
      const startAge = time - startBounceAt;
      const fireAge = time - lastFireAt;
      const startEnvelope = alive && startAge >= 0 && startAge < 0.85 ? Math.exp(-startAge * 3.6) : 0;
      const fireEnvelope = alive && fireAge >= 0 && fireAge < 0.75 ? Math.exp(-fireAge * 4.8) : 0;
      const bob = (startEnvelope ? startEnvelope * 0.105 * Math.sin(startAge * 20) : 0) -
        (fireEnvelope ? fireEnvelope * 0.09 * Math.sin(fireAge * 17 + 0.35) : 0);
      const fireRock = fireEnvelope ? fireEnvelope * 0.055 * Math.sin(fireAge * 13 + 0.8) : 0;
      const pitch = (startEnvelope ? startEnvelope * 0.013 * Math.sin(startAge * 15) : 0) -
        fireRock * Math.cos(turretYaw);
      const roll = -fireRock * Math.sin(turretYaw) +
        (fireEnvelope ? fireEnvelope * 0.017 * Math.sin(fireAge * 36) : 0);
      const retreat = fireEnvelope ? fireEnvelope * 0.18 * Math.sin(fireAge * 14 + 0.5) : 0;
      visual.root.position.y += bob;
      visual.root.position.x -= Math.sin(turretYaw) * retreat;
      visual.root.position.z -= Math.cos(turretYaw) * retreat;
      visual.root.rotation.x += pitch;
      visual.root.rotation.z += roll;
      // The native 120 mm stroke is only 9.4 cm after map scaling. Preserve
      // the original moving tube and hydraulic timing, with a readable throw.
      recoil.position.z *= 3;
      // CoT owns the cosmetic recoil/flinch/burn pose. Jackal still owns the
      // logical hull bearing, horizontal gun plane, projectiles and collision.
      if (alive) { turret.position.copy(turretSeat); turret.rotation.y = turretYaw; }
      const deathAge = deathAt === null ? 0 : Math.max(0, time - deathAt);
      const fade = deathAt === null ? 1 : 1 - THREE.MathUtils.smoothstep(deathAge, 2.1, 3.55);
      if (deathAge >= 2.1) { startFade(); setFade(fade); }
      root.updateMatrixWorld(true);
      lastReceipt = { id: unit.id, x: root.position.x / scale + manifest.map.size[0] / 2,
        y: root.position.z / scale + manifest.map.size[1] / 2,
        heading: toHeading(root.rotation.y), turret_heading: toHeading(toYaw(unit.turret_heading)),
        health: unit.health, alive, body_width: footprint[0], body_length: footprint[1],
        model_scale: visual.root.scale.toArray(), native_body_size_m: dimensions.toArray(),
        body_size_m: bodySize.toArray(), muzzle_height_m: restMuzzleHeight,
        gun_vertical_component: 0, hitbox: [], collision_box: [], collider_y: [],
        visual_opacity: fade, wreck_age_s: deathAge,
        recoil_age_s: Number.isFinite(lastFireAt) ? Math.max(0, time - lastFireAt) : null,
        cosmetic_bob_m: bob, cosmetic_pitch_deg: THREE.MathUtils.radToDeg(visual.root.rotation.x),
        chassis_roll_deg: THREE.MathUtils.radToDeg(visual.root.rotation.z),
        cosmetic_retreat_m: retreat, native_recoil_m: -recoil.position.z * modelScale,
        start_bounce_age_s: Number.isFinite(startBounceAt) ? Math.max(0, startAge) : null,
        visible: root.visible, collider_active: unit.alive };
      return lastReceipt;
    },
    presentationState() { return lastReceipt; },
    labelAnchor(out: THREE.Vector3) { visual.turretTopWorld(out); out.y += 0.6; return out; },
    // A camera riding the shaking turret cancelled most of the visible kick.
    // Keep the original chase height/distance, anchored to the physics pose.
    cameraPivot(out: THREE.Vector3) { out.copy(root.position); out.y += cameraAnchorHeight; return out; },
    dispose() {
      for (const { object, original } of fadeMaterials) object.material = original;
      fadeMaterials = [];
      for (const material of ownedFadeMaterials) material.dispose();
      ownedFadeMaterials.clear();
      visual.dispose(); root.removeFromParent();
    },
  };
}
