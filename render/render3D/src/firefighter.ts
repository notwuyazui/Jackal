/** Procedural turnout-clad firefighter driven by Jackal's invisible tank proxy. */
import * as THREE from 'three';
import { advanceFirefighterGait, createGaitState, sampleFootTarget, solveLegPose,
  type FirefighterGaitSample } from './firefighterGait.ts';
import { effectiveParameters, toWorld, toYaw, worldSize } from './mapping.ts';
import type { Manifest, Parameters, RenderReceipt, UnitFrame } from './protocol.ts';

export const prepareFirefighter = async () => undefined;
export const TEAM_CAMO = { PLAYER: 'detailed-turnout-firefighter', ENEMY: 'burning-house-aim-points' } as const;

const HIP_HEIGHT = .92;
const ANKLE_HEIGHT = .10;
const UPPER_LEG = .45;
const LOWER_LEG = .45;
const MODEL_HEIGHT = 1.92;
const wrappedAngle = (value: number) => Math.atan2(Math.sin(value), Math.cos(value));

interface LegRig {
  hip: THREE.Group;
  knee: THREE.Group;
  ankle: THREE.Group;
}

interface ArmRig {
  upper: THREE.Mesh;
  lower: THREE.Mesh;
  stripe: THREE.Mesh;
  elbow: THREE.Mesh;
  glove: THREE.Mesh;
}

function setSegment(mesh: THREE.Mesh, start: THREE.Vector3, end: THREE.Vector3) {
  const direction = new THREE.Vector3().subVectors(end, start);
  const length = Math.max(.001, direction.length());
  mesh.position.copy(start).add(end).multiplyScalar(.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.multiplyScalar(1 / length));
  mesh.scale.set(1, length, 1);
}

function pointBetween(a: THREE.Vector3, b: THREE.Vector3, t: number) {
  return new THREE.Vector3().lerpVectors(a, b, t);
}

export function createFirefighter(parameters: Parameters, team: string, manifest: Manifest) {
  const scale = manifest.meters_per_pixel;
  const gameplaySpec = effectiveParameters(parameters, scale);
  const root = new THREE.Group();
  root.name = `Jackal #${parameters.id} / procedural firefighter`;
  root.userData = { unitId: parameters.id, modelId: 'firefighter', team,
    gameplaySpec, physicsOwner: 'Jackal-invisible-proxy' };

  const materials = {
    coat: new THREE.MeshStandardMaterial({ color: 0x6f4a29, roughness: .8, metalness: .02 }),
    coatDark: new THREE.MeshStandardMaterial({ color: 0x3e2d22, roughness: .88 }),
    reflective: new THREE.MeshStandardMaterial({ color: 0xe9ee75, emissive: 0x77791f,
      emissiveIntensity: .28, roughness: .38, metalness: .12 }),
    silver: new THREE.MeshStandardMaterial({ color: 0xd6ddd5, roughness: .28, metalness: .48 }),
    helmet: new THREE.MeshStandardMaterial({ color: 0xc82120, roughness: .34, metalness: .12 }),
    helmetDark: new THREE.MeshStandardMaterial({ color: 0x56120f, roughness: .5 }),
    skin: new THREE.MeshStandardMaterial({ color: 0xb77b5f, roughness: .72 }),
    rubber: new THREE.MeshStandardMaterial({ color: 0x16191b, roughness: .72 }),
    pack: new THREE.MeshStandardMaterial({ color: 0x263137, roughness: .58, metalness: .12 }),
    cylinder: new THREE.MeshStandardMaterial({ color: 0xd7d0a6, roughness: .34, metalness: .68 }),
    brass: new THREE.MeshStandardMaterial({ color: 0xb98a36, roughness: .3, metalness: .75 }),
    hose: new THREE.MeshStandardMaterial({ color: 0x9b6435, roughness: .86 }),
    glass: new THREE.MeshPhysicalMaterial({ color: 0x9fd6e8, transparent: true, opacity: .34,
      roughness: .08, metalness: 0, clearcoat: 1, depthWrite: false }),
  };

  const body = new THREE.Group();
  body.name = 'firefighter_hips';
  body.position.y = HIP_HEIGHT;
  root.add(body);
  const turret = new THREE.Group();
  turret.name = 'firefighter_aiming_torso';
  turret.position.y = .12;
  turret.rotation.order = 'YXZ';
  body.add(turret);

  function add(geometry: THREE.BufferGeometry, material: THREE.Material, parent: THREE.Object3D,
    name: string, position: [number, number, number], scaleXYZ: [number, number, number] = [1, 1, 1],
    rotation: [number, number, number] = [0, 0, 0]) {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = name;
    mesh.position.set(...position); mesh.scale.set(...scaleXYZ); mesh.rotation.set(...rotation);
    mesh.castShadow = true; mesh.receiveShadow = true;
    parent.add(mesh);
    return mesh;
  }

  // Pelvis, tailored coat body and reflective bands.
  add(new THREE.CapsuleGeometry(.20, .12, 6, 14), materials.coatDark, body,
    'turnout trousers waist', [0, .02, 0], [1.05, .65, .76]);
  add(new THREE.CylinderGeometry(.255, .285, .25, 16), materials.coat, turret,
    'turnout coat skirt', [0, .04, 0], [1, 1, .72]);
  add(new THREE.CapsuleGeometry(.225, .18, 8, 18), materials.coat, turret,
    'padded turnout jacket', [0, .31, 0], [1.08, 1, .76]);
  add(new THREE.CylinderGeometry(.246, .258, .045, 18), materials.reflective, turret,
    'waist reflective belt', [0, .12, 0], [1, 1, .75]);
  add(new THREE.CylinderGeometry(.235, .238, .038, 18), materials.reflective, turret,
    'chest reflective belt', [0, .37, 0], [1, 1, .75]);
  for (const side of [-1, 1]) {
    add(new THREE.BoxGeometry(.04, .48, .025), materials.reflective, turret,
      `front reflective suspender ${side}`, [side * .105, .31, .178], [1, 1, 1], [0, 0, side * .07]);
  }

  // SCBA frame, air cylinder, straps and pressure gauge.
  add(new THREE.BoxGeometry(.34, .46, .10), materials.pack, turret,
    'SCBA backplate', [0, .30, -.205]);
  add(new THREE.CylinderGeometry(.095, .095, .48, 16), materials.cylinder, turret,
    'SCBA air cylinder', [0, .31, -.315]);
  add(new THREE.CylinderGeometry(.10, .10, .035, 16), materials.brass, turret,
    'SCBA valve', [0, .565, -.315]);
  for (const side of [-1, 1]) add(new THREE.BoxGeometry(.045, .49, .035), materials.rubber, turret,
    `SCBA shoulder strap ${side}`, [side * .13, .34, .185], [1, 1, 1], [0, 0, side * .16]);
  add(new THREE.CylinderGeometry(.042, .042, .025, 14), materials.glass, turret,
    'pressure gauge', [.245, .21, .13], [1, 1, 1], [Math.PI / 2, 0, 0]);

  // Neck, face, ear protection, helmet shell, brim, crest and visor.
  add(new THREE.CylinderGeometry(.082, .092, .12, 14), materials.coatDark, turret,
    'protective hood neck', [0, .625, 0]);
  const head = new THREE.Group(); head.name = 'helmeted head'; head.position.y = .73; turret.add(head);
  add(new THREE.SphereGeometry(.145, 20, 14), materials.skin, head, 'face', [0, 0, .012], [1, 1.05, .94]);
  add(new THREE.SphereGeometry(.18, 22, 14, 0, Math.PI * 2, 0, Math.PI * .66), materials.helmet,
    head, 'red fire helmet shell', [0, .075, -.005], [1.04, .78, 1.10]);
  add(new THREE.CylinderGeometry(.225, .225, .035, 24), materials.helmetDark,
    head, 'helmet brim', [0, .045, 0]);
  add(new THREE.BoxGeometry(.055, .075, .28), materials.helmet, head,
    'helmet crown ridge', [0, .165, -.015], [1, 1, 1], [-.08, 0, 0]);
  add(new THREE.BoxGeometry(.225, .17, .018), materials.glass, head,
    'lowered face visor', [0, -.01, .145], [1, 1, 1], [-.12, 0, 0]);
  add(new THREE.BoxGeometry(.17, .055, .015), materials.rubber, head,
    'respirator facepiece', [0, -.055, .158]);

  function createLeg(side: number): LegRig {
    const hip = new THREE.Group(); hip.name = `${side < 0 ? 'left' : 'right'} hip`;
    hip.position.x = side * .14; body.add(hip);
    add(new THREE.CapsuleGeometry(.105, .24, 6, 12), materials.coat, hip,
      'turnout trouser thigh', [0, -UPPER_LEG / 2, 0]);
    add(new THREE.CylinderGeometry(.112, .112, .045, 14), materials.reflective, hip,
      'thigh reflective stripe', [0, -.31, 0]);
    const knee = new THREE.Group(); knee.name = 'articulated knee'; knee.position.y = -UPPER_LEG; hip.add(knee);
    add(new THREE.SphereGeometry(.108, 14, 10), materials.coatDark, knee, 'reinforced knee pad', [0, 0, .025], [1, 1, .72]);
    add(new THREE.CapsuleGeometry(.09, .27, 6, 12), materials.coat, knee,
      'turnout trouser shin', [0, -LOWER_LEG / 2, 0]);
    add(new THREE.CylinderGeometry(.104, .104, .05, 14), materials.reflective, knee,
      'ankle reflective stripe', [0, -.31, 0]);
    const ankle = new THREE.Group(); ankle.name = 'articulated ankle'; ankle.position.y = -LOWER_LEG; knee.add(ankle);
    add(new THREE.BoxGeometry(.20, .145, .34), materials.rubber, ankle,
      'steel toe fire boot', [0, -.035, .075]);
    add(new THREE.BoxGeometry(.21, .035, .35), materials.coatDark, ankle,
      'boot tread', [0, -.115, .075]);
    return { hip, knee, ankle };
  }
  const leftLeg = createLeg(-1), rightLeg = createLeg(1);

  const armGeometry = new THREE.CylinderGeometry(.082, .09, 1, 12);
  const forearmGeometry = new THREE.CylinderGeometry(.073, .082, 1, 12);
  const stripeGeometry = new THREE.CylinderGeometry(.088, .088, 1, 12);
  const jointGeometry = new THREE.SphereGeometry(.09, 14, 10);
  const gloveGeometry = new THREE.SphereGeometry(.075, 14, 10);
  function createArm(side: number): ArmRig {
    const upper = add(armGeometry, materials.coat, turret, `${side < 0 ? 'left' : 'right'} upper arm`, [0, 0, 0]);
    const lower = add(forearmGeometry, materials.coat, turret, 'forearm', [0, 0, 0]);
    const stripe = add(stripeGeometry, materials.reflective, turret, 'sleeve reflective stripe', [0, 0, 0]);
    const elbow = add(jointGeometry, materials.coatDark, turret, 'elbow reinforcement', [0, 0, 0], [1, .9, .8]);
    const glove = add(gloveGeometry, materials.rubber, turret, 'hose glove', [0, 0, 0], [1, .8, 1.15]);
    return { upper, lower, stripe, elbow, glove };
  }
  const leftArm = createArm(-1), rightArm = createArm(1);

  const nozzle = add(new THREE.CylinderGeometry(.05, .065, 1, 16), materials.silver, turret,
    'metal water nozzle', [0, 0, 0]);
  const nozzleTip = add(new THREE.CylinderGeometry(.085, .062, 1, 16), materials.brass, turret,
    'brass nozzle tip', [0, 0, 0]);
  const nozzleGrip = add(new THREE.BoxGeometry(.16, .24, .055), materials.rubber, turret,
    'nozzle pistol grip', [0, 0, 0]);
  const muzzleMarker = new THREE.Object3D(); muzzleMarker.name = 'water nozzle outlet'; turret.add(muzzleMarker);
  const hoseCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-.03, .25, .40), new THREE.Vector3(.24, .08, .18),
    new THREE.Vector3(.30, -.38, -.06), new THREE.Vector3(.18, -.76, -.34),
    new THREE.Vector3(.07, -1.00, -.60),
  ]);
  add(new THREE.TubeGeometry(hoseCurve, 48, .047, 10, false), materials.hose, turret,
    'charged fire hose', [0, 0, 0]);

  const tempA = new THREE.Vector3(), tempB = new THREE.Vector3(), tempC = new THREE.Vector3();
  function poseArm(rig: ArmRig, side: number, sample: FirefighterGaitSample, spraying: boolean) {
    const gaitSway = Math.sin(gait.phase + (side < 0 ? 0 : Math.PI)) * gait.blend * .014;
    const brace = spraying ? 1 : 0;
    const shoulder = tempA.set(side * .265, .48, -.005);
    const elbow = tempB.set(side * (.30 + brace * .015), .285 + gaitSway, .28 - brace * .015);
    const hand = tempC.set(side * .105, .265 + gaitSway * .35, .59 + (side < 0 ? -.035 : .035));
    setSegment(rig.upper, shoulder, elbow);
    setSegment(rig.lower, elbow, hand);
    setSegment(rig.stripe, pointBetween(shoulder, elbow, .58), pointBetween(shoulder, elbow, .76));
    rig.elbow.position.copy(elbow); rig.glove.position.copy(hand);
  }

  function poseNozzle(sample: FirefighterGaitSample, spraying: boolean) {
    const settle = gait.blend * Math.sin(gait.phase * 2) * .006;
    const rear = new THREE.Vector3(-.015, .255 + settle, .43);
    const front = new THREE.Vector3(0, .48 + settle + (spraying ? .015 : 0), 1.04);
    setSegment(nozzle, rear, front);
    setSegment(nozzleTip, pointBetween(rear, front, .84), front);
    nozzleTip.scale.x = nozzleTip.scale.z = 1;
    nozzleGrip.position.copy(pointBetween(rear, front, .25));
    nozzleGrip.rotation.copy(nozzle.rotation);
    nozzleGrip.rotateX(.42);
    muzzleMarker.position.copy(front);
  }

  root.traverse(object => {
    if (object instanceof THREE.Mesh) { object.castShadow = true; object.receiveShadow = true; }
  });

  const gait = createGaitState();
  let previousTime = 0, previousFacingYaw = NaN, lastSprayAt = -Infinity;
  let lastReceipt: RenderReceipt | null = null;
  let lastSample: FirefighterGaitSample = { moving: false, locomotionMps: 0, strideLengthM: .68,
    cycleRateHz: 0, stepRateHz: 0, runBlend: 0 };
  const muzzleWorldPosition = new THREE.Vector3();

  function poseLeg(rig: LegRig, foot: ReturnType<typeof sampleFootTarget>, blend: number) {
    const lift = foot.lift * blend;
    const targetZ = foot.z * blend;
    const targetDown = body.position.y - (ANKLE_HEIGHT + lift);
    const solved = solveLegPose(targetZ, targetDown, UPPER_LEG, LOWER_LEG);
    rig.hip.rotation.x = solved.hip;
    rig.knee.rotation.x = solved.knee;
    rig.ankle.rotation.x = solved.ankle;
  }

  return {
    root, body, turret, gameplaySpec, modelHeight: MODEL_HEIGHT,
    get muzzleHeight() { return 1.58; },
    muzzleWorld(out: THREE.Vector3) { root.updateMatrixWorld(true); return muzzleMarker.getWorldPosition(out); },
    fire(time: number) { lastSprayAt = time; },
    pose(unit: UnitFrame, time: number, _physicsDt = 0): RenderReceipt {
      const dt = Math.max(0, Math.min(.05, time - previousTime));
      previousTime = time;
      const position = toWorld(...unit.position, ...worldSize(manifest), scale);
      root.position.set(position.x, 0, position.z);
      // Firefighters keep their chest, hands and nozzle on the fire while
      // their feet travel along the AI's side-forward movement direction.
      // This avoids an impossible 120° torso twist after a random retarget.
      const facingYaw = toYaw(unit.turret_heading);
      const visualTurnDegS = dt > 1e-6 && Number.isFinite(previousFacingYaw)
        ? THREE.MathUtils.radToDeg(wrappedAngle(facingYaw - previousFacingYaw)) / dt
        : unit.angular_speed;
      previousFacingYaw = facingYaw;
      root.rotation.set(0, facingYaw, 0);
      root.visible = unit.alive;

      lastSample = advanceFirefighterGait(gait, unit.speed * scale, visualTurnDegS, dt);
      const motion = gait.blend;
      const bob = motion * (.008 + .012 * Math.abs(Math.sin(gait.phase * 2)));
      body.position.y = HIP_HEIGHT + bob;
      body.rotation.z = Math.sin(gait.phase) * motion * .026;
      body.rotation.y = Math.sin(gait.phase * 2) * motion * .012;
      const leftFoot = sampleFootTarget(gait.phase, 0, lastSample);
      const rightFoot = sampleFootTarget(gait.phase, .5, lastSample);
      const travelYaw = wrappedAngle(toYaw(unit.heading) - facingYaw);
      leftLeg.hip.rotation.y = travelYaw - body.rotation.y;
      rightLeg.hip.rotation.y = travelYaw - body.rotation.y;
      poseLeg(leftLeg, leftFoot, motion); poseLeg(rightLeg, rightFoot, motion);

      const spraying = time - lastSprayAt < .22;
      turret.rotation.y = -body.rotation.y;
      turret.rotation.x = motion * lastSample.runBlend * .045 - (spraying ? .028 : 0);
      turret.rotation.z = -body.rotation.z * .55;
      poseArm(leftArm, -1, lastSample, spraying); poseArm(rightArm, 1, lastSample, spraying);
      poseNozzle(lastSample, spraying);
      root.updateMatrixWorld(true);
      muzzleMarker.getWorldPosition(muzzleWorldPosition);

      lastReceipt = { id: unit.id, x: unit.position[0], y: unit.position[1],
        heading: unit.heading, turret_heading: unit.turret_heading,
        health: unit.health, alive: unit.alive,
        body_width: .72 / scale, body_length: .82 / scale,
        model_scale: [1, 1, 1], native_body_size_m: [.72, MODEL_HEIGHT, .82],
        body_size_m: [.72, MODEL_HEIGHT, .82], muzzle_height_m: muzzleWorldPosition.y,
        gun_vertical_component: 0, hitbox: [], collision_box: [], collider_y: [],
        visual_opacity: unit.alive ? 1 : 0, wreck_age_s: 0, recoil_age_s: null,
        cosmetic_bob_m: bob, cosmetic_pitch_deg: THREE.MathUtils.radToDeg(turret.rotation.x),
        chassis_roll_deg: THREE.MathUtils.radToDeg(body.rotation.z), cosmetic_retreat_m: 0,
        native_recoil_m: 0, start_bounce_age_s: null, visible: root.visible,
        collider_active: unit.alive, locomotion_mode: lastSample.moving
          ? (lastSample.runBlend > .45 ? 'distance-linked-jog' : 'distance-linked-walk') : 'planted-idle',
        gait_phase_rad: gait.phase, gait_blend: gait.blend,
        step_rate_hz: lastSample.stepRateHz, stride_length_m: lastSample.strideLengthM,
        left_foot_contact: leftFoot.contact && motion > .5,
        right_foot_contact: rightFoot.contact && motion > .5,
        nozzle_yaw_error_deg: 0 };
      return lastReceipt;
    },
    presentationState() { return lastReceipt; },
    labelAnchor(out: THREE.Vector3) { return out.copy(root.position).setY(MODEL_HEIGHT + .15); },
    cameraPivot(out: THREE.Vector3) { return out.copy(root.position).setY(1.34); },
    dispose() {
      const geometries = new Set<THREE.BufferGeometry>();
      const ownedMaterials = new Set<THREE.Material>();
      root.traverse(object => {
        if (!(object instanceof THREE.Mesh)) return;
        geometries.add(object.geometry);
        for (const material of Array.isArray(object.material) ? object.material : [object.material]) ownedMaterials.add(material);
      });
      for (const geometry of geometries) geometry.dispose();
      for (const material of ownedMaterials) material.dispose();
      root.removeFromParent();
    },
  };
}
