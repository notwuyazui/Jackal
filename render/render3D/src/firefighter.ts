/** Procedural turnout-clad firefighter driven by Jackal's invisible tank proxy. */
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { solveLegPose, type FirefighterGaitSample } from './firefighterGait.ts';
import { createFirefighterLocomotion } from './firefighterLocomotion.ts';
import { createSupplyHose } from './supplyHose.ts';
import { FIREFIGHTER_HEIGHT_M, FIREFIGHTER_RIG_HEIGHT_M, FIREFIGHTER_MODEL_SCALE,
  FIREFIGHTER_MUZZLE_HEIGHT_M } from './firefighterDimensions.ts';
import { effectiveParameters, toWorld, toYaw, worldSize } from './mapping.ts';
import type { Manifest, Parameters, RenderReceipt, UnitFrame } from './protocol.ts';

export const prepareFirefighter = async () => undefined;
export const TEAM_CAMO = { PLAYER: 'detailed-turnout-firefighter', ENEMY: 'burning-house-aim-points' } as const;

const HIP_HEIGHT = .98;
const UPPER_LEG = .43;
const LOWER_LEG = .43;
const MODEL_HEIGHT = FIREFIGHTER_HEIGHT_M;

interface LegRig {
  hip: THREE.Group;
  knee: THREE.Group;
  ankle: THREE.Group;
}

/** Smooth sewn volumes with small cloth folds, rather than stacked primitives. */
function garment(profile: Array<[number, number]>, depth = 1, folds = .0025) {
  const curve = new THREE.SplineCurve(profile.map(([r, y]) => new THREE.Vector2(r, y)));
  const geometry = new THREE.LatheGeometry(curve.getPoints(profile.length * 5)
    .map(p => new THREE.Vector2(Math.max(0, p.x), p.y)), 40);
  const p = geometry.getAttribute('position');
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i), a = Math.atan2(x, z);
    const radius = Math.hypot(x, z);
    const crease = radius > .01 ? 1 + folds / radius *
      (Math.sin(a * 5 + y * 17) * .6 + Math.sin(y * 42 + a * 2) * .4) : 1;
    p.setXYZ(i, x * crease, y, z * crease * depth);
  }
  geometry.computeVertexNormals(); return geometry;
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
  root.scale.setScalar(FIREFIGHTER_MODEL_SCALE);
  root.name = `Jackal #${parameters.id} / procedural firefighter`;
  root.userData = { unitId: parameters.id, modelId: 'firefighter', team,
    gameplaySpec, physicsOwner: 'Jackal-invisible-proxy' };

  const materials = {
    coat: new THREE.MeshStandardMaterial({ color: 0x6f4a29, roughness: .8, metalness: .02 }),
    coatDark: new THREE.MeshStandardMaterial({ color: 0x3e2d22, roughness: .88 }),
    reflective: new THREE.MeshStandardMaterial({ color: 0xd9db78, emissive: 0x77791f,
      emissiveIntensity: .08, roughness: .7, metalness: .05 }),
    reflectiveSilver: new THREE.MeshStandardMaterial({ color: 0xbfc6b9, roughness: .62, metalness: .18 }),
    silver: new THREE.MeshStandardMaterial({ color: 0xd6ddd5, roughness: .28, metalness: .48 }),
    helmet: new THREE.MeshStandardMaterial({ color: 0xc82120, roughness: .34, metalness: .12 }),
    helmetDark: new THREE.MeshStandardMaterial({ color: 0x56120f, roughness: .5 }),
    skin: new THREE.MeshStandardMaterial({ color: 0xb77b5f, roughness: .72 }),
    rubber: new THREE.MeshStandardMaterial({ color: 0x16191b, roughness: .72 }),
    pack: new THREE.MeshStandardMaterial({ color: 0x263137, roughness: .58, metalness: .12 }),
    cylinder: new THREE.MeshStandardMaterial({ color: 0xd7d0a6, roughness: .34, metalness: .68 }),
    brass: new THREE.MeshStandardMaterial({ color: 0xb98a36, roughness: .3, metalness: .75 }),
    hose: new THREE.MeshStandardMaterial({ color: 0xc19b51, roughness: .86 }),
    glass: new THREE.MeshPhysicalMaterial({ color: 0x9fd6e8, transparent: true, opacity: .34,
      roughness: .08, metalness: 0, clearcoat: 1, depthWrite: false }),
  };
  const fabricPixels = new Uint8Array(96 * 96 * 4);
  for (let y = 0; y < 96; y++) for (let x = 0; x < 96; x++) {
    const value = 126 + (x % 3 === 0 ? 18 : -5) + (y % 3 === 0 ? 12 : -4);
    const offset = (y * 96 + x) * 4;
    fabricPixels.set([value, value, value, 255], offset);
  }
  const fabric = new THREE.DataTexture(fabricPixels, 96, 96, THREE.RGBAFormat);
  fabric.wrapS = fabric.wrapT = THREE.RepeatWrapping; fabric.repeat.set(9, 9); fabric.needsUpdate = true;
  for (const material of [materials.coat, materials.coatDark]) {
    material.bumpMap = fabric; material.bumpScale = .0012;
  }
  materials.hose.bumpMap = fabric; materials.hose.bumpScale = .0008;

  const body = new THREE.Group();
  body.name = 'firefighter_hips';
  body.position.y = HIP_HEIGHT;
  root.add(body);
  const turret = new THREE.Group();
  turret.name = 'firefighter_aiming_torso';
  turret.position.y = .08;
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
  add(garment([[0, -.10], [.17, -.10], [.20, -.04], [.195, .07], [.16, .11], [0, .11]], .78),
    materials.coatDark, body, 'anatomical pelvis and trouser seat', [0, 0, 0]);
  add(garment([[0, -.13], [.225, -.13], [.25, -.10], [.253, .03], [.233, .14],
    [.213, .26], [.23, .38], [.265, .46], [.26, .49], [.20, .54], [.108, .57], [0, .57]], .72),
    materials.coat, turret, 'tailored turnout jacket shoulders chest waist and hem', [0, 0, 0]);
  add(new THREE.CylinderGeometry(.238, .247, .045, 40, 1, true), materials.reflective, turret,
    'waist reflective fabric band', [0, .12, 0], [1, 1, .725]);
  add(new THREE.CylinderGeometry(.243, .245, .019, 40, 1, true), materials.reflectiveSilver, turret,
    'waist silver reflective tape', [0, .12, 0], [1, 1, .727]);
  add(new THREE.CylinderGeometry(.235, .232, .038, 40, 1, true), materials.reflective, turret,
    'chest reflective fabric band', [0, .37, 0], [1, 1, .725]);
  add(new THREE.CylinderGeometry(.235, .234, .016, 40, 1, true), materials.reflectiveSilver, turret,
    'chest silver reflective tape', [0, .37, 0], [1, 1, .727]);
  for (const side of [-1, 1]) {
    add(new THREE.BoxGeometry(.04, .48, .025), materials.reflective, turret,
      `front reflective suspender ${side}`, [side * .105, .31, .178], [1, 1, 1], [0, 0, side * .07]);
    add(new RoundedBoxGeometry(.12, .14, .028, 3, .008), materials.coatDark, turret,
      `stitched cargo pocket ${side}`, [side * .145, .015, .165]);
    add(new RoundedBoxGeometry(.133, .032, .030, 3, .006), materials.coat, turret,
      `pocket flap ${side}`, [side * .145, .074, .181]);
    add(new THREE.SphereGeometry(.008, 10, 8), materials.brass, turret,
      `pocket fastener ${side}`, [side * .155, .065, .198]);
  }
  add(new THREE.CylinderGeometry(.004, .004, .42, 10), materials.silver, turret,
    'front jacket zipper', [0, .26, .182]);
  add(new RoundedBoxGeometry(.092, .13, .035, 3, .008), materials.pack, turret,
    'shoulder radio and fabric holster', [-.17, .40, .172]);
  add(new THREE.CylinderGeometry(.004, .004, .14, 10), materials.rubber, turret,
    'radio antenna', [-.195, .52, .176]);
  for (const side of [-1, 1]) add(new RoundedBoxGeometry(.047, .031, .015, 3, .003), materials.silver,
    turret, 'SCBA webbing buckle', [side * .13, .255, .204]);
  add(garment([[.10, 0], [.108, .02], [.095, .09], [.082, .10]], .88, .001), materials.coatDark,
    turret, 'soft protective standing collar', [0, .54, 0]);

  // SCBA frame, air cylinder, straps and pressure gauge.
  add(new THREE.BoxGeometry(.34, .46, .10), materials.pack, turret,
    'SCBA backplate', [0, .30, -.205]);
  add(new THREE.CapsuleGeometry(.086, .36, 10, 28), materials.cylinder, turret,
    'rounded composite SCBA air cylinder', [0, .31, -.315]);
  for (const height of [.18, .43]) add(new THREE.CylinderGeometry(.09, .09, .045, 28), materials.rubber,
    turret, 'SCBA retaining strap', [0, height, -.315]);
  add(new THREE.CylinderGeometry(.10, .10, .035, 16), materials.brass, turret,
    'SCBA valve', [0, .565, -.315]);
  for (const side of [-1, 1]) add(new THREE.BoxGeometry(.045, .49, .035), materials.rubber, turret,
    `SCBA shoulder strap ${side}`, [side * .13, .34, .185], [1, 1, 1], [0, 0, side * .16]);
  add(new THREE.CylinderGeometry(.042, .042, .025, 14), materials.glass, turret,
    'pressure gauge', [.245, .21, .13], [1, 1, 1], [Math.PI / 2, 0, 0]);

  // Neck, face, ear protection, helmet shell, brim, crest and visor.
  add(new THREE.CylinderGeometry(.082, .092, .12, 14), materials.coatDark, turret,
    'protective hood neck', [0, .625, 0]);
  const head = new THREE.Group(); head.name = 'helmeted head'; head.position.y = .70; turret.add(head);
  add(garment([[0, -.13], [.06, -.125], [.089, -.08], [.101, -.015], [.103, .06],
    [.085, .115], [.045, .14], [0, .145]], .92, 0), materials.skin, head,
    'human jaw cheekbones brow and cranium', [0, 0, .025]);
  for (const side of [-1, 1]) {
    add(new THREE.SphereGeometry(.023, 18, 12), materials.skin, head, 'ear', [side * .103, -.015, .02], [.5, 1, .72]);
    add(new THREE.SphereGeometry(.015, 16, 12), materials.rubber, head, 'eye socket', [side * .042, .022, .115], [1.2, .52, .4]);
    add(new THREE.SphereGeometry(.009, 14, 10), materials.skin, head, 'brow ridge', [side * .041, .045, .119], [1.9, .55, .6]);
  }
  add(new THREE.SphereGeometry(.021, 18, 14), materials.skin, head, 'nose bridge', [0, -.007, .124], [.55, 1.2, .85]);
  add(new THREE.SphereGeometry(.18, 22, 14, 0, Math.PI * 2, 0, Math.PI * .66), materials.helmet,
    head, 'red fire helmet shell', [0, .075, -.005], [1.04, .78, 1.10]);
  add(new THREE.CylinderGeometry(.225, .225, .035, 24), materials.helmetDark,
    head, 'helmet brim', [0, .045, 0]);
  add(new THREE.CapsuleGeometry(.024, .23, 8, 18), materials.helmet, head,
    'rounded helmet crown reinforcement', [0, .165, -.015], [1, 1, .75], [Math.PI / 2, 0, 0]);
  const visor = add(new THREE.CylinderGeometry(.164, .155, .165, 36, 1, true, -.95, 1.90), materials.glass, head,
    'curved wraparound face shield', [0, -.006, .005]);
  visor.material.side = THREE.DoubleSide;
  add(new THREE.SphereGeometry(.075, 28, 18), materials.rubber, head,
    'moulded respirator face seal', [0, -.065, .105], [1, .8, .65]);
  add(new THREE.CylinderGeometry(.026, .031, .035, 20), materials.pack, head,
    'respirator demand valve', [0, -.07, .16], [1, 1, 1], [Math.PI / 2, 0, 0]);
  add(new THREE.SphereGeometry(.025, 20, 12), materials.silver, head,
    'helmet front shield', [0, .084, .178], [.8, 1.05, .18]);
  const chinStrap = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-.14, .035, .01), new THREE.Vector3(-.105, -.09, .075),
    new THREE.Vector3(0, -.14, .09), new THREE.Vector3(.105, -.09, .075), new THREE.Vector3(.14, .035, .01),
  ]);
  add(new THREE.TubeGeometry(chinStrap, 24, .006, 8, false), materials.rubber, head, 'helmet chin strap', [0, 0, 0]);
  add(new THREE.CylinderGeometry(.022, .022, .12, 18), materials.pack, head,
    'helmet side flashlight', [-.175, .06, .025], [1, 1, 1], [Math.PI / 2, 0, 0]);

  function createLeg(side: number): LegRig {
    const hip = new THREE.Group(); hip.name = `${side < 0 ? 'left' : 'right'} hip`;
    hip.position.x = side * .14; body.add(hip);
    hip.rotation.order = 'ZXY';
    add(garment([[0, .035], [.09, .03], [.108, -.03], [.112, -.13], [.101, -.27],
      [.083, -.39], [.075, -.43], [0, -.43]], .89, .003), materials.coat, hip,
      'tailored trouser thigh', [0, 0, 0]);
    add(new THREE.CylinderGeometry(.097, .091, .045, 36, 1, true), materials.reflective, hip,
      'sewn thigh reflective band', [0, -.31, 0], [1, 1, .90]);
    add(new THREE.CylinderGeometry(.095, .093, .019, 36, 1, true), materials.reflectiveSilver, hip,
      'thigh silver reflective tape', [0, -.31, 0], [1, 1, .905]);
    const knee = new THREE.Group(); knee.name = 'articulated knee'; knee.position.y = -UPPER_LEG; hip.add(knee);
    add(new THREE.SphereGeometry(.09, 22, 16), materials.coatDark, knee, 'reinforced knee pad', [0, 0, .06], [.9, 1.1, .42]);
    add(garment([[0, .025], [.083, .02], [.096, -.06], [.095, -.14], [.079, -.29],
      [.075, -.39], [.07, -.43], [0, -.43]], .92, .003), materials.coat, knee,
      'shaped trouser calf and cuff', [0, 0, 0]);
    add(new THREE.CylinderGeometry(.083, .079, .05, 36, 1, true), materials.reflective, knee,
      'sewn ankle reflective band', [0, -.31, 0], [1, 1, .93]);
    add(new THREE.CylinderGeometry(.082, .080, .021, 36, 1, true), materials.reflectiveSilver, knee,
      'ankle silver reflective tape', [0, -.31, 0], [1, 1, .935]);
    const ankle = new THREE.Group(); ankle.name = 'articulated ankle'; ankle.position.y = -LOWER_LEG; knee.add(ankle);
    add(new THREE.SphereGeometry(1, 28, 18), materials.rubber, ankle,
      'anatomical reinforced boot toe and instep', [0, -.035, .065], [.10, .083, .18]);
    add(new THREE.CapsuleGeometry(.076, .10, 8, 22), materials.rubber, ankle,
      'leather boot shaft', [0, .045, -.02], [1, 1, .9]);
    add(new THREE.CapsuleGeometry(.102, .15, 8, 24), materials.coatDark, ankle,
      'rounded profiled boot sole', [0, -.113, .065], [1, 1, .16], [Math.PI / 2, 0, 0]);
    return { hip, knee, ankle };
  }
  const leftLeg = createLeg(-1), rightLeg = createLeg(1);

  const armGeometry = garment([[0, -.5], [.061, -.5], [.072, -.34], [.078, -.05],
    [.085, .30], [.072, .47], [0, .50]], 1, .004);
  const forearmGeometry = garment([[0, -.5], [.050, -.5], [.062, -.36], [.072, -.06],
    [.070, .33], [.059, .48], [0, .50]], 1, .004);
  const stripeGeometry = new THREE.CylinderGeometry(.080, .076, 1, 28, 1, true);
  const jointGeometry = new THREE.SphereGeometry(.077, 22, 16);
  const gloveGeometry = new THREE.SphereGeometry(.065, 24, 16);
  function createArm(side: number): ArmRig {
    const upper = add(armGeometry, materials.coat, turret, `${side < 0 ? 'left' : 'right'} upper arm`, [0, 0, 0]);
    const lower = add(forearmGeometry, materials.coat, turret, 'forearm', [0, 0, 0]);
    const stripe = add(stripeGeometry, materials.reflective, turret, 'sleeve reflective stripe', [0, 0, 0]);
    const elbow = add(jointGeometry, materials.coat, turret, 'soft reinforced elbow', [0, 0, 0], [1, .9, .8]);
    const glove = add(gloveGeometry, materials.rubber, turret, 'shaped leather hose glove', [0, 0, 0], [.85, 1.1, .72]);
    for (let finger = 0; finger < 4; finger++) add(new THREE.CapsuleGeometry(.012, .035, 6, 12), materials.rubber,
      glove, 'curled glove finger', [(finger - 1.5) * .023, -.045, .025], [1, 1, 1], [.75, 0, 0]);
    add(new THREE.CapsuleGeometry(.018, .035, 6, 12), materials.rubber,
      glove, 'gripping thumb', [side * -.047, .005, .025], [1, 1, 1], [.45, 0, side * .5]);
    return { upper, lower, stripe, elbow, glove };
  }
  const leftArm = createArm(-1), rightArm = createArm(1);

  const nozzle = add(new THREE.CylinderGeometry(.05, .065, 1, 16), materials.silver, turret,
    'metal water nozzle', [0, 0, 0]);
  const nozzleTip = add(new THREE.CylinderGeometry(.085, .062, 1, 16), materials.brass, turret,
    'brass nozzle tip', [0, 0, 0]);
  add(new THREE.CircleGeometry(.052, 32), materials.rubber, nozzleTip,
    'dark open water nozzle bore', [0, .501, 0], [1, 1, 1], [-Math.PI / 2, 0, 0]);
  const nozzleGrip = add(new THREE.BoxGeometry(.16, .24, .055), materials.rubber, turret,
    'nozzle pistol grip', [0, 0, 0]);
  const muzzleMarker = new THREE.Object3D(); muzzleMarker.name = 'water nozzle outlet'; turret.add(muzzleMarker);
  const hoseAttachment = new THREE.Object3D();
  hoseAttachment.name = 'supply hose coupling at nozzle rear'; turret.add(hoseAttachment);
  const supplyHose = createSupplyHose(root, materials.hose);

  const tempA = new THREE.Vector3(), tempB = new THREE.Vector3(), tempC = new THREE.Vector3();
  function poseArm(rig: ArmRig, side: number, sample: FirefighterGaitSample, spraying: boolean) {
    const gaitSway = Math.sin(gait.phase + (side < 0 ? 0 : Math.PI)) * gait.blend * .014;
    const brace = spraying ? 1 : 0;
    const shoulder = tempA.set(side * .265, .48, -.005);
    const elbow = tempB.set(side * (.30 + brace * .015), .285 + gaitSway, .28 - brace * .015);
    const hand = tempC.set(side * .075, .335 + gaitSway * .35, .55 + (side < 0 ? -.035 : .035));
    setSegment(rig.upper, shoulder, elbow);
    setSegment(rig.lower, elbow, hand);
    setSegment(rig.stripe, pointBetween(shoulder, elbow, .58), pointBetween(shoulder, elbow, .76));
    rig.elbow.position.copy(elbow); rig.glove.position.copy(hand);
  }

  function poseNozzle(sample: FirefighterGaitSample, spraying: boolean) {
    const settle = gait.blend * Math.sin(gait.phase * 2) * .006;
    const rear = new THREE.Vector3(-.015, .295 + settle, .43);
    const front = new THREE.Vector3(0, .52, .98);
    setSegment(nozzle, rear, front);
    setSegment(nozzleTip, pointBetween(rear, front, .84), front);
    nozzleTip.scale.x = nozzleTip.scale.z = 1;
    nozzleGrip.position.copy(pointBetween(rear, front, .25));
    nozzleGrip.rotation.copy(nozzle.rotation);
    nozzleGrip.rotateX(.42);
    muzzleMarker.position.copy(front);
    hoseAttachment.position.copy(rear);
  }

  root.traverse(object => {
    if (object instanceof THREE.Mesh) { object.castShadow = true; object.receiveShadow = true; }
  });

  const locomotion = createFirefighterLocomotion(FIREFIGHTER_MODEL_SCALE);
  const gait = locomotion.gait;
  let lastSprayAt = -Infinity;
  let lastReceipt: RenderReceipt | null = null;
  let lastSample: FirefighterGaitSample = { moving: false, locomotionMps: 0, strideLengthM: .68,
    cycleRateHz: 0, stepRateHz: 0, runBlend: 0 };
  const muzzleWorldPosition = new THREE.Vector3();
  let lastHosePoseTime = -Infinity;

  const localFoot = new THREE.Vector3(), measuredFoot = new THREE.Vector3();
  const desiredFootRotation = new THREE.Quaternion(), parentRotation = new THREE.Quaternion();
  const upAxis = new THREE.Vector3(0, 1, 0);
  function poseLeg(rig: LegRig, foot: typeof locomotion.feet[number]) {
    body.worldToLocal(localFoot.copy(foot.position));
    localFoot.x -= rig.hip.position.x;
    const down = -localFoot.y;
    const solved = solveLegPose(localFoot.z, Math.hypot(down, localFoot.x), UPPER_LEG, LOWER_LEG);
    rig.hip.rotation.y = 0;
    rig.hip.rotation.z = Math.atan2(localFoot.x, down);
    rig.hip.rotation.x = solved.hip;
    rig.knee.rotation.x = solved.knee;
    // Counter-rotate the ankle against the complete parent transform. The
    // soles stay level even during lateral steps and pelvis rotation.
    rig.knee.getWorldQuaternion(parentRotation);
    desiredFootRotation.setFromAxisAngle(upAxis, foot.yaw);
    rig.ankle.quaternion.copy(parentRotation).invert().multiply(desiredFootRotation);
    rig.ankle.getWorldPosition(measuredFoot);
    return measuredFoot.distanceTo(foot.position);
  }

  return {
    root, body, turret, gameplaySpec, modelHeight: MODEL_HEIGHT,
    get muzzleHeight() { return FIREFIGHTER_MUZZLE_HEIGHT_M; },
    muzzleWorld(out: THREE.Vector3) { root.updateMatrixWorld(true); return muzzleMarker.getWorldPosition(out); },
    fire(time: number) { lastSprayAt = time; },
    pose(unit: UnitFrame, time: number, _physicsDt = 0): RenderReceipt {
      const position = toWorld(...unit.position, ...worldSize(manifest), scale);
      const motionState = locomotion.update(tempA.set(position.x, 0, position.z), toYaw(unit.turret_heading), time, unit.alive);
      root.position.copy(locomotion.position);
      root.rotation.set(0, motionState.yaw, 0);
      root.visible = unit.alive;

      lastSample = motionState.sample;
      const motion = gait.blend;
      const hipInRig = motionState.hipHeight / FIREFIGHTER_MODEL_SCALE;
      const bob = hipInRig - HIP_HEIGHT;
      body.position.y = hipInRig;
      body.rotation.set(0, 0, 0);
      const [leftFoot, rightFoot] = locomotion.feet;

      const spraying = time - lastSprayAt < .22;
      turret.position.y = .08 - bob;
      turret.rotation.set(0, motionState.aimOffset, 0);
      poseArm(leftArm, -1, lastSample, spraying); poseArm(rightArm, 1, lastSample, spraying);
      poseNozzle(lastSample, spraying);
      root.updateMatrixWorld(true);
      const leftError = poseLeg(leftLeg, leftFoot), rightError = poseLeg(rightLeg, rightFoot);
      root.updateMatrixWorld(true);
      muzzleMarker.getWorldPosition(muzzleWorldPosition);
      if (time !== lastHosePoseTime) {
        supplyHose.update(hoseAttachment.getWorldPosition(new THREE.Vector3()), motionState.yaw, time);
        lastHosePoseTime = time;
      }

      lastReceipt = { id: unit.id, x: unit.position[0], y: unit.position[1],
        heading: unit.heading, turret_heading: unit.turret_heading,
        health: unit.health, alive: unit.alive,
        body_width: .72 * FIREFIGHTER_MODEL_SCALE / scale, body_length: .82 * FIREFIGHTER_MODEL_SCALE / scale,
        model_scale: [FIREFIGHTER_MODEL_SCALE, FIREFIGHTER_MODEL_SCALE, FIREFIGHTER_MODEL_SCALE],
        native_body_size_m: [.72, FIREFIGHTER_RIG_HEIGHT_M, .82],
        body_size_m: [.72 * FIREFIGHTER_MODEL_SCALE, MODEL_HEIGHT, .82 * FIREFIGHTER_MODEL_SCALE], muzzle_height_m: muzzleWorldPosition.y,
        gun_vertical_component: 0, hitbox: [], collision_box: [], collider_y: [],
        visual_opacity: unit.alive ? 1 : 0, wreck_age_s: 0, recoil_age_s: null,
        cosmetic_bob_m: bob * FIREFIGHTER_MODEL_SCALE, cosmetic_pitch_deg: THREE.MathUtils.radToDeg(turret.rotation.x),
        chassis_roll_deg: THREE.MathUtils.radToDeg(body.rotation.z), cosmetic_retreat_m: 0,
        native_recoil_m: 0, start_bounce_age_s: null, visible: root.visible,
        collider_active: unit.alive, locomotion_mode: lastSample.moving
          ? (lastSample.runBlend > .45 ? 'distance-linked-jog' : 'distance-linked-walk') : 'planted-idle',
        gait_phase_rad: gait.phase, gait_blend: gait.blend,
        step_rate_hz: lastSample.stepRateHz, stride_length_m: lastSample.strideLengthM,
        left_foot_contact: leftFoot.contact,
        right_foot_contact: rightFoot.contact,
        left_foot_world: leftLeg.ankle.getWorldPosition(new THREE.Vector3()).toArray(),
        right_foot_world: rightLeg.ankle.getWorldPosition(new THREE.Vector3()).toArray(),
        foot_ik_error_m: Math.max(leftError, rightError),
        pelvis_yaw_deg: THREE.MathUtils.radToDeg(motionState.yaw),
        torso_twist_deg: THREE.MathUtils.radToDeg(motionState.aimOffset),
        nozzle_yaw_error_deg: 0 };
      return lastReceipt;
    },
    presentationState() { return lastReceipt; },
    labelAnchor(out: THREE.Vector3) { return out.copy(root.position).setY(MODEL_HEIGHT + .15); },
    cameraPivot(out: THREE.Vector3) { return out.copy(root.position).setY(1.34 * FIREFIGHTER_MODEL_SCALE); },
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
      fabric.dispose();
      root.removeFromParent();
    },
  };
}
