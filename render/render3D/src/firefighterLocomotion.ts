/** Grounded, world-space footsteps driven by the displayed body's displacement. */
import * as THREE from 'three';
import { advanceFirefighterGait, createGaitState } from './firefighterGait.ts';

const wrap = (angle: number) => Math.atan2(Math.sin(angle), Math.cos(angle));
const clamp = THREE.MathUtils.clamp;
const smooth = (x: number) => x * x * (3 - 2 * x);
export const BOOT_ANKLE_HEIGHT = .133;
export const FIREFIGHTER_HIP_HEIGHT = .98;

interface Foot {
  position: THREE.Vector3;
  start: THREE.Vector3;
  target: THREE.Vector3;
  yaw: number;
  startYaw: number;
  contact: boolean;
  progress: number;
  duration: number;
}

export function createFirefighterLocomotion(modelScale = 1) {
  const ankleHeight = BOOT_ANKLE_HEIGHT * modelScale;
  const position = new THREE.Vector3(), velocity = new THREE.Vector3();
  const previousPosition = new THREE.Vector3(), nominal = new THREE.Vector3();
  const gait = createGaitState();
  // Begin near mid-stance rather than forcing the first planted foot to stay
  // fixed for an entire half-metre of acceleration.
  gait.phase = Math.PI * .42;
  const feet: Foot[] = [-1, 1].map(() => ({ position: new THREE.Vector3(),
    start: new THREE.Vector3(), target: new THREE.Vector3(), yaw: 0, startYaw: 0,
    contact: true, progress: 1, duration: .25 }));
  let initialized = false, previousTime = 0, yaw = 0;
  let nextFoot = 0, stepProgress = 1;
  let lastResult: { yaw: number; aimOffset: number; speed: number;
    sample: ReturnType<typeof advanceFirefighterGait>; hipHeight: number; lean: number } | null = null;

  function footHome(index: number, out: THREE.Vector3) {
    const side = (index === 0 ? -.135 : .135) * modelScale;
    return out.set(position.x + Math.cos(yaw) * side, ankleHeight,
      position.z - Math.sin(yaw) * side);
  }

  return {
    position, feet, gait,
    update(target: THREE.Vector3, aimYaw: number, time: number, alive: boolean) {
      const dt = clamp(time - previousTime, 0, .05);
      previousTime = time;
      // Ingesting a snapshot and rendering it can occur at the same visual
      // time. A zero-time call must not turn a moving pose into an idle one.
      if (initialized && dt === 0 && lastResult) return lastResult;
      if (!initialized || target.distanceTo(position) > 3) {
        initialized = true; position.copy(target); yaw = aimYaw;
        for (let i = 0; i < 2; i++) {
          footHome(i, feet[i].position); feet[i].yaw = yaw; feet[i].contact = true;
          feet[i].progress = 1;
        }
      }
      previousPosition.copy(position);
      // One short visual delay smooths the 33 Hz authoritative positions into
      // 60 Hz motion. Cadence uses this same displacement, never stale speed.
      if (dt > 0) position.lerp(target, 1 - Math.exp(-dt / .04));
      velocity.subVectors(position, previousPosition);
      if (dt > 0) velocity.multiplyScalar(1 / dt); else velocity.set(0, 0, 0);
      const speed = velocity.length();
      const travelYaw = speed > .04 ? Math.atan2(velocity.x, velocity.z) : aimYaw;
      // Keep the whole pelvis coherent; a sideways step uses hip abduction,
      // not a 90–180 degree rotation of each individual thigh.
      const wantedYaw = speed > .10 ? aimYaw + clamp(wrap(travelYaw - aimYaw), -.95, .95) : aimYaw;
      const change = clamp(wrap(wantedYaw - yaw), -2.6 * dt, 2.6 * dt);
      yaw += change;
      const sample = advanceFirefighterGait(gait, alive ? speed : 0,
        dt > 0 && alive ? THREE.MathUtils.radToDeg(change) / dt : 0, dt);
      const duty = .62 - .24 * sample.runBlend;
      const period = 1 / Math.max(.5, sample.cycleRateHz);
      const duration = clamp(period * (1 - duty), .18, .36);

      if (alive && sample.moving) {
        stepProgress += sample.cycleRateHz * dt * 2;
        const foot = feet[nextFoot]; footHome(nextFoot, nominal);
        const stretched = Math.hypot(foot.position.x - nominal.x, foot.position.z - nominal.z) > .35 * modelScale;
        const turning = Math.abs(wrap(foot.yaw - yaw)) > .45;
        // Always alternate. Independent reach-limit triggers used to pull both
        // feet into the same phase, producing a two-legged hopping motion.
        if (foot.contact && (stepProgress >= 1 || stretched || turning)) {
          foot.contact = false; foot.progress = 0; foot.duration = duration;
          foot.start.copy(foot.position); foot.startYaw = foot.yaw;
          stepProgress = 0; nextFoot = 1 - nextFoot;
        }
      } else {
        stepProgress = 1;
      }

      for (let i = 0; i < 2; i++) {
        const foot = feet[i]; footHome(i, nominal);
        if (!foot.contact) {
          foot.progress = Math.min(1, foot.progress + dt / foot.duration);
          const remaining = (1 - foot.progress) * foot.duration;
          // Predict the next contact, then replan gently under braking. The
          // foot lands beneath the slowing body instead of reaching far ahead.
          foot.target.copy(nominal).addScaledVector(velocity, remaining + period * duty * .50);
          const lead = foot.target.clone().sub(nominal);
          if (lead.length() > .95 * modelScale) foot.target.copy(nominal).addScaledVector(lead.normalize(), .95 * modelScale);
          const u = smooth(foot.progress);
          foot.position.lerpVectors(foot.start, foot.target, u);
          foot.position.y = ankleHeight + Math.sin(Math.PI * foot.progress) ** 2 *
            (.08 + .055 * sample.runBlend) * modelScale;
          const toeYaw = yaw + clamp(wrap(travelYaw - yaw), -.25, .25);
          foot.yaw = foot.startYaw + wrap(toeYaw - foot.startYaw) * u;
          if (foot.progress >= 1) {
            foot.position.y = ankleHeight; foot.contact = true;
          }
        }
      }
      // Stand almost straight at rest. Lower the pelvis only as required by a
      // long step, instead of keeping both knees in a permanent deep squat.
      let hipHeight = FIREFIGHTER_HIP_HEIGHT * modelScale;
      for (let i = 0; i < 2; i++) {
        footHome(i, nominal);
        const span = Math.hypot(feet[i].position.x - nominal.x, feet[i].position.z - nominal.z);
        const reachableHeight = feet[i].position.y + Math.sqrt(Math.max(.01 * modelScale ** 2,
          (.858 * modelScale) ** 2 - span ** 2)) - .002 * modelScale;
        hipHeight = Math.min(hipHeight, reachableHeight);
      }
      lastResult = { yaw, aimOffset: wrap(aimYaw - yaw), speed, sample,
        hipHeight: Math.max(.86 * modelScale, hipHeight),
        lean: sample.runBlend * gait.blend * .055 };
      return lastResult;
    },
  };
}
