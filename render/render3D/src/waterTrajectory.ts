/** Vertical water ballistics above Jackal's authoritative horizontal plane. */
import { worldSize, toWorld } from './mapping.ts';
import { houseGroundHeight } from './firefighterScene.ts';
import type { BulletFrame, Frame } from './protocol.ts';

// Scale with the faster water so the established high arc keeps its shape.
export const WATER_GRAVITY = 8.1;
export interface WaterArc {
  x: number; z: number; vx: number; vz: number;
  height: number; launchVy: number; flightTime: number; lifetime: number;
  gravity: number;
}
export interface WaterSample { x: number; y: number; z: number; vy: number }

export function createWaterArc(bullet: BulletFrame, frame: Frame, height: number): WaterArc {
  const scale = frame.manifest.meters_per_pixel;
  const spec = frame.manifest.units.find(unit => unit.id === bullet.source_id);
  const lifetime = spec?.shell_lifetime ?? bullet.lifetime;
  const age = Math.max(0, lifetime - bullet.lifetime);
  const vx = bullet.velocity[0] * scale, vz = bullet.velocity[1] * scale;
  const speed = Math.max(0.001, Math.hypot(vx, vz));
  const head = toWorld(...bullet.position, ...worldSize(frame.manifest), scale);
  // Recover the actual birth position even if several simulation ticks elapsed
  // before this bullet's first rendered frame. Jackal still launches from its
  // invisible tank-sized proxy; move that birth point back to the firefighter's
  // one-metre hand nozzle without following the moving shooter afterward.
  const dirX = vx / speed;
  const dirZ = vz / speed;
  const proxyOffsetX = spec?.size ? (spec.size[0] / 2 + 5) * scale : 0;
  const proxyOffsetZ = spec?.size ? (spec.size[1] / 2 + 5) * scale : 0;
  const nozzleReachM = spec?.size ? .98 : 0;
  const x = head.x - vx * age - dirX * proxyOffsetX + dirX * nozzleReachM;
  const z = head.z - vz * age - dirZ * proxyOffsetZ + dirZ * nozzleReachM;
  const sites = frame.units.filter(target => target.alive && target.team !== bullet.team);
  let distance = speed * lifetime;
  if (sites.length) {
    // All recording fire sites share a straight front. Intersect the actual
    // firing ray with that front instead of snapping to the nearest fire site;
    // a turning nozzle then sweeps its landing point smoothly along the line.
    const frontX = sites.reduce((sum, target) => sum + target.position[0], 0) / sites.length;
    const front = toWorld(frontX, 0, ...worldSize(frame.manifest), scale);
    const intersection = (front.x - x) / dirX;
    if (Number.isFinite(intersection) && intersection > 0) {
      distance = Math.min(intersection, speed * lifetime);
    }
  }
  const flightTime = Math.max(0.35, distance / speed);
  const house = Boolean(frame.manifest.scene?.scenery_size);
  const targetHeight = house ? houseGroundHeight() + 5.5 : 1.6;
  // The farther house needs a longer flight, not a hundreds-of-metres-high
  // fountain. Keep the established high parabola at a visible 18 m arch.
  const gravity = house ? Math.min(WATER_GRAVITY, 8 * 18 / flightTime ** 2) : WATER_GRAVITY;
  const launchVy = (targetHeight - height) / flightTime + gravity * flightTime / 2;
  return { x, z, vx, vz, height, launchVy, flightTime, lifetime, gravity };
}

export function sampleWaterArc(arc: WaterArc, age: number,
  out: WaterSample = { x: 0, y: 0, z: 0, vy: 0 }): WaterSample {
  const t = Math.max(0, age);
  out.x = arc.x + arc.vx * t;
  out.z = arc.z + arc.vz * t;
  out.y = arc.height + arc.launchVy * t - arc.gravity * t * t / 2;
  out.vy = arc.launchVy - arc.gravity * t;
  return out;
}
