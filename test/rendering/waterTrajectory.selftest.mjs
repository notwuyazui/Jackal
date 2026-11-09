import assert from 'node:assert/strict';
import { createWaterArc, sampleWaterArc, WATER_GRAVITY } from '../../render/render3D/src/waterTrajectory.ts';
import { FIREFIGHTER_NOZZLE_REACH_M } from '../../render/render3D/src/firefighterDimensions.ts';

const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-8, `${a} != ${b}`);
const frame = {
  manifest: { meters_per_pixel: .25, map: { size: [960, 640] },
    units: [{ id: 2, shell_lifetime: 6 }] },
  units: [
    { id: 2, team: 'PLAYER', position: [480, 320], alive: true },
    { id: 100, team: 'ENEMY', position: [740, 220], alive: true },
    { id: 101, team: 'ENEMY', position: [740, 320], alive: true },
  ],
};
const bullet = { source_id: 2, team: 'PLAYER', position: [495.7, 320],
  velocity: [90, 0], lifetime: 5.97 };
const arc = createWaterArc(bullet, frame, 1.5);
const start = sampleWaterArc(arc, 0);
const middle = sampleWaterArc(arc, arc.flightTime / 2);
const end = sampleWaterArc(arc, arc.flightTime);
close(start.x, 3.25); close(start.y, 1.5); close(start.z, 0);
close(end.x, 65); close(end.y, 1.6); close(end.z, 0);
assert.ok(middle.y > Math.max(start.y, end.y) + 7, 'higher arc above the fire sites');
assert.ok(start.vy > 0 && end.vy < 0, 'water rises then falls');
close((start.y + end.y - 2 * middle.y) / (arc.flightTime / 2) ** 2, -WATER_GRAVITY);
close(sampleWaterArc(arc, .03).x, (bullet.position[0] - 480) * .25);

// Reconstructing a bullet after a delayed render must give the same trajectory.
const delayed = createWaterArc({ ...bullet, position: [583, 320], lifetime: 5 }, frame, 1.5);
close(delayed.x, arc.x); close(delayed.launchVy, arc.launchVy);
const beforeHead = sampleWaterArc(arc, arc.flightTime - .14);
assert.ok(beforeHead.y > end.y, 'descending tails follow the curved trajectory');
assert.ok(sampleWaterArc(arc, arc.flightTime + 2).y < 0, 'missed jets eventually hit the ground');
console.log('PASS: parabolic rise/fall, target landing, curved tail and delayed-frame reconstruction');

// Every intermediate nozzle angle lands on the line connecting the old/new
// fire sites, rather than snapping to whichever site is currently nearest.
let previousLanding = -Infinity;
const span = Math.atan2(100, 247);
for (let step = 0; step <= 20; step++) {
  const angle = -span + 2 * span * step / 20;
  const velocity = [Math.cos(angle) * 90, Math.sin(angle) * 90];
  const sweeping = { ...bullet, velocity, position: [493 + velocity[0] * .03, 320 + velocity[1] * .03] };
  const path = createWaterArc(sweeping, frame, 1.5);
  const landing = sampleWaterArc(path, path.flightTime);
  close(landing.x, 65); close(landing.y, 1.6);
  assert.ok(landing.z >= -25 - 1e-8 && landing.z <= 25 + 1e-8);
  assert.ok(landing.z > previousLanding, 'landing moves smoothly along the target segment');
  previousLanding = landing.z;
}
console.log('PASS: continuous parabolic nozzle sweep lands on the line between fire sites');

const { setHouseGround } = await import('../../render/render3D/src/firefighterScene.ts');
setHouseGround(7);
const longFrame = { ...frame, manifest: { ...frame.manifest,
  scene: { scenery_size: [960, 640] }, map: { size: [1984, 640] },
  units: [{ id: 2, shell_lifetime: 18 }] }, units: [
    { id: 2, team: 'PLAYER', position: [480, 320], alive: true },
    { id: 100, team: 'ENEMY', position: [1824, 100], alive: true },
    { id: 101, team: 'ENEMY', position: [1824, 120], alive: true },
  ] };
const longArc = createWaterArc({ ...bullet, lifetime: 17.97 }, longFrame, 1.5);
close(sampleWaterArc(longArc, longArc.flightTime).x, 336);
close(sampleWaterArc(longArc, longArc.flightTime).y, 12.5);
assert.ok(longArc.flightTime < 18 && longArc.flightTime > 10, 'retain water speed and reach distant house');
const highPoint = sampleWaterArc(longArc, longArc.flightTime / 2).y;
assert.ok(highPoint > 20 && highPoint < 40, 'high arc stays visible below the sky despite increased range');
const fasterArc = createWaterArc({ ...bullet, velocity: [135, 0], position: [497.05, 320],
  lifetime: 17.97 }, longFrame, 1.5);
close(fasterArc.flightTime, longArc.flightTime / 1.5);
for (const fraction of [0, .25, .5, .75, 1]) {
  const slow = sampleWaterArc(longArc, longArc.flightTime * fraction);
  const fast = sampleWaterArc(fasterArc, fasterArc.flightTime * fraction);
  close(fast.x, slow.x); close(fast.y, slow.y); close(fast.z, slow.z);
}
setHouseGround(0);
console.log('PASS: 50% faster water preserves the elevated-house landing and full parabolic shape');

// Moving the fire to the existing left-front house extends only the engine
// grid. The old scenery origin and each nozzle ray must still agree exactly.
const { readFileSync } = await import('node:fs');
const { worldSize, toWorld } = await import('../../render/render3D/src/mapping.ts');
const state = JSON.parse(readFileSync(new URL('../../engine/scenarios/game_states/border_map_3t_vs_3t.json', import.meta.url)));
const scene = JSON.parse(readFileSync(new URL('../../render/render3D/maps/catalog/border-verdant.json', import.meta.url)));
const houseFrame = { manifest: { scene, meters_per_pixel: .25,
  map: { size: [state.map.tiles[0].length * 64, state.map.tiles.length * 64] },
  units: [{ id: 2, shell_lifetime: 18 }] },
  units: state.units.map(unit => ({ id: unit.unit_id, position: unit.position, team: unit.team, alive: true })) };
const source = houseFrame.units.find(unit => unit.id === 2);
const sites = houseFrame.units.filter(unit => unit.team === 'ENEMY');
const origin = [source.position[0] + 52, source.position[1]];
let lastLanding = -Infinity;
for (let step = 0; step <= 20; step++) {
  const aim = [sites[0].position[0], sites[0].position[1] +
    (sites[2].position[1] - sites[0].position[1]) * step / 20];
  const angle = Math.atan2(aim[1] - origin[1], aim[0] - origin[0]);
  const vx = Math.cos(angle) * 135, vz = Math.sin(angle) * 135;
  const jet = createWaterArc({ source_id: 2, team: 'PLAYER', velocity: [vx, vz], lifetime: 17.97,
    position: [origin[0] + Math.cos(angle) * 13 + vx * .03,
      origin[1] + Math.sin(angle) * 13 + vz * .03] }, houseFrame, 1.5);
  const landing = sampleWaterArc(jet, jet.flightTime);
  const expected = toWorld(...aim, ...worldSize(houseFrame.manifest), .25);
  close(landing.x, expected.x); close(landing.z, expected.z); close(landing.y, 5.5);
  assert.ok(landing.z > lastLanding, 'nozzle sweep stays continuous on the new left-front house');
  lastLanding = landing.z;
}
console.log('PASS: shifted engine origin preserves every intermediate landing on the left-front house');

// The engine still emits from a tank-sized invisible proxy. Presentation must
// relocate the visual birth to the firefighter's hand nozzle.
const diagonal = [.8, .6];
const humanFrame = { ...houseFrame, manifest: { ...houseFrame.manifest,
  units: [{ id: 2, shell_lifetime: 18, size: [32, 46] }] } };
const proxyBirth = [source.position[0] + diagonal[0] * 21,
  source.position[1] + diagonal[1] * 28];
const humanJet = createWaterArc({ source_id: 2, team: 'PLAYER',
  velocity: [diagonal[0] * 135, diagonal[1] * 135], lifetime: 17.97,
  position: [proxyBirth[0] + diagonal[0] * 135 * .03,
    proxyBirth[1] + diagonal[1] * 135 * .03] }, humanFrame, 1.58);
const humanStart = sampleWaterArc(humanJet, 0);
const humanCenter = toWorld(...source.position, ...worldSize(humanFrame.manifest), .25);
close(humanStart.x, humanCenter.x + diagonal[0] * FIREFIGHTER_NOZZLE_REACH_M);
close(humanStart.z, humanCenter.z + diagonal[1] * FIREFIGHTER_NOZZLE_REACH_M);
console.log('PASS: tank proxy emission is visually relocated to the firefighter hand nozzle');
