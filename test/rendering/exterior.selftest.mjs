import assert from 'node:assert/strict';
import { outsideMap, planExterior } from '../../render/render3D/src/exteriorLayout.ts';
import { onRoad, roadRelief, routes } from '../../render/render3D/src/roads.ts';

const bounds = { width: 240, height: 160 };
const scenery = planExterior(42, bounds);
assert.deepEqual(planExterior(42, bounds), scenery, 'a map seed must reproduce its scenery');
assert.notDeepEqual(planExterior(43, bounds).trees, scenery.trees);
assert.equal(scenery.trees.length, 3450);
assert.equal(scenery.rocks.length, 210);
assert.equal(scenery.houses.length, 12);
assert.equal(new Set(scenery.houses.map(h => h.kind)).size, 6);
assert.equal(scenery.poles.length, 94);
assert.equal(scenery.hay.length, 46);
assert.equal(scenery.boundaryFences.length, 89);
assert.equal(scenery.boundaryRocks.length, 26);
assert.equal(scenery.boundaryHouses.length, 2);
for (const kind of ['trees', 'rocks', 'houses', 'poles', 'hay']) {
  const records = scenery[kind];
  for (const record of records) {
    assert.ok(outsideMap(record.x, record.z, bounds, 12), `${kind} entered the Jackal map`);
    assert.ok(Number.isFinite(record.x) && Number.isFinite(record.z));
  }
}
for (const kind of ['boundaryFences', 'boundaryRocks', 'boundaryHouses']) {
  for (const record of scenery[kind]) {
    assert.ok(!outsideMap(record.x, record.z, bounds), `${kind} left the border tile`);
    assert.ok(Math.abs(record.x) >= bounds.width / 2 - 16 ||
      Math.abs(record.z) >= bounds.height / 2 - 16, `${kind} left the border band`);
  }
}
for (const kind of ['trees', 'rocks', 'houses', 'hay',
  'boundaryFences', 'boundaryRocks', 'boundaryHouses']) {
  for (const record of scenery[kind])
    assert.ok(!onRoad(record.x, record.z), `${kind} was placed on a dirt road`);
}
for (const route of routes) {
  assert.ok(roadRelief(1.55, route.width) > 0, 'a rut must not disappear below the Jackal plane');
  assert.ok(roadRelief(2.8, route.width) - roadRelief(1.55, route.width) > 0.11,
    'wheel tracks must be modeled as troughs, not only painted dark');
}
assert.equal(outsideMap(0, 0, bounds), false);
assert.equal(outsideMap(120, 0, bounds), true);
console.log('PASS: reproducible scenery stays beyond Jackal combat bounds');

const { moveBurningHouse, BURNING_HOUSE, ORIGINAL_BURNING_HOUSE, RELOCATED_BATHHOUSE,
  mountainCoordinate, sceneryManifest } =
  await import('../../render/render3D/src/firefighterScene.ts');
const { worldSize, toWorld } = await import('../../render/render3D/src/mapping.ts');
const { readFileSync } = await import('node:fs');
const moved = moveBurningHouse(scenery);
for (const key of Object.keys(scenery).filter(key => key !== 'boundaryHouses' && key !== 'houses'))
  assert.deepEqual(moved[key], scenery[key], `${key} must keep its authored positions`);
assert.deepEqual(moved.houses, scenery.houses.filter(house => !(house.kind === 'school' &&
  house.x === ORIGINAL_BURNING_HOUSE.x && house.z === ORIGINAL_BURNING_HOUSE.z)),
  'remove the fire-centre schoolhouse and preserve the other buildings');
assert.deepEqual(moved.boundaryHouses[0], scenery.boundaryHouses[0]);
assert.deepEqual(moved.boundaryHouses[1], { ...scenery.boundaryHouses[1], ...RELOCATED_BATHHOUSE });
for (const extent of [120, 80]) {
  for (const old of [-1200, -600, -extent, 0, extent, 600, 1200]) {
    const pulled = mountainCoordinate(old, extent);
    assert.equal(Math.max(0, Math.abs(pulled) - extent), Math.max(0, Math.abs(old) - extent) / 4);
    assert.equal(mountainCoordinate(pulled, extent, true), old);
  }
}
const state = JSON.parse(readFileSync(new URL('../../engine/scenarios/game_states/border_map_3t_vs_3t.json', import.meta.url)));
const scene = JSON.parse(readFileSync(new URL('../../render/render3D/maps/catalog/border-verdant.json', import.meta.url)));
const manifest = { scene, meters_per_pixel: .25, map: { ...state.map,
  size: [state.map.tiles[0].length * 64, state.map.tiles.length * 64] } };
assert.deepEqual(worldSize(manifest), [960, 1536]);
assert.deepEqual(sceneryManifest(manifest).map.tiles, ['x'.repeat(15), ...Array(8).fill(`x${'o'.repeat(13)}x`), 'x'.repeat(15)]);
assert.deepEqual(toWorld(400, 768, ...worldSize(manifest), .25), { x: -20, z: 0 });
assert.ok(BURNING_HOUSE.x > 120, 'relocated house outside original map');
assert.ok(BURNING_HOUSE.x + 10 < (manifest.map.size[0] - 480 - 64) * .25, 'entire house within extended front');
for (const unit of state.units) {
  if (unit.team === 'PLAYER') assert.deepEqual(unit.movement_bounds, [64, 512, 896, 1024]);
  else {
    const point = toWorld(...unit.position, ...worldSize(manifest), .25);
    assert.equal(point.x, BURNING_HOUSE.x - 4);
    assert.ok(Math.abs(point.z - BURNING_HOUSE.z) <= 5, 'all targets belong to one house');
  }
}
assert.ok(!moved.houses.some(house => house.x === BURNING_HOUSE.x && house.z === BURNING_HOUSE.z),
  'only the fire-centre building is removed');
const crewDistances = state.units.filter(unit => unit.team === 'PLAYER').map(unit => {
  const staged = toWorld(unit.position[0] + 52, unit.position[1], ...worldSize(manifest), .25);
  return Math.hypot(staged.x - BURNING_HOUSE.x, staged.z - BURNING_HOUSE.z);
});
assert.ok(Math.abs(crewDistances[1] - 30) < 1e-8, 'crew #2 starts spraying 30 m from the house centre');
assert.ok(crewDistances.every(distance => distance >= 30 && distance <= 33), 'the other crew members are also about 30 m away');
assert.notDeepEqual(BURNING_HOUSE, RELOCATED_BATHHOUSE, 'previous mountain-foot house is no longer the fire source');
const top = (0 - worldSize(manifest)[1] / 2 + 64) * .25;
assert.ok(BURNING_HOUSE.z - 15 > top, 'new left-front fire fits within extended northern map');
console.log('PASS: centre house removed, restored mountain layout, unchanged fire anchor and 30 m crew standoff');
