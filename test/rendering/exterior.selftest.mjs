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
