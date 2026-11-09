import assert from 'node:assert/strict';
import * as THREE from '../../render/render3D/node_modules/three/build/three.module.js';
import { createSupplyHose } from '../../render/render3D/src/supplyHose.ts';

const root = new THREE.Group(), material = new THREE.MeshStandardMaterial();
const terrain = (x, z) => .07 + .04 * Math.sin(x * .5) + .025 * Math.cos(z * .7);
const hose = createSupplyHose(root, material, terrain);
const hand = new THREE.Vector3();
let feed = null, maxAttachmentError = 0;
for (let frame = 0; frame <= 180; frame++) {
  const t = frame / 60, yaw = t * .8;
  root.position.set(t * 1.5, 0, Math.sin(t) * 1.2); root.rotation.y = yaw; root.updateMatrixWorld(true);
  hand.set(-.015, 1.35, .43).applyMatrix4(root.matrixWorld);
  hose.update(hand, yaw);
  const state = hose.state();
  if (!feed) feed = state.feed.clone();
  assert.ok(state.feed.distanceTo(feed) < 1e-8, 'the rear supply end must not follow the person');
  const ground = root.getObjectByName('hose laid out behind the firefighter');
  const groundPositions = ground.geometry.getAttribute('position');
  const renderedFeed = new THREE.Vector3();
  for (let side = 0; side < 10; side++) renderedFeed.add(new THREE.Vector3().fromBufferAttribute(groundPositions, side));
  ground.localToWorld(renderedFeed.multiplyScalar(.1));
  assert.ok(renderedFeed.distanceTo(feed) < 1e-5, 'cached ground geometry must stay fixed between updates');
  assert.ok(state.length > 32, 'the hose extends far behind the person instead of ending at their boots');
  for (const point of [...state.groundCenters, ...state.liftedCenters])
    assert.ok(point.y - state.radius >= terrain(point.x, point.z) + .011, 'hose stays above the terrain');
  const raised = root.getObjectByName('hose from the ground to the two-handed nozzle');
  const attribute = raised.geometry.getAttribute('position');
  const endpoint = new THREE.Vector3();
  // Average a complete ring, omitting its duplicated seam vertex.
  const ringStart = attribute.count - 11;
  for (let side = 0; side < 10; side++) endpoint.add(new THREE.Vector3().fromBufferAttribute(attribute, ringStart + side));
  endpoint.multiplyScalar(.1).applyMatrix4(root.matrixWorld);
  maxAttachmentError = Math.max(maxAttachmentError, endpoint.distanceTo(hand));
}
assert.ok(maxAttachmentError < 1e-5, 'turning and moving must not disconnect the hose from the nozzle');
assert.ok(hose.state().trailKnots > 8, 'movement lays a trail rather than rotating the entire hose');
for (const object of hose.group.children) {
  const index = object.geometry.getIndex();
  const p = object.geometry.getAttribute('position'), n = object.geometry.getAttribute('normal');
  const a = new THREE.Vector3().fromBufferAttribute(p, index.getX(0));
  const b = new THREE.Vector3().fromBufferAttribute(p, index.getX(1));
  const c = new THREE.Vector3().fromBufferAttribute(p, index.getX(2));
  const face = b.sub(a).cross(c.sub(a));
  assert.ok(face.dot(new THREE.Vector3().fromBufferAttribute(n, index.getX(0))) > 0,
    'hose faces point outward so it renders as a solid tube, not its inside surface');
}
for (const object of hose.group.children) object.geometry.dispose();
material.dispose();
console.log('PASS: 34 m rear supply, fixed feed point, laid-out movement trail, terrain contact and nozzle attachment');
