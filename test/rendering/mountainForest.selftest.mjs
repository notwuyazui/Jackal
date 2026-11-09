import assert from 'node:assert/strict';
import * as THREE from '../../render/render3D/node_modules/three/build/three.module.js';
import { partitionForest } from '../../render/render3D/src/forestBatches.ts';
import { buildHorizonForest } from '../../render/render3D/vendor/cot/src/world/horizonVista.ts';

const columns = 16, radii = [512, 650, 800, 1100, 1500, 2000], elevation = [0, 20, 65, 160, 130, 80];
const positions = new Float32Array(radii.length * columns * 3), heights = new Float32Array(radii.length * columns);
for (let row = 0; row < radii.length; row++) for (let c = 0; c < columns; c++) {
  const index = row * columns + c, angle = c / columns * Math.PI * 2;
  positions.set([Math.cos(angle) * radii[row], elevation[row], Math.sin(angle) * radii[row]], index * 3);
  heights[index] = elevation[row];
}
const forest = buildHorizonForest({ columns, rows: radii.map(r => ({ r, aer: 0 })), positions, heights,
  forestCover: new Float32Array(heights.length).fill(1), maxHeight: 160, treeline: 1.4, snowline: 2,
  forest: new THREE.Color(0x2e5027), fog: new THREE.Color(0xadc3cf), seed: 42,
  coniferShare: .24, maxInstances: 40000, maxRadius: 2100, densityScale: 1.5,
  coverSummits: true, ridgeRow: 3 });
assert.ok(forest);
const metadata = forest.userData.horizonForest;
assert.equal(metadata.instances, 40000);
assert.equal(metadata.band, 8000, 'extra trees go to the mountains, not an oversized foothill band');
assert.equal(metadata.range, 32000);
const matrix = new THREE.Matrix4(), color = new THREE.Color();
const before = new Map();
for (const pool of forest.children) for (let i = 0; i < pool.count; i++) {
  pool.getMatrixAt(i, matrix); pool.getColorAt(i, color);
  before.set(matrix.elements.join(','), color.toArray());
}
const chunks = partitionForest(forest);
assert.ok(chunks > 20, 'the 360-degree pools split into local cullable batches');
let count = 0, high = 0;
for (const pool of forest.children) {
  assert.ok(pool.frustumCulled && pool.boundingSphere && pool.boundingBox);
  for (let i = 0; i < pool.count; i++) {
    pool.getMatrixAt(i, matrix); pool.getColorAt(i, color); count++;
    const original = before.get(matrix.elements.join(','));
    assert.ok(original, 'partitioning must preserve each individual tree transform');
    assert.deepEqual(color.toArray(), original);
    high += matrix.elements[13] > 120;
  }
}
assert.equal(count, 40000); assert.ok(high > 5000, 'ridge tops and upper slopes carry trees');
const camera = new THREE.PerspectiveCamera(60, 1.5, 1, 3000);
camera.position.set(0, 80, 0); camera.lookAt(0, 100, -1200); camera.updateMatrixWorld(true);
forest.updateMatrixWorld(true);
const frustum = new THREE.Frustum().setFromProjectionMatrix(
  new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse));
const visible = forest.children.filter(pool => frustum.intersectsObject(pool)).length;
assert.ok(visible > 0 && visible < forest.children.length / 2, 'off-camera mountain forest is culled');
const geometries = new Set(), materials = new Set();
for (const pool of forest.children) { geometries.add(pool.geometry); materials.add(pool.material); pool.dispose(); }
for (const geometry of geometries) geometry.dispose();
for (const material of materials) material.dispose();
console.log('PASS: mountain-focused forest budget, summit coverage, unchanged tree transforms/colors and spatial culling');
