/** The original Verdant Fields vista, including its relief atlas and far range. */
import * as THREE from 'three';
import verdant from '@cot/world/maps/verdant.ts';
import { buildHorizonRingSteps, HORIZON_SEGMENTS } from '@cot/world/maps/horizon.ts';
import { disposeObject3DResources } from '@cot/engine/resourceLifetime.ts';
import { createHorizonReliefField, resolveHorizonRelief } from '@cot/world/horizonRelief.ts';
import { onRoad } from './roads.ts';

export async function prepareVerdantMountains(seed: number) {
  const steps = buildHorizonRingSteps(null, {
    ...verdant, horizon: { ...verdant.horizon, farRange: true },
    sky: { ...verdant.sky, sunElevationDeg: 32, sunAzimuthDeg: 115,
      cloudAltM: 1400, sunIntensity: 2, hemiIntensity: 1.1 },
  }, seed, { getHeightAt: () => 0,
    getOutlandSeatWeightAt: (x, z) => onRoad(x, z, 6) ? 1 : 0 });
  let step = steps.next();
  while (!step.done) {
    // Preserve the native generator and RNG order, but avoid one animation
    // frame per tiny bake yield (Safari can throttle those during startup).
    const deadline = performance.now() + 6;
    do { step = steps.next(); } while (!step.done && performance.now() < deadline);
    if (!step.done) await new Promise<void>(resolve => setTimeout(resolve, 0));
  }
  const mesh = step.value;
  const position = mesh.geometry.getAttribute('position');
  const stride = HORIZON_SEGMENTS + 1; // CoT duplicates the closing vertex of every row.
  const rows = position.count / stride;
  const columns = HORIZON_SEGMENTS;
  const point = new THREE.Vector3();
  let forestInstances = 0;
  mesh.traverse(object => {
    if (object instanceof THREE.InstancedMesh) forestInstances += object.count;
  });
  // Sample the actual rendered triangles to seat the bridge's hillside trees.
  function groundAt(x: number, z: number) {
    const radius = Math.hypot(x, z);
    let columnFloat = Math.atan2(z, x) / (Math.PI * 2) * columns;
    if (columnFloat < 0) columnFloat += columns;
    const col = Math.floor(columnFloat), next = col + 1;
    const fraction = columnFloat - col;
    const radiusAt = (row: number) => Math.hypot(
      THREE.MathUtils.lerp(position.getX(row * stride + col), position.getX(row * stride + next), fraction),
      THREE.MathUtils.lerp(position.getZ(row * stride + col), position.getZ(row * stride + next), fraction));
    if (radius <= radiusAt(0)) return 0;
    let row = 0;
    while (row < rows - 2 && radius > radiusAt(row + 1)) row++;
    const a = row * stride + col, b = row * stride + next;
    const c = (row + 1) * stride + col, d = (row + 1) * stride + next;
    function triangle(i: number, j: number, k: number) {
      const ax = position.getX(i), az = position.getZ(i);
      const bx = position.getX(j) - ax, bz = position.getZ(j) - az;
      const cx = position.getX(k) - ax, cz = position.getZ(k) - az;
      const det = bx * cz - bz * cx;
      const u = ((x - ax) * cz - (z - az) * cx) / det;
      const v = (bx * (z - az) - bz * (x - ax)) / det;
      point.set(u, v, position.getY(i) + u * (position.getY(j) - position.getY(i)) +
        v * (position.getY(k) - position.getY(i)));
      return u >= -0.001 && v >= -0.001 && u + v <= 1.001;
    }
    if (!triangle(a, b, c)) triangle(d, c, b);
    return Math.max(0, point.z);
  }
  return { mesh, groundAt,
    counts: { mountains: 5, terrain_layers: 5, mountain_rows: rows,
      mountain_forest: forestInstances, mountain_style: 'cot-verdant-native-vista',
      mountain_relief_bake: Boolean(mesh.userData.horizonRing?.reliefBake),
      far_range: Boolean(mesh.getObjectByName('horizon-far-range')) },
    dispose() { disposeObject3DResources(mesh); } };
}

/** A low, textured apron joins the rectangular Jackal plane to CoT's 1024 m vista. */
export function createMountainApron(seed: number, width: number, height: number,
  surface: THREE.MeshStandardMaterial, groundSize: number,
  backdrop: Awaited<ReturnType<typeof prepareVerdantMountains>>) {
  const relief = createHorizonReliefField(seed ^ 0x712a, resolveHorizonRelief('rolling'));
  function apronHeight(x: number, z: number) {
    const outside = Math.max(Math.abs(x) - width / 2, Math.abs(z) - height / 2);
    const rim = Math.max(Math.abs(x), Math.abs(z));
    if (outside <= 0 || rim >= 512) return 0;
    const weight = THREE.MathUtils.smoothstep(outside, 0, 60) *
      (1 - THREE.MathUtils.smoothstep(rim, 370, 510));
    return Math.max(0, 1.5 + relief.low(x, z) * 0.12) * weight;
  }
  const geometry = new THREE.PlaneGeometry(1024, 1024, 192, 192);
  geometry.rotateX(-Math.PI / 2);
  const positions = geometry.getAttribute('position');
  const uv = geometry.getAttribute('uv');
  for (let i = 0; i < positions.count; i++) {
    const x = positions.getX(i), z = positions.getZ(i);
    positions.setY(i, apronHeight(x, z) + 0.012);
    uv.setXY(i, x / groundSize + 0.5, 0.5 - z / groundSize);
  }
  const index = geometry.getIndex()!;
  const triangles: number[] = [];
  for (let i = 0; i < index.count; i += 3) {
    const a = index.getX(i), b = index.getX(i + 1), c = index.getX(i + 2);
    const x = (positions.getX(a) + positions.getX(b) + positions.getX(c)) / 3;
    const z = (positions.getZ(a) + positions.getZ(b) + positions.getZ(c)) / 3;
    if (Math.abs(x) < width / 2 && Math.abs(z) < height / 2) continue;
    triangles.push(a, b, c);
  }
  geometry.setIndex(triangles);
  geometry.computeVertexNormals();
  const material = surface.clone();
  material.aoMap = null;
  material.onBeforeCompile = surface.onBeforeCompile;
  material.customProgramCacheKey = () => 'jackal-verdant-apron-pbr-v6';
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'textured low foothills / flat Jackal border to native Verdant vista';
  mesh.receiveShadow = true;
  return { mesh, groundAt(x: number, z: number) {
    if (Math.max(Math.abs(x), Math.abs(z)) < 512) return apronHeight(x, z);
    return backdrop.groundAt(x, z);
  } };
}
