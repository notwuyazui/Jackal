/** The original Verdant Fields vista, including its relief atlas and far range. */
import * as THREE from 'three';
import verdant from '@cot/world/maps/verdant.ts';
import { buildHorizonRingSteps, HORIZON_SEGMENTS } from '@cot/world/maps/horizon.ts';
import { disposeObject3DResources } from '@cot/engine/resourceLifetime.ts';
import { createHorizonReliefField, resolveHorizonRelief } from '@cot/world/horizonRelief.ts';
import { onRoad } from './roads.ts';
import { mountainCoordinate } from './firefighterScene.ts';

export async function prepareVerdantMountains(seed: number) {
  const steps = buildHorizonRingSteps(null, {
    ...verdant, horizon: { ...verdant.horizon, farRange: true,
      treeline: 1.4, snowline: 2, baseHex: 0x4b7139, forestHex: 0x2e5027,
      bareRock: 0, outcrops: 0,
      forestInstances: 32000, forestRadius: 2600, forestDensity: 2.8, forestSummits: true },
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
  // Keep living meadow and canopy textures across every slope and summit;
  // the recording's mountains have no exposed rock or scree caps.
  for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
    const vista = material.userData.horizonVista as { uniforms?: Record<string, THREE.IUniform> } | undefined;
    for (const name of ['uVRockAmp', 'uVPeakRock', 'uVScreeAmp', 'uVBareRock', 'uVOutcrop']) {
      if (vista?.uniforms?.[name]) vista.uniforms[name].value = 0;
    }
  }
  // Keep the original relief sampler for unchanged scenery placements.
  const position = mesh.geometry.getAttribute('position').clone();
  const stride = HORIZON_SEGMENTS + 1; // CoT duplicates the closing vertex of every row.
  const rows = position.count / stride;
  const columns = HORIZON_SEGMENTS;
  const point = new THREE.Vector3();
  let forestInstances = 0;
  mesh.traverse(object => {
    if (object instanceof THREE.InstancedMesh) forestInstances += object.count;
  });
  // Sample the actual rendered triangles to seat the bridge's hillside trees.
  function originalGroundAt(x: number, z: number) {
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
  // Move the complete mountain vista, including its own forest/decor, while
  // keeping tree sizes and mountain heights. The old battlefield stays fixed.
  mesh.updateMatrixWorld(true);
  const matrix = new THREE.Matrix4(), inverse = new THREE.Matrix4();
  const translation = new THREE.Vector3();
  const movedGeometry = new Set<THREE.BufferGeometry>();
  mesh.traverse(object => {
    if (!(object instanceof THREE.Mesh)) return;
    inverse.copy(object.matrixWorld).invert();
    if (object instanceof THREE.InstancedMesh) {
      for (let i = 0; i < object.count; i++) {
        object.getMatrixAt(i, matrix);
        translation.setFromMatrixPosition(matrix).applyMatrix4(object.matrixWorld);
        translation.set(mountainCoordinate(translation.x, 120), translation.y,
          mountainCoordinate(translation.z, 80)).applyMatrix4(inverse);
        matrix.setPosition(translation); object.setMatrixAt(i, matrix);
      }
      object.instanceMatrix.needsUpdate = true; object.computeBoundingSphere();
    } else if (!movedGeometry.has(object.geometry)) {
      movedGeometry.add(object.geometry);
      const vertices = object.geometry.getAttribute('position');
      for (let i = 0; i < vertices.count; i++) {
        translation.fromBufferAttribute(vertices, i).applyMatrix4(object.matrixWorld);
        translation.set(mountainCoordinate(translation.x, 120), translation.y,
          mountainCoordinate(translation.z, 80)).applyMatrix4(inverse);
        vertices.setXYZ(i, translation.x, translation.y, translation.z);
      }
      vertices.needsUpdate = true;
      object.geometry.computeVertexNormals(); object.geometry.computeBoundingSphere();
    }
  });
  return { mesh, originalGroundAt, groundAt(x: number, z: number) {
    return originalGroundAt(mountainCoordinate(x, 120, true), mountainCoordinate(z, 80, true));
  },
    counts: { mountains: 5, terrain_layers: 5, mountain_rows: rows,
      mountain_forest: forestInstances, mountain_style: 'cot-verdant-native-vista',
      summit_vegetation: true, mountain_ground_cover: 'continuous-grass-and-forest',
      mountain_relief_bake: Boolean(mesh.userData.horizonRing?.reliefBake), mountain_gap_scale: .5,
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
    positions.setXYZ(i, mountainCoordinate(x, width / 2), apronHeight(x, z) + 0.012,
      mountainCoordinate(z, height / 2));
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
  function originalGroundAt(x: number, z: number) {
    if (Math.max(Math.abs(x), Math.abs(z)) < 512) return apronHeight(x, z);
    return backdrop.originalGroundAt(x, z);
  }
  return { mesh, originalGroundAt, groundAt(x: number, z: number) {
    return originalGroundAt(mountainCoordinate(x, width / 2, true), mountainCoordinate(z, height / 2, true));
  } };
}
