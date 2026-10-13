/** CoT civilian cars and fuel drums, driven only by Jackal's read-only poses. */
import * as THREE from 'three';
import { DESTRUCTIBLE_TYPES } from '@cot/world/maps/inhabitKit.ts';
import { routes, center } from './roads.ts';
import type { Frame, Manifest } from './protocol.ts';

type Kind = 'jeep' | 'sedan' | 'wagon' | 'drumred';
type Record = { x: number; z: number; kind: Kind; intact: THREE.Mesh;
  broken: THREE.Mesh; crushed: boolean };

function seeded(seed: number) {
  let value = seed | 0;
  return () => {
    value = value + 0x6d2b79f5 | 0;
    let t = Math.imul(value ^ value >>> 15, 1 | value);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

export function createRoadsideProps(manifest: Manifest,
  groundAt: (x: number, z: number) => number,
  unitStarts: Array<{ x: number; z: number }>) {
  const group = new THREE.Group();
  group.name = 'CoT original crushable cars and red fuel drums / visual only';
  const random = seeded(manifest.seed ^ 0x71cd);
  const kinds: Kind[] = ['jeep', 'sedan', 'wagon', 'drumred'];
  const geometry = new Map<Kind, { intact: THREE.BufferGeometry; broken: THREE.BufferGeometry }>();
  for (const kind of kinds) {
    const spec = DESTRUCTIBLE_TYPES[kind];
    geometry.set(kind, { intact: spec.build(random), broken: spec.broken!(random) });
  }
  // The original CoT geometry carries its own per-part paint, glass, rubber
  // and hazard-band colors in vertex colors. These material settings match
  // CoT's vehicle/baked prop finish without introducing a collision object.
  const vehicleMaterial = new THREE.MeshStandardMaterial({ vertexColors: true,
    roughness: 0.84, metalness: 0.045 });
  const drumMaterial = new THREE.MeshStandardMaterial({ vertexColors: true,
    roughness: 0.88, metalness: 0 });
  const records: Record[] = [];
  const [pixelWidth, pixelHeight] = manifest.map.size;
  const scale = manifest.meters_per_pixel;
  const width = pixelWidth * scale, height = pixelHeight * scale;

  routes.forEach((route, routeIndex) => {
    const edge = (route.across ? width : height) / 2;
    // Two western shoulder pieces lie along the deterministic opening drive,
    // so a tank visibly flattens both a car and a drum in the demo.
    const stations = routeIndex === 0
      ? [-edge - 42, -44, -36, edge * 0.58, edge + 42]
      : [-edge - 42, -edge * 0.58, 0, edge * 0.58, edge + 42];
    stations.forEach((along, index) => {
      const car = index === 1 || index === 3;
      const kind: Kind = car ? (['jeep', 'sedan', 'wagon'] as const)[
        (routeIndex + (index === 3 ? 1 : 0)) % 3] : 'drumred';
      const side = routeIndex === 0 && index === 1 ? 1 :
        (routeIndex + index) % 2 === 0 ? 1 : -1;
      const offset = side * (route.width / 2 + (car ? 2.3 : 1.25));
      const cross = center(route, along) + offset;
      const x = route.across ? along : cross;
      const z = route.across ? cross : along;
      if (unitStarts.some(start => Math.hypot(start.x - x, start.z - z) < 9)) return;
      const source = geometry.get(kind)!;
      const material = kind === 'drumred' ? drumMaterial : vehicleMaterial;
      const intact = new THREE.Mesh(source.intact, material);
      const broken = new THREE.Mesh(source.broken, material);
      const yaw = (route.across ? Math.PI / 2 : 0) + (random() - 0.5) * 0.23;
      const y = groundAt(x, z) + 0.025;
      for (const mesh of [intact, broken]) {
        mesh.position.set(x, y, z);
        mesh.rotation.y = yaw;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        group.add(mesh);
      }
      intact.name = `CoT original ${kind} / crushable roadside prop`;
      broken.name = `CoT original ${kind} / crushed debris`;
      broken.visible = false;
      records.push({ x, z, kind, intact, broken, crushed: false });
    });
  });

  const counts = { cars: records.filter(record => record.kind !== 'drumred').length,
    fuel_drums: records.filter(record => record.kind === 'drumred').length };
  let crushed = 0;
  return {
    group, counts,
    get crushed() { return crushed; },
    sync(frame: Frame) {
      for (const record of records) {
        if (record.crushed) continue;
        const x = record.x / scale + pixelWidth / 2;
        const y = record.z / scale + pixelHeight / 2;
        const radius = DESTRUCTIBLE_TYPES[record.kind].r / scale;
        const overrun = frame.units.some(unit => {
          if (!unit.alive) return false;
          const [left, top, boxWidth, boxHeight] = unit.collision_box;
          return x >= left - radius && x <= left + boxWidth + radius &&
            y >= top - radius && y <= top + boxHeight + radius;
        });
        if (!overrun) continue;
        record.crushed = true;
        record.intact.visible = false;
        record.broken.visible = true;
        crushed++;
      }
    },
    dispose() {
      group.removeFromParent();
      for (const item of geometry.values()) {
        item.intact.dispose();
        item.broken.dispose();
      }
      vehicleMaterial.dispose();
      drumMaterial.dispose();
    },
  };
}
