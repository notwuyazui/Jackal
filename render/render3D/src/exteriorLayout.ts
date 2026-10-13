/** Reproducible scenery placement in metres. None of these records is a combat entity. */
import { onRoad, routes, center } from './roads.ts';
export interface ExteriorBounds { width: number; height: number }
export interface ExteriorTree { x: number; z: number; species: 'pine' | 'oak' | 'birch'; size: number; yaw: number }
export interface ExteriorRock { x: number; z: number; size: number; variant: number; yaw: number }
export interface ExteriorHouse { x: number; z: number; kind: 'ranger' | 'school' | 'tavern' | 'bathhouse' | 'firestation' | 'foundry'; yaw: number }
export interface ExteriorPole { x: number; z: number; yaw: number; line: number }
export interface ExteriorHay { x: number; z: number; kind: 'haystack' | 'bale' | 'stook'; size: number; yaw: number }
export interface BoundaryFence { x: number; z: number; yaw: number; kind: 'fenceplank' | 'fencepicket' }

export function outsideMap(x: number, z: number, bounds: ExteriorBounds, margin = 0) {
  return Math.abs(x) >= bounds.width / 2 + margin || Math.abs(z) >= bounds.height / 2 + margin;
}

function seeded(seed: number) {
  let value = seed | 0;
  return () => {
    value = value + 0x6d2b79f5 | 0;
    let t = Math.imul(value ^ value >>> 15, 1 | value);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

export function planExterior(seed: number, bounds: ExteriorBounds) {
  const random = seeded(seed ^ 0x519a);
  const outerPoint = (distance: number) => {
    const angle = random() * Math.PI * 2;
    const dx = Math.cos(angle), dz = Math.sin(angle);
    const edge = Math.min(bounds.width / 2 / Math.max(1e-5, Math.abs(dx)),
      bounds.height / 2 / Math.max(1e-5, Math.abs(dz)));
    return { x: dx * (edge + distance), z: dz * (edge + distance) };
  };
  const w = bounds.width / 2, h = bounds.height / 2;
  const houses: ExteriorHouse[] = [
    { x: -w - 52, z: -h - 48, kind: 'ranger', yaw: 0.25 },
    { x: -w - 91, z: -h - 43, kind: 'tavern', yaw: -0.4 },
    { x: -w - 58, z: h + 48, kind: 'bathhouse', yaw: 0.2 },
    { x: -w - 96, z: h + 67, kind: 'school', yaw: -0.7 },
    { x: w + 57, z: -h - 45, kind: 'school', yaw: 0.8 },
    { x: w + 94, z: -h - 70, kind: 'firestation', yaw: 0.6 },
    { x: w + 63, z: h + 48, kind: 'ranger', yaw: -0.4 },
    { x: w + 100, z: h + 75, kind: 'foundry', yaw: -0.7 },
    { x: -62, z: -h - 64, kind: 'tavern', yaw: 0.25 },
    { x: 3, z: -h - 85, kind: 'bathhouse', yaw: -0.15 },
    { x: 40, z: -h - 68, kind: 'ranger', yaw: -0.4 },
    { x: 32, z: h + 93, kind: 'firestation', yaw: 0.2 },
  ];
  const poles: ExteriorPole[] = [];
  for (let i = 0; i < 13; i++) {
    poles.push({ x: -w - 94 + i * (2 * w + 188) / 12,
      z: -h - 23 - Math.sin(i * 0.57) * 2.5, yaw: Math.PI / 2, line: 0 });
  }
  for (let i = 0; i < 9; i++) {
    poles.push({ x: w + 24 + Math.sin(i * 0.62) * 1.8,
      z: -h - 86 + i * (2 * h + 172) / 8, yaw: 0, line: 1 });
  }
  // Each dirt track continues into the exterior. Two orderly rows of native
  // CoT telephone poles flank each exterior branch, with no wires crossing
  // the playable map between opposite branches.
  routes.forEach((route, routeIndex) => {
    const edge = route.across ? w : h;
    for (const [sideIndex, side] of [-1, 1].entries()) {
      for (const [branchIndex, branch] of [-1, 1].entries()) {
        const line = 2 + routeIndex * 4 + sideIndex * 2 + branchIndex;
        for (let station = 0; station < 6; station++) {
          const along = branch * (edge + 22 + station * 27);
          const cross = center(route, along) + side * (route.width / 2 + 6.5);
          const x = route.across ? along : cross;
          const z = route.across ? cross : along;
          if (houses.some(house => Math.hypot(x - house.x, z - house.z) < 12)) continue;
          poles.push({ x, z, yaw: route.across ? Math.PI / 2 : 0, line });
        }
      }
    }
  });
  const clearHouse = (x: number, z: number, radius: number) =>
    houses.every(house => Math.hypot(x - house.x, z - house.z) > radius);
  const trees: ExteriorTree[] = [];
  for (let attempt = 0; trees.length < 1850 && attempt < 7000; attempt++) {
    const p = outerPoint(28 + Math.pow(random(), 1.8) * 400);
    const villageApproach = Math.abs(p.x) < 77 && p.z < -h - 10 && p.z > -h - 132;
    if (villageApproach || onRoad(p.x, p.z, 5) || !clearHouse(p.x, p.z, 28) ||
      poles.some(pole => Math.hypot(p.x - pole.x, p.z - pole.z) < 7)) continue;
    const choice = random();
    trees.push({ ...p, species: choice < 0.46 ? 'pine' : choice < 0.77 ? 'oak' : 'birch',
      size: 1.15 + random() * 1.08, yaw: random() * Math.PI * 2 });
  }
  // The original tree belt ends before the mountain ring; populate its
  // middle and upper slopes so the ridge is not an empty monochrome wall.
  for (let attempt = 0, placed = 0; placed < 1600 && attempt < 9000; attempt++) {
    const p = outerPoint(425 + random() * 445);
    if (onRoad(p.x, p.z, 5)) continue;
    const choice = random();
    trees.push({ ...p, species: choice < 0.62 ? 'pine' : choice < 0.86 ? 'oak' : 'birch',
      size: 0.82 + random() * 0.75, yaw: random() * Math.PI * 2 });
    placed++;
  }
  const rocks: ExteriorRock[] = [];
  for (let attempt = 0; rocks.length < 210 && attempt < 1500; attempt++) {
    const p = outerPoint(26 + random() * 290);
    if (onRoad(p.x, p.z, 4)) continue;
    rocks.push({ ...p, size: 1.4 + random() * 2.6,
      variant: Math.floor(random() * 3), yaw: random() * Math.PI * 2 });
  }
  const hay: ExteriorHay[] = [];
  for (let i = 0; i < 10; i++) {
    const x = -105 + i * 23, z = -h - 18 - (i % 3) * 7;
    if (onRoad(x, z, 3)) continue;
    hay.push({ x, z,
      kind: i % 3 === 0 ? 'haystack' : i % 3 === 1 ? 'bale' : 'stook',
      size: 0.9 + (i % 4) * 0.11, yaw: i * 0.7 });
  }
  for (let attempt = 0; hay.length < 46 && attempt < 800; attempt++) {
    const p = outerPoint(28 + random() * 154);
    if (onRoad(p.x, p.z, 4) || !clearHouse(p.x, p.z, 11) ||
      poles.some(pole => Math.hypot(p.x - pole.x, p.z - pole.z) < 4)) continue;
    const choice = random();
    hay.push({ ...p, kind: choice < 0.44 ? 'haystack' : choice < 0.81 ? 'bale' : 'stook',
      size: 0.88 + random() * 0.42, yaw: random() * Math.PI * 2 });
  }
  // Jackal's outer tile is impassable. These sparse CoT objects mark it
  // without drawing a continuous visual wall or adding new collision.
  const boundaryHouses: ExteriorHouse[] = [
    { x: -w + 8, z: 20, kind: 'ranger', yaw: 1.35 },
    { x: w - 8, z: 57, kind: 'bathhouse', yaw: -1.2 },
  ];
  const fenceRuns = [
    { side: 'north', from: -w + 23, to: -w + 48 },
    { side: 'north', from: 8, to: 30 },
    { side: 'north', from: 44, to: 60 },
    { side: 'south', from: -w + 25, to: -w + 49 },
    { side: 'south', from: -9, to: 17 },
    { side: 'south', from: 34, to: 54 },
    { side: 'west', from: -h + 21, to: -h + 36 },
    { side: 'west', from: -8, to: 7 },
    { side: 'west', from: 36, to: 52 },
    { side: 'east', from: -h + 23, to: -h + 38 },
    { side: 'east', from: -6, to: 10 },
    { side: 'east', from: 30, to: 43 },
  ] as const;
  const boundaryFences: BoundaryFence[] = [];
  for (const [index, run] of fenceRuns.entries()) {
    for (let along = run.from + 1.2; along < run.to - 0.6; along += 2.4) {
      const horizontal = run.side === 'north' || run.side === 'south';
      const x = horizontal ? along : run.side === 'west' ? -w + 8 : w - 8;
      const z = horizontal ? run.side === 'north' ? -h + 8 : h - 8 : along;
      if (onRoad(x, z, 1.2) || boundaryHouses.some(house => Math.hypot(x - house.x, z - house.z) < 10)) continue;
      boundaryFences.push({ x, z, yaw: horizontal ? Math.PI / 2 : 0,
        kind: index % 3 === 0 ? 'fencepicket' : 'fenceplank' });
    }
  }
  const boundaryRocks: ExteriorRock[] = [];
  for (let attempt = 0; boundaryRocks.length < 26 && attempt < 900; attempt++) {
    const side = Math.floor(random() * 4);
    const x = side < 2 ? (side === 0 ? -w + 7 : w - 7) : (random() * 2 - 1) * (w - 15);
    const z = side >= 2 ? (side === 2 ? -h + 7 : h - 7) : (random() * 2 - 1) * (h - 15);
    if (onRoad(x, z, 3.5) ||
      boundaryHouses.some(house => Math.hypot(x - house.x, z - house.z) < 11) ||
      boundaryFences.some(fence => Math.hypot(x - fence.x, z - fence.z) < 4) ||
      boundaryRocks.some(rock => Math.hypot(x - rock.x, z - rock.z) < 6)) continue;
    boundaryRocks.push({ x, z, size: 1.3 + random() * 1.9,
      variant: Math.floor(random() * 3), yaw: random() * Math.PI * 2 });
  }
  return { trees, rocks, houses, poles, hay, boundaryFences, boundaryRocks, boundaryHouses };
}
