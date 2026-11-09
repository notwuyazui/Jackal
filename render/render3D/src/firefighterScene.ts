/** The recording extends east; its original scenery keeps the same world origin. */
import type { Manifest } from './protocol.ts';

export const ORIGINAL_SIZE: [number, number] = [960, 640];
export const RELOCATED_BATHHOUSE = { x: 340, z: -50, yaw: -1.2 };
export const ORIGINAL_BURNING_HOUSE = { x: 177, z: -125, yaw: .8 };
// Crew #2 stages at (98, -44.5): 30 m to the fire centre.
export const BURNING_HOUSE = { x: 128, z: -44.5, yaw: .8 };
// Keep the burning ground clear of the nearby crew.
export const FIRE_GROUND_SIZE: [number, number] = [36, 40];
export const MOUNTAIN_GAP_SCALE = .25;
export const HOUSE_FIRE_SCALE = 10.4;
let houseGround = 0;
export function setHouseGround(height: number) { houseGround = height; }
export function houseGroundHeight() { return houseGround; }

export function sceneryManifest(manifest: Manifest): Manifest {
  if (!manifest.scene.scenery_size) return manifest;
  const [width, height] = manifest.scene.scenery_size;
  const columns = width / manifest.map.tile_size, rows = height / manifest.map.tile_size;
  return { ...manifest, map: { ...manifest.map, size: [width, height],
    tiles: Array.from({ length: rows }, (_, row) => row === 0 || row === rows - 1 ?
      'x'.repeat(columns) : `x${'o'.repeat(columns - 2)}x`) } };
}

/** Pull the forested slopes closer while leaving the old battlefield flat. */
export function mountainCoordinate(value: number, halfExtent: number, inverse = false, gapScale = MOUNTAIN_GAP_SCALE) {
  const excess = Math.max(0, Math.abs(value) - halfExtent);
  return Math.sign(value) * (Math.min(Math.abs(value), halfExtent) + excess * (inverse ? 1 / gapScale : gapScale));
}

export function moveBurningHouse<T extends { houses: Array<{ x: number; z: number; yaw: number; kind: string }>; boundaryHouses: Array<{ x: number; z: number; yaw: number; kind: string }> }>(layout: T): T {
  // The bathhouse behind the right-hand brush pile, not another village house.
  return { ...layout, houses: layout.houses.filter(house => !(house.kind === 'school' &&
    ((house.x === ORIGINAL_BURNING_HOUSE.x && house.z === ORIGINAL_BURNING_HOUSE.z) ||
     (house.x === BURNING_HOUSE.x && house.z === BURNING_HOUSE.z)))),
    boundaryHouses: layout.boundaryHouses.map(house =>
    house.kind === 'bathhouse' && house.x > 0 ? { ...house, ...RELOCATED_BATHHOUSE } : house) };
}
