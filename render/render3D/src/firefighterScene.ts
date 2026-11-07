/** The recording extends east; its original scenery keeps the same world origin. */
import type { Manifest } from './protocol.ts';

export const ORIGINAL_SIZE: [number, number] = [960, 640];
export const RELOCATED_BATHHOUSE = { x: 340, z: -50, yaw: -1.2 };
// The existing red-roof schoolhouse beside the rocks, ahead and left of the
// original movement area. Switching the fire must not move either building.
export const BURNING_HOUSE = { x: 177, z: -125, yaw: .8 };
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

/** Halve only the horizontal gap beyond the old battlefield's edges. */
export function mountainCoordinate(value: number, halfExtent: number, inverse = false) {
  const excess = Math.max(0, Math.abs(value) - halfExtent);
  return Math.sign(value) * (Math.min(Math.abs(value), halfExtent) + excess * (inverse ? 2 : .5));
}

export function moveBurningHouse<T extends { boundaryHouses: Array<{ x: number; z: number; yaw: number; kind: string }> }>(layout: T): T {
  // The bathhouse behind the right-hand brush pile, not another village house.
  return { ...layout, boundaryHouses: layout.boundaryHouses.map(house =>
    house.kind === 'bathhouse' && house.x > 0 ? { ...house, ...RELOCATED_BATHHOUSE } : house) };
}
