/** Surface coordinates of the existing CoT schoolhouse, in building metres. */
import { BURNING_HOUSE } from './firefighterScene.ts';

export function houseSurfaceWorld(x: number, z: number) {
  const c = Math.cos(BURNING_HOUSE.yaw), s = Math.sin(BURNING_HOUSE.yaw);
  return { x: BURNING_HOUSE.x + c * x + s * z, z: BURNING_HOUSE.z - s * x + c * z };
}
