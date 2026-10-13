import type * as THREE from 'three';
import type { TallGrass } from '@cot/world/tallGrass.ts';
import type { Frame } from '../src/protocol.ts';

export interface InitialPlacement {
  id: number; model: string; team: string; x: number; y: number; z: number;
  hull_yaw: number; turret_yaw: number;
}
export interface SceneDefinition {
  id: string; name: string; builder: string; meters_per_pixel: number;
  unit_models: Record<string, string>;
  initial_units?: InitialPlacement[];
}
export interface MapContext {
  renderer: THREE.WebGLRenderer;
  loadingStage: (message: string) => void;
}
export interface MapScene {
  group: THREE.Group;
  grass: TallGrass | null;
  exterior: { counts: object; setTime(time: number): void } | null;
  groundDressing: { counts: object } | null;
  roadside: { counts: object; crushed: number; sync(frame: Frame): void } | null;
  dispose(): void;
}
export interface PreparedMap {
  build(frame: Frame): Promise<MapScene>;
  dispose(): void;
}

const catalog = import.meta.glob<SceneDefinition>('./catalog/*.json', { eager: true, import: 'default' });
const builders = import.meta.glob<{ prepare(context: MapContext): Promise<PreparedMap> }>('./scenes/*.ts');

export async function prepareMap(scene: SceneDefinition, context: MapContext): Promise<PreparedMap> {
  const definition = Object.values(catalog).find(candidate => candidate.id === scene.id);
  if (!definition || definition.builder !== scene.builder || definition.meters_per_pixel !== scene.meters_per_pixel
      || JSON.stringify(definition.unit_models) !== JSON.stringify(scene.unit_models)) {
    throw new Error(`3D 地图注册信息与 Python 不一致：${scene.id}`);
  }
  const load = builders[`./scenes/${scene.builder}.ts`];
  if (!load) throw new Error(`没有找到地图构建器：${scene.builder}`);
  return (await load()).prepare(context);
}
