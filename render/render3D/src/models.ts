import type { createM1A1 } from './m1a1.ts';
import type { Manifest, Parameters } from './protocol.ts';
import type { SceneDefinition } from '../maps/registry.ts';

export { TEAM_CAMO } from './m1a1.ts';
export type Vehicle = ReturnType<typeof createM1A1>;
interface ModelAdapter {
  prepare(): Promise<unknown>;
  create(parameters: Parameters, team: string, manifest: Manifest): Vehicle;
}
const catalog = import.meta.glob<{ id: string; adapter: string }>(
  '../models/catalog/*.json', { eager: true, import: 'default' });
const loaders = import.meta.glob<ModelAdapter>('../models/adapters/*.ts');
const models = new Map<string, ModelAdapter>();
function resolve(id: string) {
  const model = models.get(id);
  if (!model) throw new Error(`3D 车型尚未准备：${id}`);
  return model;
}
export async function prepareVehicles(scene: SceneDefinition) {
  await Promise.all([...new Set(Object.values(scene.unit_models))].map(async id => {
    if (!models.has(id)) {
      const definition = Object.values(catalog).find(candidate => candidate.id === id);
      const load = definition && loaders[`../models/adapters/${definition.adapter}.ts`];
      if (!load) throw new Error(`未实现的 3D 车型：${id}`);
      models.set(id, await load());
    }
    await resolve(id).prepare();
  }));
}
export function createVehicle(parameters: Parameters, team: string, manifest: Manifest): Vehicle {
  return resolve(manifest.scene.unit_models[parameters.unit_type]).create(parameters, team, manifest);
}
