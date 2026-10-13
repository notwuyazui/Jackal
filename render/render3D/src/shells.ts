import * as THREE from 'three';
import { createFx } from '@cot/fx/effects.ts';
import { registerFxClock, registerPopTrail } from '@cot/fx/clock.ts';
import { LATE_FX_LAYER } from '@cot/fx/layers.ts';
import { toWorld } from './mapping.ts';
import type { Frame } from './protocol.ts';

interface Shell {
  id: number; pos: THREE.Vector3; prevPos: THREE.Vector3; vel: THREE.Vector3;
  distM: number; isPlayer: boolean; spec: { tracer: 'HE' };
}

/** Original CoT HE tracer geometry/shader and particle atlases, without ballistics. */
export async function createShellPresentation(scene: THREE.Scene, camera: THREE.Camera) {
  const fx = createFx({ scene, camera, anisotropy: 8 }, { getHeightAt: () => 0 }, { seed: 42 });
  if (!await fx.preloadTextures()) throw new Error('CoT 原版炮弹特效贴图未加载');
  fx.warmTextures();
  scene.add(fx.group);
  // CoT composites these effects in a late pass. This direct renderer sees that layer too.
  camera.layers.enable(LATE_FX_LAYER);
  let tracerGeometry: THREE.InstancedBufferGeometry | undefined;
  fx.group.traverse(object => {
    if (object instanceof THREE.Mesh && object.geometry.getAttribute('aA') && object.geometry.getAttribute('aB')) {
      tracerGeometry = object.geometry;
    }
  });
  if (!tracerGeometry) throw new Error('CoT native tracer geometry is missing');
  const shells = new Map<number, Shell>();
  const live: Shell[] = [];
  const normal = new THREE.Vector3();
  const particleMeshes = fx.group.getObjectByName('fx-particles')!.children as
    THREE.Mesh<THREE.InstancedBufferGeometry, THREE.ShaderMaterial>[];
  const softParticles = fx.group.getObjectByName('fx-particles')!.userData.softParticles;
  function activePuffs(index: number) {
    const mesh = particleMeshes[index];
    const births = mesh.geometry.getAttribute('aPB');
    const lives = mesh.geometry.getAttribute('aVL');
    const time = mesh.material.uniforms.uTime.value as number;
    let active = 0;
    for (let i = 0; i < mesh.geometry.instanceCount; i++) {
      const age = time - births.getW(i), life = lives.getW(i);
      if (life > 0 && age >= 0 && age <= life) active++;
    }
    return active;
  }

  return {
    bindSceneDepth(depth: THREE.DepthTexture, width: number, height: number) {
      softParticles.uSceneDepth.value = depth;
      softParticles.uSoftViewport.value.set(width, height);
      softParticles.uCameraNear.value = (camera as THREE.PerspectiveCamera).near;
      softParticles.uCameraFar.value = (camera as THREE.PerspectiveCamera).far;
    },
    pinTime(time: number) { fx.setFrozen(true, time); },
    reset(seed: number) { fx.setFrozen(true, 0); fx.resetAll(); fx.resetSeed(seed); shells.clear(); live.length = 0; },
    muzzleFlash(position: THREE.Vector3, direction: THREE.Vector3) {
      // A stronger visual recipe scales CoT's original flash, ember fan,
      // smoke ring and ground blast; Jackal's shell stays exactly NormalShell.
      fx.muzzleFlash(position, direction, 200);
    },
    destruction(position: THREE.Vector3) {
      // Native CoT fireball, debris, blast light and smoke; the vehicle's
      // burnt material/turret animation is driven separately by its visual.
      fx.destruction(position, null, 'shot');
    },
    impact(id: number | undefined, position: THREE.Vector3) {
      const shell = id === undefined ? undefined : shells.get(id);
      if (shell) normal.copy(shell.vel).negate().normalize();
      else normal.set(0, 0, -1);
      // This selects a fireball recipe only; no CoT penetration/ricochet simulation runs.
      fx.impact('he_pen', position, normal, 120);
    },
    sync(frame: Frame, planeHeight: number) {
      live.length = 0;
      const seen = new Set<number>();
      const scale = frame.manifest.meters_per_pixel;
      for (const bullet of frame.bullets) {
        const point = toWorld(...bullet.position, ...frame.manifest.map.size, scale);
        let shell = shells.get(bullet.id);
        if (!shell) {
          shell = { id: bullet.id, pos: new THREE.Vector3(point.x, planeHeight, point.z),
            prevPos: new THREE.Vector3(point.x, planeHeight, point.z), vel: new THREE.Vector3(),
            distM: Math.hypot(...bullet.velocity) * scale * frame.manifest.dt,
            isPlayer: bullet.source_id === frame.manifest.follow_id, spec: { tracer: 'HE' } };
          shells.set(bullet.id, shell);
        }
        shell.prevPos.copy(shell.pos);
        shell.pos.set(point.x, planeHeight, point.z);
        shell.distM += shell.pos.distanceTo(shell.prevPos);
        shell.vel.set(bullet.velocity[0] * scale, 0, bullet.velocity[1] * scale);
        seen.add(bullet.id);
        live.push(shell);
      }
      for (const id of shells.keys()) if (!seen.has(id)) shells.delete(id);
    },
    render() { fx.update(0, live, camera); },
    effectState() { return { flash_particles: activePuffs(5), smoke_particles: activePuffs(3),
      soft_depth_bound: softParticles.uSceneDepth.value instanceof THREE.DepthTexture }; },
    receipts(manifest: Frame['manifest']) {
      const heads = tracerGeometry!.getAttribute('aB');
      const tails = tracerGeometry!.getAttribute('aA');
      if (tracerGeometry!.instanceCount < live.length) throw new Error('CoT omitted a live shell');
      return live.map((shell, index) => ({ id: shell.id,
        x: heads.getX(index) / manifest.meters_per_pixel + manifest.map.size[0] / 2,
        y: heads.getZ(index) / manifest.meters_per_pixel + manifest.map.size[1] / 2,
        height_m: heads.getY(index), tail_height_m: tails.getY(index),
        vertical_speed_mps: shell.vel.y, tracer: shell.spec.tracer }));
    },
    dispose() {
      fx.resetAll(); fx.group.removeFromParent();
      const geometries = new Set<THREE.BufferGeometry>();
      const materials = new Set<THREE.Material>();
      const textures = new Set<THREE.Texture>();
      fx.group.traverse(object => {
        if (!(object instanceof THREE.Mesh)) return;
        geometries.add(object.geometry);
        for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
          materials.add(material);
          for (const value of Object.values(material)) if (value instanceof THREE.Texture) textures.add(value);
          if (material instanceof THREE.ShaderMaterial) {
            for (const uniform of Object.values(material.uniforms)) {
              if (uniform.value instanceof THREE.Texture) textures.add(uniform.value);
            }
          }
        }
      });
      for (const geometry of geometries) geometry.dispose();
      for (const material of materials) material.dispose();
      for (const texture of textures) texture.dispose();
      registerFxClock(() => 0); registerPopTrail(() => {});
      shells.clear(); live.length = 0;
    },
  };
}
