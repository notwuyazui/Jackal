import * as THREE from 'three';
import { createTallGrass, type TallGrass } from '@cot/world/tallGrass.ts';
import { resolveGroundReduxProfile } from '@cot/world/groundRedux.ts';
import grassColorUrl from '../../public/textures/terrain/Grass004_1K-JPG_Color.jpg?url';
import grassNormalUrl from '../../public/textures/terrain/Grass004_1K-JPG_NormalGL.jpg?url';
import grassRoughnessUrl from '../../public/textures/terrain/Grass004_1K-JPG_Roughness.jpg?url';
import grassAOUrl from '../../public/textures/terrain/Grass004_1K-JPG_AmbientOcclusion.jpg?url';
import { createExteriorFactory } from '../../src/exterior.ts';
import { createGroundDressingFactory } from '../../src/groundDressing.ts';
import { createRoadsideProps } from '../../src/roadsideProps.ts';
import { onRoad } from '../../src/roads.ts';
import { toWorld, worldSize } from '../../src/mapping.ts';
import { sceneryManifest } from '../../src/firefighterScene.ts';
import type { MapContext, PreparedMap } from '../registry.ts';
import type { Frame } from '../../src/protocol.ts';

export async function prepare({ renderer, loadingStage }: MapContext): Promise<PreparedMap> {
  loadingStage("正在加载地表材质…");
  const loader = new THREE.TextureLoader();
  const [grassColor, grassNormal, grassRoughness, grassAO] = await Promise.all([
    loader.loadAsync(grassColorUrl), loader.loadAsync(grassNormalUrl),
    loader.loadAsync(grassRoughnessUrl), loader.loadAsync(grassAOUrl),
  ]);
  loadingStage("正在加载山林与外景…");
  grassColor.colorSpace = THREE.SRGBColorSpace;
  const [exteriorFactory, groundDressingFactory] = await Promise.all([
    createExteriorFactory(loader), createGroundDressingFactory(loader, grassColor),
  ]);
  const grassTextures = [grassColor, grassNormal, grassRoughness, grassAO];
  for (const texture of grassTextures) {
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  }
  return {
    async build(frame: Frame) {
      const manifest = sceneryManifest(frame.manifest);
      await exteriorFactory.prepare(manifest.seed);
      const group = new THREE.Group();
      group.name = `Jackal scene: ${manifest.scene.id}`;
      let grass: TallGrass | null = null;
      let exterior: ReturnType<typeof exteriorFactory.build> | null = null;
      let groundDressing: ReturnType<typeof groundDressingFactory.build> | null = null;
      let roadside: ReturnType<typeof createRoadsideProps> | null = null;
      const [width, height] = manifest.map.size;
      const scale = manifest.meters_per_pixel;
      const groundSize = Math.max(width, height) * scale + 6600;
      for (const texture of grassTextures) texture.repeat.set(groundSize / 4, groundSize / 4);
      const ground = new THREE.Mesh(new THREE.PlaneGeometry(groundSize, groundSize),
        new THREE.MeshStandardMaterial({ color: '#b2bba0', map: grassColor, normalMap: grassNormal,
          normalScale: new THREE.Vector2(0.45, 0.45), roughnessMap: grassRoughness,
          aoMap: grassAO, aoMapIntensity: 0.55, roughness: 1,
          metalness: 0, envMapIntensity: 0.04 }));
      (ground.material as THREE.MeshStandardMaterial).onBeforeCompile = shader => {
        shader.fragmentShader = shader.fragmentShader.replace('#include <roughnessmap_fragment>',
          '#include <roughnessmap_fragment>\n roughnessFactor = max(0.94, roughnessFactor);');
      };
      (ground.material as THREE.MeshStandardMaterial).customProgramCacheKey = () => 'jackal-diffuse-soil-grass-v2';
      ground.name = 'flat playable map and continuous exterior / CoT Grass004 PBR';
      ground.rotation.x = -Math.PI / 2;
      ground.receiveShadow = true;
      group.add(ground);
      exterior = exteriorFactory.build(manifest, ground.material as THREE.MeshStandardMaterial, groundSize,
        frame.manifest.map.size);
      group.add(exterior.group);
      const noVegetation = (x: number, z: number) => {
        const column = Math.floor((x / scale + width / 2) / manifest.map.tile_size);
        const row = Math.floor((z / scale + height / 2) / manifest.map.tile_size);
        const tile = manifest.map.tiles[row]?.[column];
        return (tile !== 'o' && tile !== 'x') || onRoad(x, z, 0.55);
      };
      // Preserve the old scenery exclusions too: removing the brush piles must
      // not resample props or move any of the original roadside decorations.
      const starts = manifest.scene.scenery_size ? [[400, 220], [400, 320], [400, 420],
        [780, 190], [780, 320], [780, 450]] : frame.units.map(unit => unit.position);
      const unitStarts = starts.map(p => toWorld(p[0], p[1], ...manifest.map.size, scale));
      groundDressing = groundDressingFactory.build(manifest, exterior.groundAt, noVegetation, unitStarts);
      const followedCrew = frame.units.find(unit => unit.id === 2 && unit.team === 'PLAYER');
      if (manifest.scene.scenery_size && followedCrew) {
        const start = toWorld(...followedCrew.position, ...worldSize(frame.manifest), scale);
        // The crew approaches 52 engine pixels (13 m) east before spraying.
        // Include the actual leaf bounds and a 0.7 m body/arm clearance. Cull
        // after seeded placement so no other bush or decoration is resampled.
        const approach = new THREE.Box3(
          new THREE.Vector3(start.x - .7, -.1, start.z - .7),
          new THREE.Vector3(start.x + 13 + .7, 2.2, start.z + .7));
        const instance = new THREE.Matrix4(), worldMatrix = new THREE.Matrix4();
        const bounds = new THREE.Box3();
        groundDressing.group.updateMatrixWorld(true);
        for (const bush of groundDressing.group.children) {
          if (!(bush instanceof THREE.InstancedMesh) || bush.userData.vegetationKind !== 'bush') continue;
          bush.geometry.computeBoundingBox();
          let kept = 0;
          for (let i = 0; i < bush.count; i++) {
            bush.getMatrixAt(i, instance);
            worldMatrix.multiplyMatrices(bush.matrixWorld, instance);
            bounds.copy(bush.geometry.boundingBox!).applyMatrix4(worldMatrix);
            if (bounds.intersectsBox(approach)) continue;
            bush.setMatrixAt(kept++, instance);
          }
          const removed = bush.count - kept;
          groundDressing.counts.bushes -= removed;
          groundDressing.counts.cleared_approach_bushes += removed;
          bush.count = kept; bush.instanceMatrix.needsUpdate = true;
          bush.computeBoundingBox(); bush.computeBoundingSphere();
        }
      }
      group.add(groundDressing.group);
      roadside = createRoadsideProps(manifest, exterior.groundAt, unitStarts);
      group.add(roadside.group);
      grass = createTallGrass({ getHeightAt: () => 0,
        getNormalAt: () => ({ x: 0, y: 1, z: 0 }), _noVeg: noVegetation }, {
        seed: manifest.seed, biome: resolveGroundReduxProfile('verdant').grass,
        renderer, tier: 'desktop', qualityScale: () => 0.82, candidatesPerUpdate: 15000,
      });
      group.add(grass.group);
      return {
        group, grass, exterior, groundDressing, roadside,
        dispose() {
          grass?.dispose(); exterior?.dispose(); groundDressing?.dispose(); roadside?.dispose();
          ground.geometry.dispose(); ground.material.dispose();
          group.removeFromParent(); group.clear();
        },
      };
    },
    dispose() {
      exteriorFactory.dispose(); groundDressingFactory.dispose();
      for (const texture of grassTextures) texture.dispose();
    },
  };
}
