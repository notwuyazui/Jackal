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
import { toWorld } from '../../src/mapping.ts';
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
      const manifest = frame.manifest;
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
      exterior = exteriorFactory.build(manifest, ground.material as THREE.MeshStandardMaterial, groundSize);
      group.add(exterior.group);
      const noVegetation = (x: number, z: number) => {
        const column = Math.floor((x / scale + width / 2) / manifest.map.tile_size);
        const row = Math.floor((z / scale + height / 2) / manifest.map.tile_size);
        const tile = manifest.map.tiles[row]?.[column];
        return (tile !== 'o' && tile !== 'x') || onRoad(x, z, 0.55);
      };
      const unitStarts = frame.units.map(unit => toWorld(...unit.position, ...manifest.map.size, scale));
      groundDressing = groundDressingFactory.build(manifest, exterior.groundAt, noVegetation, unitStarts);
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
