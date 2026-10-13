import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { createGarageTreeKit } from '@cot/world/vegetation.ts';
import { buildHorizonBoulder } from '@cot/world/horizonRockfield.ts';
import { makeFireStation, makeFoundryOffice, makeRangerLodge, makeSchoolhouse,
  makeTavern, makeTimberBathhouse } from '@cot/world/maps/structureKit.ts';
import { makeTelephonePoleDistanceGeometry } from '@cot/world/propGeometry.ts';
import { DESTRUCTIBLE_TYPES, FENCE_SEG } from '@cot/world/maps/inhabitKit.ts';
import type { GeometryBuckets } from '@cot/world/maps/exteriorDetailKit.ts';
import roofUrl from '../public/textures/buildings/RoofingTiles012A_1K-JPG_Color.jpg?url';
import woodUrl from '../public/textures/buildings/Planks023A_1K-JPG_Color.jpg?url';
import stoneUrl from '../public/textures/buildings/Bricks097_1K-JPG_Color.jpg?url';
import plasterUrl from '../public/textures/buildings/Plaster007_1K-JPG_Color.jpg?url';
import strawUrl from '../public/textures/terrain/withered_grass_diff_1k.jpg?url';
import { prepareVerdantMountains, createMountainApron } from './mountains.ts';
import { planExterior } from './exteriorLayout.ts';
import type { Manifest } from './protocol.ts';

function airBarrier(width: number, height: number, tile: number) {
  const group = new THREE.Group();
  group.name = 'Jackal boundary / transparent air barrier';
  const material = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 } }, transparent: true, depthWrite: false,
    side: THREE.DoubleSide, toneMapped: false,
    vertexShader: `varying vec2 vUv; void main() { vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
    fragmentShader: `varying vec2 vUv; uniform float uTime;
      void main() {
        float top = vUv.y * vUv.y;
        float shimmer = 0.5 + 0.5 * sin(vUv.x*31.0 + vUv.y*39.0 + uTime*1.7);
        float alpha = 0.011 + top*(0.047 + shimmer*0.019);
        gl_FragColor = vec4(0.38,0.68,0.81,alpha);
      }`,
  });
  const innerWidth = width - 2 * tile, innerHeight = height - 2 * tile;
  const panels: [number, number, number, number, number][] = [
    [0, -innerHeight / 2, innerWidth, 0, 0],
    [0, innerHeight / 2, innerWidth, 0, 0],
    [-innerWidth / 2, 0, innerHeight, Math.PI / 2, 0],
    [innerWidth / 2, 0, innerHeight, Math.PI / 2, 0],
  ];
  const geometries: THREE.BufferGeometry[] = [];
  for (const [x, z, length, yaw] of panels) {
    const geometry = new THREE.PlaneGeometry(length, 3.2);
    geometries.push(geometry);
    const panel = new THREE.Mesh(geometry, material);
    panel.position.set(x, 1.6, z);
    panel.rotation.y = yaw;
    panel.renderOrder = 10;
    group.add(panel);
  }
  return { group, material, geometries, setTime(time: number) { material.uniforms.uTime.value = time; } };
}

function structureBuckets(): GeometryBuckets {
  return {
    plaster: [], plaster2: [], plaster3: [], stone: [], roof: [], wood: [],
    dark: [], glass: [], curtain: [], straw: [], baked: [], steel: [], structureMetal: [],
  };
}

export async function createExteriorFactory(loader: THREE.TextureLoader) {
  const [roof, wood, stone, plaster, straw] = await Promise.all([
    loader.loadAsync(roofUrl), loader.loadAsync(woodUrl),
    loader.loadAsync(stoneUrl), loader.loadAsync(plasterUrl), loader.loadAsync(strawUrl),
  ]);
  const textures = { roof, wood, stone, plaster, straw };
  let backdrop: Awaited<ReturnType<typeof prepareVerdantMountains>> | null = null;
  let backdropSeed: number | null = null;
  for (const texture of Object.values(textures)) {
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.anisotropy = 8;
  }

  return {
    async prepare(seed: number) {
      if (backdrop && backdropSeed === seed) return;
      backdrop?.dispose();
      backdrop = await prepareVerdantMountains(seed);
      backdropSeed = seed;
    },
    build(manifest: Manifest, groundMaterial: THREE.MeshStandardMaterial, groundSize: number) {
      const [mapWidth, mapHeight] = manifest.map.size;
      const width = mapWidth * manifest.meters_per_pixel;
      const height = mapHeight * manifest.meters_per_pixel;
      const tile = manifest.map.tile_size * manifest.meters_per_pixel;
      const layout = planExterior(manifest.seed, { width, height });
      const group = new THREE.Group();
      group.name = 'CoT scenery outside Jackal combat bounds / visual only';
      const ownedGeometry = new Set<THREE.BufferGeometry>();
      const ownedMaterial = new Set<THREE.Material>();
      const treeKits: ReturnType<typeof createGarageTreeKit>[] = [];
      const modelMatrix = new THREE.Matrix4();
      const rotation = new THREE.Quaternion();
      const position = new THREE.Vector3();
      const scale = new THREE.Vector3();
      const up = new THREE.Vector3(0, 1, 0);

      if (!backdrop) throw new Error('CoT Verdant mountains were not prepared');
      const mountains = createMountainApron(manifest.seed, width, height, groundMaterial, groundSize, backdrop);
      group.add(backdrop.mesh);
      group.add(mountains.mesh);
      ownedGeometry.add(mountains.mesh.geometry);
      ownedMaterial.add(mountains.mesh.material);
      const barrier = airBarrier(width, height, tile);
      group.add(barrier.group);
      for (const geometry of barrier.geometries) ownedGeometry.add(geometry);
      ownedMaterial.add(barrier.material);

      for (const species of ['pine', 'oak', 'birch'] as const) {
        const kit = createGarageTreeKit({}, null, species, manifest.seed + species.length, 1);
        kit.trunkMaterial.color.set('#c4b6a0');
        kit.foliageMaterial.color.set('#829e76');
        treeKits.push(kit);
        for (const [zone, records] of [
          ['near', layout.trees.slice(0, 1850).filter(tree => tree.species === species)],
          ['hillside', layout.trees.slice(1850).filter(tree => tree.species === species)],
        ] as const) {
          for (const [name, geometry, material] of [
            ['trunk', kit.trunk, kit.trunkMaterial],
            ['foliage', kit.foliage, kit.foliageMaterial],
          ] as const) {
            const mesh = new THREE.InstancedMesh(geometry, material, records.length);
            mesh.name = `CoT ${species} ${name} / ${zone} exterior`;
            records.forEach((record, index) => {
              position.set(record.x, mountains.groundAt(record.x, record.z), record.z);
              rotation.setFromAxisAngle(up, record.yaw);
              scale.setScalar(record.size);
              modelMatrix.compose(position, rotation, scale);
              mesh.setMatrixAt(index, modelMatrix);
            });
            mesh.instanceMatrix.needsUpdate = true;
            mesh.computeBoundingSphere();
            mesh.castShadow = zone === 'near';
            mesh.receiveShadow = true;
            group.add(mesh);
          }
        }
      }

      const rockMaterial = new THREE.MeshStandardMaterial({ color: '#b2b4a7', vertexColors: true, roughness: 1 });
      ownedMaterial.add(rockMaterial);
      for (let variant = 0; variant < 3; variant++) {
        const geometry = buildHorizonBoulder(variant, manifest.seed, new THREE.Color('#898d7f'));
        ownedGeometry.add(geometry);
        for (const [zone, records] of [
          ['exterior', layout.rocks.filter(rock => rock.variant === variant)],
          ['border', layout.boundaryRocks.filter(rock => rock.variant === variant)],
        ] as const) {
          const mesh = new THREE.InstancedMesh(geometry, rockMaterial, records.length);
          mesh.name = `CoT ${zone} boulders / variant ${variant}`;
          records.forEach((record, index) => {
            position.set(record.x, mountains.groundAt(record.x, record.z) + record.size * 0.25, record.z);
            rotation.setFromAxisAngle(up, record.yaw);
            scale.setScalar(record.size);
            modelMatrix.compose(position, rotation, scale);
            mesh.setMatrixAt(index, modelMatrix);
          });
          mesh.instanceMatrix.needsUpdate = true;
          mesh.computeBoundingSphere();
          mesh.castShadow = zone === 'border';
          mesh.receiveShadow = true;
          group.add(mesh);
        }
      }

      const fenceMaterials = {
        fenceplank: new THREE.MeshStandardMaterial({ map: wood, color: '#ad9d7b',
          metalness: 0, roughness: 1, envMapIntensity: 0.05 }),
        fencepicket: new THREE.MeshStandardMaterial({ vertexColors: true,
          metalness: 0, roughness: 1, envMapIntensity: 0.05 }),
      };
      for (const kind of ['fenceplank', 'fencepicket'] as const) {
        const records = layout.boundaryFences.filter(fence => fence.kind === kind);
        const geometry = DESTRUCTIBLE_TYPES[kind].build(() => 0.5);
        ownedGeometry.add(geometry);
        ownedMaterial.add(fenceMaterials[kind]);
        const mesh = new THREE.InstancedMesh(geometry, fenceMaterials[kind], records.length);
        mesh.name = `CoT original ${kind} / discontinuous border`;
        records.forEach((record, index) => {
          position.set(record.x, 0, record.z);
          rotation.setFromAxisAngle(up, record.yaw);
          modelMatrix.compose(position, rotation, scale.setScalar(1));
          mesh.setMatrixAt(index, modelMatrix);
        });
        mesh.instanceMatrix.needsUpdate = true;
        mesh.computeBoundingSphere();
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        group.add(mesh);
      }
      if (Math.abs(FENCE_SEG - 2.4) > 1e-6) throw new Error('CoT fence pitch changed');

      // CoT's original distance pole and inhabiting-object geometry; all are visual only.
      const poleGeometry = makeTelephonePoleDistanceGeometry();
      ownedGeometry.add(poleGeometry);
      const poleMaterial = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, side: THREE.DoubleSide });
      ownedMaterial.add(poleMaterial);
      const poleMesh = new THREE.InstancedMesh(poleGeometry, poleMaterial, layout.poles.length);
      poleMesh.name = 'CoT telephone pole kit / exterior';
      layout.poles.forEach((record, index) => {
        position.set(record.x, mountains.groundAt(record.x, record.z), record.z);
        rotation.setFromAxisAngle(up, record.yaw);
        modelMatrix.compose(position, rotation, scale.setScalar(1));
        poleMesh.setMatrixAt(index, modelMatrix);
      });
      poleMesh.instanceMatrix.needsUpdate = true;
      poleMesh.computeBoundingSphere();
      poleMesh.castShadow = true;
      poleMesh.receiveShadow = true;
      group.add(poleMesh);
      const wirePoints: number[] = [];
      for (let i = 0; i + 1 < layout.poles.length; i++) {
        const a = layout.poles[i], b = layout.poles[i + 1];
        if (a.line !== b.line) continue;
        const across = Math.abs(a.x - b.x) > Math.abs(a.z - b.z);
        for (const side of [-0.47, 0.47]) {
          for (let segment = 0; segment < 8; segment++) {
            for (const step of [segment, segment + 1]) {
              const t = step / 8;
              const x = THREE.MathUtils.lerp(a.x, b.x, t) + (across ? 0 : side);
              const z = THREE.MathUtils.lerp(a.z, b.z, t) + (across ? side : 0);
              const y = THREE.MathUtils.lerp(mountains.groundAt(a.x, a.z), mountains.groundAt(b.x, b.z), t) +
                6.1 - Math.sin(Math.PI * t) * 0.55;
              wirePoints.push(x, y, z);
            }
          }
        }
      }
      const wireGeometry = new THREE.BufferGeometry();
      wireGeometry.setAttribute('position', new THREE.Float32BufferAttribute(wirePoints, 3));
      ownedGeometry.add(wireGeometry);
      const wireMaterial = new THREE.LineBasicMaterial({ color: '#302d29', transparent: true, opacity: 0.82 });
      ownedMaterial.add(wireMaterial);
      group.add(new THREE.LineSegments(wireGeometry, wireMaterial));
      const hayMaterial = new THREE.MeshStandardMaterial({ color: '#c7ae72', map: straw,
        roughness: 1, side: THREE.DoubleSide });
      ownedMaterial.add(hayMaterial);
      for (const kind of ['haystack', 'bale', 'stook'] as const) {
        const records = layout.hay.filter(hay => hay.kind === kind);
        const geometry = DESTRUCTIBLE_TYPES[kind].build(() => 0.48);
        ownedGeometry.add(geometry);
        const mesh = new THREE.InstancedMesh(geometry, hayMaterial, records.length);
        mesh.name = `CoT native ${kind} / exterior`;
        records.forEach((record, index) => {
          position.set(record.x, mountains.groundAt(record.x, record.z), record.z);
          rotation.setFromAxisAngle(up, record.yaw);
          modelMatrix.compose(position, rotation, scale.setScalar(record.size));
          mesh.setMatrixAt(index, modelMatrix);
        });
        mesh.instanceMatrix.needsUpdate = true;
        mesh.computeBoundingSphere();
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        group.add(mesh);
      }

      const houseMaterials = new Map<string, THREE.MeshStandardMaterial>();
      function houseMaterial(bucket: string, colors: boolean) {
        const key = bucket + ':' + colors;
        let material = houseMaterials.get(key);
        if (!material) {
          const map = bucket === 'roof' ? roof : bucket === 'wood' ? wood :
            bucket.startsWith('plaster') ? plaster : bucket === 'stone' ? stone : null;
          const tone = bucket === 'dark' ? '#353b39' : bucket === 'glass' ? '#94a8aa' :
            bucket === 'curtain' ? '#576b6d' : bucket === 'straw' ? '#a8925c' : '#ffffff';
          material = new THREE.MeshStandardMaterial({ color: tone, map, vertexColors: colors,
            roughness: bucket === 'glass' ? 0.38 : 0.94, metalness: bucket === 'glass' ? 0.1 : 0 });
          houseMaterials.set(key, material);
          ownedMaterial.add(material);
        }
        return material;
      }
      const houseTemplates = new Map<string, THREE.Group>();
      for (const house of [...layout.houses, ...layout.boundaryHouses]) {
        let template = houseTemplates.get(house.kind);
        if (!template) {
          const buckets = structureBuckets();
          const random = () => 0.5;
          if (house.kind === 'ranger') makeRangerLodge(random, buckets);
          else if (house.kind === 'school') makeSchoolhouse(random, buckets);
          else if (house.kind === 'bathhouse') makeTimberBathhouse(random, buckets);
          else if (house.kind === 'firestation') makeFireStation(random, buckets);
          else if (house.kind === 'foundry') makeFoundryOffice(random, buckets);
          else makeTavern(random, buckets);
          template = new THREE.Group();
          template.name = `CoT native ${house.kind} structure`;
          for (const [bucket, rawGeometries] of Object.entries(buckets)) {
            if (!rawGeometries?.length) continue;
            const geometries = rawGeometries as THREE.BufferGeometry[];
            // Native CoT parts mix indexed/non-indexed and emission-mask layouts.
            // Merge only equal vertex formats so the original attributes survive.
            const groups = new Map<string, THREE.BufferGeometry[]>();
            for (const geometry of geometries) {
              const attributes = Object.entries(geometry.attributes)
                .map(([name, attr]) => `${name}:${attr.itemSize}:${attr.normalized}`)
                .sort().join('|');
              const signature = `${geometry.index ? 'indexed' : 'plain'}:${attributes}`;
              const matching = groups.get(signature) ?? [];
              matching.push(geometry);
              groups.set(signature, matching);
            }
            for (const matching of groups.values()) {
              const merged = matching.length > 1 ? mergeGeometries(matching, false) : null;
              if (merged) for (const geometry of matching) geometry.dispose();
              for (const geometry of merged ? [merged] : matching) {
                ownedGeometry.add(geometry);
                template.add(new THREE.Mesh(geometry, houseMaterial(bucket, !!geometry.getAttribute('color'))));
              }
            }
          }
          houseTemplates.set(house.kind, template);
        }
        const instance = template.clone(true);
        instance.position.set(house.x, mountains.groundAt(house.x, house.z), house.z);
        instance.rotation.y = house.yaw;
        instance.traverse(object => {
          if (object instanceof THREE.Mesh) {
            object.castShadow = true;
            object.receiveShadow = true;
          }
        });
        group.add(instance);
      }

      return {
        group,
        counts: { trees: layout.trees.length, hillside_trees: 1600, ...backdrop.counts,
          rocks: layout.rocks.length,
          houses: layout.houses.length, poles: layout.poles.length, hay: layout.hay.length,
          barrier_panels: barrier.group.children.length,
          boundary_fences: layout.boundaryFences.length,
          boundary_rocks: layout.boundaryRocks.length,
          boundary_houses: layout.boundaryHouses.length },
        groundAt: mountains.groundAt,
        setTime(time: number) { barrier.setTime(time); },
        dispose() {
          backdrop?.mesh.removeFromParent();
          group.removeFromParent();
          for (const kit of treeKits) kit.dispose();
          for (const geometry of ownedGeometry) geometry.dispose();
          for (const material of ownedMaterial) material.dispose();
        },
      };
    },
    dispose() { backdrop?.dispose(); backdrop = null;
      for (const texture of Object.values(textures)) texture.dispose(); },
  };
}
