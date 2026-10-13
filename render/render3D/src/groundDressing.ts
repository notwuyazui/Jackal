/** Seeded, visual-only CoT ground dressing. The Jackal map and physics remain flat. */
import * as THREE from 'three';
import { buildBushCards, buildGrassTuftGeometry, makeGrassCardTexture,
  makeLeafClusterTexture } from '@cot/world/vegetation.ts';
import dirtUrl from '../public/textures/terrain/Ground071_1K-JPG_Color.jpg?url';
import dirtNormalUrl from '../public/textures/terrain/Ground071_1K-JPG_NormalGL.jpg?url';
import dirtRoughUrl from '../public/textures/terrain/Ground071_1K-JPG_Roughness.jpg?url';
import dirtAoUrl from '../public/textures/terrain/Ground071_1K-JPG_AmbientOcclusion.jpg?url';
import dryUrl from '../public/textures/terrain/withered_grass_diff_1k.jpg?url';
import { routes, center, distanceToRoad, onRoad, roadRelief } from './roads.ts';
import type { Manifest } from './protocol.ts';

function seeded(seed: number) {
  let value = seed | 0;
  return () => {
    value = value + 0x6d2b79f5 | 0;
    let t = Math.imul(value ^ value >>> 15, 1 | value);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function patchMaterial(map: THREE.Texture) {
  const material = new THREE.MeshStandardMaterial({ map, color: '#c6bd98',
    roughness: 1, metalness: 0, envMapIntensity: 0.04, transparent: true, opacity: 0.67,
    depthWrite: false, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -1 });
  material.onBeforeCompile = shader => {
    shader.fragmentShader = shader.fragmentShader.replace('#include <map_fragment>',
      '#include <map_fragment>\n diffuseColor.a *= 1.0 - smoothstep(0.31, 0.51, length(vMapUv - vec2(0.5)));');
  };
  material.customProgramCacheKey = () => 'jackal-ground-patch-v2';
  return material;
}

function verdantRoadMaterial(color: THREE.Texture, normal: THREE.Texture,
  roughness: THREE.Texture, ao: THREE.Texture, grass: THREE.Texture, width: number) {
  const material = new THREE.MeshStandardMaterial({ map: color, normalMap: normal,
    normalScale: new THREE.Vector2(1.05, 1.05), roughnessMap: roughness,
    aoMap: ao, aoMapIntensity: 0.85, color: '#9e9378', roughness: 1,
    metalness: 0, envMapIntensity: 0.04, alphaTest: 0.08,
    depthWrite: true, side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: -1 });
  // The CoT Verdant road uses a compacted dirt core, two wheel ruts and a
  // noisy shoulder. Keep that profile in the decal so the Jackal ground stays
  // physically flat and the path runs seamlessly beyond its border.
  material.onBeforeCompile = shader => {
    shader.uniforms.uGrassRoad = { value: grass };
    shader.vertexShader = 'varying vec2 vRoadUv;\n' + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace('#include <uv_vertex>',
      '#include <uv_vertex>\n vRoadUv = uv;');
    shader.fragmentShader = 'varying vec2 vRoadUv; uniform sampler2D uGrassRoad;\n' + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace('#include <map_fragment>',
      `#include <map_fragment>
      float acrossM = abs(vRoadUv.x - 0.5) * ${width.toFixed(3)};
      float soilNoise = texture2D(map, vMapUv * vec2(0.17, 0.21) + vec2(0.29, 0.53)).r;
      float grains = texture2D(map, vMapUv * vec2(1.7, 1.35) + vec2(0.41, 0.17)).g;
      float fineGrains = texture2D(map, vMapUv * vec2(4.1, 3.7) + vec2(0.17, 0.39)).r;
      float edge = 1.0 - smoothstep(${(width / 2 - 1.15).toFixed(3)}, ${(width / 2).toFixed(3)},
        acrossM + (soilNoise - 0.5) * 0.7);
      float core = 1.0 - smoothstep(${(width / 2 - 1.65).toFixed(3)}, ${(width / 2 - 0.55).toFixed(3)}, acrossM);
      float chatter = 0.83 + 0.17 * sin(vRoadUv.y * 4.7 + soilNoise * 4.0);
      float wheelRut = exp(-pow((acrossM - 1.55) / 0.43, 2.0)) * core * chatter;
      float crown = 1.0 - smoothstep(0.0, 1.25, acrossM);
      vec3 meadowTone = texture2D(uGrassRoad, vMapUv * 0.63 + vec2(0.23, 0.71)).rgb;
      diffuseColor.rgb = mix(diffuseColor.rgb, meadowTone, (1.0 - core) * 0.18);
      float soilGray = dot(diffuseColor.rgb, vec3(0.34, 0.45, 0.21));
      diffuseColor.rgb = mix(diffuseColor.rgb, vec3(soilGray), core * 0.18);
      diffuseColor.rgb *= 0.96 + (soilNoise - 0.5) * 0.18 +
        (grains - 0.5) * 0.20 + (fineGrains - 0.5) * 0.18;
      diffuseColor.rgb *= 1.0 - wheelRut * 0.35;
      diffuseColor.rgb *= 1.0 + crown * 0.055;
      float shoulder = (1.0 - core) * edge;
      diffuseColor.rgb *= 1.0 - shoulder * (0.08 + grains * 0.09);
      diffuseColor.a *= edge;`);
    shader.fragmentShader = shader.fragmentShader.replace('#include <roughnessmap_fragment>',
      '#include <roughnessmap_fragment>\n roughnessFactor = max(0.94, roughnessFactor);');
  };
  material.customProgramCacheKey = () => `jackal-verdant-road-cut-ruts-v5-${width}`;
  return material;
}

export async function createGroundDressingFactory(loader: THREE.TextureLoader, grassColor: THREE.Texture) {
  const [dirt, dirtNormal, dirtRough, dirtAo, dry] = await Promise.all([
    loader.loadAsync(dirtUrl), loader.loadAsync(dirtNormalUrl),
    loader.loadAsync(dirtRoughUrl), loader.loadAsync(dirtAoUrl), loader.loadAsync(dryUrl),
  ]);
  for (const texture of [dirt, dirtNormal, dirtRough, dirtAo, dry]) {
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.anisotropy = 8;
  }
  for (const texture of [dirt, dry]) texture.colorSpace = THREE.SRGBColorSpace;
  for (const texture of [dirt, dirtNormal, dirtRough, dirtAo]) texture.repeat.set(2.6, 2);
  const bladeMaps = [makeGrassCardTexture(seeded(24681), 0), makeGrassCardTexture(seeded(51912), 1)];
  for (const texture of bladeMaps) texture.colorSpace = THREE.SRGBColorSpace;
  const bushMap = makeLeafClusterTexture(seeded(17032));
  bushMap.colorSpace = THREE.SRGBColorSpace;

  return {
    build(manifest: Manifest, groundAt: (x: number, z: number) => number,
      blocked: (x: number, z: number) => boolean, unitStarts: Array<{ x: number; z: number }>) {
      const group = new THREE.Group();
      group.name = 'CoT mixed meadow, exposed soil and continuous farm tracks / visual only';
      const geometries: THREE.BufferGeometry[] = [];
      const materials: THREE.Material[] = [];
      const rng = seeded(manifest.seed ^ 0x3c7f);
      const width = manifest.map.size[0] * manifest.meters_per_pixel;
      const height = manifest.map.size[1] * manifest.meters_per_pixel;
      const halfW = width / 2, halfH = height / 2;
      const dryPatchMaterial = patchMaterial(dry);
      materials.push(dryPatchMaterial);

      for (const [index, route] of routes.entries()) {
        const roadMaterial = verdantRoadMaterial(dirt, dirtNormal, dirtRough, dirtAo, grassColor, route.width);
        materials.push(roadMaterial);
        const extent = (route.across ? halfW : halfH) + 185;
        const segments = 220;
        const crossSteps = 64;
        const positions: number[] = [], uv: number[] = [], indices: number[] = [];
        for (let i = 0; i <= segments; i++) {
          const along = -extent + i / segments * extent * 2;
          const cross = center(route, along);
          for (let j = 0; j <= crossSteps; j++) {
            const crossM = (j / crossSteps - 0.5) * route.width;
            const x = route.across ? along : cross + crossM;
            const z = route.across ? cross + crossM : along;
            positions.push(x, groundAt(x, z) + 0.014 + index * 0.002 +
              roadRelief(crossM, route.width), z);
            uv.push(j / crossSteps, (along + extent) / 4);
          }
          if (i < segments) {
            for (let j = 0; j < crossSteps; j++) {
              const a = i * (crossSteps + 1) + j;
              const b = a + crossSteps + 1;
              if (route.across) indices.push(a, a + 1, b, a + 1, b + 1, b);
              else indices.push(a, b, a + 1, a + 1, b, b + 1);
            }
          }
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
        geometry.setAttribute('uv2', new THREE.Float32BufferAttribute(uv, 2));
        geometry.setIndex(indices);
        geometry.computeVertexNormals();
        geometries.push(geometry);
        const mesh = new THREE.Mesh(geometry, roadMaterial);
        mesh.name = `CoT Verdant Ground071 compacted earth path ${index + 1}`;
        mesh.receiveShadow = true;
        mesh.renderOrder = 1;
        group.add(mesh);
      }

      const patchGeometry = new THREE.CircleGeometry(1, 24);
      patchGeometry.rotateX(-Math.PI / 2);
      geometries.push(patchGeometry);
      const patches = new THREE.InstancedMesh(patchGeometry, dryPatchMaterial, 560);
      patches.name = 'CoT withered-grass and bare earth patches';
      const matrix = new THREE.Matrix4(), rotation = new THREE.Quaternion();
      const position = new THREE.Vector3(), scale = new THREE.Vector3();
      const up = new THREE.Vector3(0, 1, 0);
      let patchCount = 0;
      for (let attempt = 0; patchCount < 560 && attempt < 1800; attempt++) {
        const x = (rng() * 2 - 1) * (halfW + 110);
        const z = (rng() * 2 - 1) * (halfH + 110);
        if (Math.abs(x) <= halfW && Math.abs(z) <= halfH && blocked(x, z)) continue;
        if (distanceToRoad(x, z) < 6) continue;
        position.set(x, groundAt(x, z) + 0.014, z);
        rotation.setFromAxisAngle(up, rng() * Math.PI * 2);
        scale.set(2.0 + rng() * 4.5, 1, 1.2 + rng() * 3.0);
        matrix.compose(position, rotation, scale);
        patches.setMatrixAt(patchCount++, matrix);
      }
      patches.count = patchCount;
      patches.instanceMatrix.needsUpdate = true;
      patches.computeBoundingSphere();
      patches.renderOrder = 1;
      group.add(patches);

      let tuftCount = 0;
      for (const variant of [0, 1] as const) {
        const target = variant === 0 ? 11000 : 6200;
        const geometry = buildGrassTuftGeometry(variant === 0 ? 0.62 : 0.85,
          variant === 0 ? 0.42 : 0.68);
        geometries.push(geometry);
        const material = new THREE.MeshBasicMaterial({ map: bladeMaps[variant],
          color: variant === 0 ? '#e7efd3' : '#e7dda7', alphaTest: 0.34,
          side: THREE.DoubleSide, depthWrite: true });
        materials.push(material);
        const mesh = new THREE.InstancedMesh(geometry, material, target);
        mesh.name = variant === 0 ? 'CoT lush meadow tufts' : 'CoT dry field tufts';
        let placed = 0;
        for (let attempt = 0; placed < target && attempt < target * 4; attempt++) {
          const inner = attempt % 3 !== 0;
          const x = (rng() * 2 - 1) * (halfW + (inner ? -16 : 80));
          const z = (rng() * 2 - 1) * (halfH + (inner ? -16 : 80));
          if (Math.abs(x) < halfW && Math.abs(z) < halfH && blocked(x, z)) continue;
          if (onRoad(x, z, 0.55)) continue;
          position.set(x, groundAt(x, z), z);
          rotation.setFromAxisAngle(up, rng() * Math.PI * 2);
          scale.setScalar(0.75 + rng() * 0.75);
          matrix.compose(position, rotation, scale);
          mesh.setMatrixAt(placed++, matrix);
        }
        mesh.count = placed;
        tuftCount += placed;
        mesh.instanceMatrix.needsUpdate = true;
        mesh.computeBoundingSphere();
        group.add(mesh);
        // Match the playable meadow's local density across the border. Farther
        // out the tree line, settlements and hill texture carry the detail.
        const exteriorTarget = variant === 0 ? 12000 : 8000;
        const exteriorMesh = new THREE.InstancedMesh(geometry, material, exteriorTarget);
        exteriorMesh.name = `${mesh.name} / exterior meadow continuity`;
        let exteriorPlaced = 0;
        for (let attempt = 0; exteriorPlaced < exteriorTarget && attempt < exteriorTarget * 8; attempt++) {
          const x = (rng() * 2 - 1) * (halfW + 55);
          const z = (rng() * 2 - 1) * (halfH + 55);
          if (Math.abs(x) < halfW && Math.abs(z) < halfH) continue;
          if (onRoad(x, z, 0.55)) continue;
          position.set(x, groundAt(x, z), z);
          rotation.setFromAxisAngle(up, rng() * Math.PI * 2);
          scale.setScalar(0.78 + rng() * 0.72);
          matrix.compose(position, rotation, scale);
          exteriorMesh.setMatrixAt(exteriorPlaced++, matrix);
        }
        exteriorMesh.count = exteriorPlaced;
        tuftCount += exteriorPlaced;
        exteriorMesh.instanceMatrix.needsUpdate = true;
        exteriorMesh.computeBoundingSphere();
        group.add(exteriorMesh);
      }
      let bushCount = 0;
      for (const variant of [0, 1] as const) {
        const geometry = buildBushCards(seeded(manifest.seed + 712 + variant));
        geometries.push(geometry);
        const material = new THREE.MeshStandardMaterial({ map: bushMap, vertexColors: true,
          color: variant === 0 ? '#a8b995' : '#c0c6a5', alphaTest: 0.38,
          alphaToCoverage: true, side: THREE.DoubleSide, roughness: 1 });
        materials.push(material);
        const target = 210;
        const mesh = new THREE.InstancedMesh(geometry, material, target);
        mesh.name = `CoT native bush spray ${variant} / visual only`;
        let placed = 0;
        for (let attempt = 0; placed < target && attempt < 4000; attempt++) {
          const inner = placed < 165;
          const x = (rng() * 2 - 1) * (halfW + (inner ? -18 : 48));
          const z = (rng() * 2 - 1) * (halfH + (inner ? -18 : 48));
          if (inner && blocked(x, z)) continue;
          if (!inner && Math.abs(x) < halfW && Math.abs(z) < halfH) continue;
          if (distanceToRoad(x, z) < 6.5 ||
            unitStarts.some(start => Math.hypot(start.x - x, start.z - z) < 8)) continue;
          position.set(x, groundAt(x, z), z);
          rotation.setFromAxisAngle(up, rng() * Math.PI * 2);
          scale.setScalar(1.22 + rng() * 0.64);
          matrix.compose(position, rotation, scale);
          mesh.setMatrixAt(placed++, matrix);
        }
        mesh.count = placed;
        bushCount += placed;
        mesh.instanceMatrix.needsUpdate = true;
        mesh.computeBoundingSphere();
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        group.add(mesh);
      }
      return {
        group, counts: { paths: routes.length, patches: patchCount, tufts: tuftCount,
          bushes: bushCount, road_style: 'cot-verdant-ground071-cut-ruts-v5',
          road_grass_clearance_m: 0.55 },
        dispose() {
          group.removeFromParent();
          for (const geometry of geometries) geometry.dispose();
          for (const material of materials) material.dispose();
        },
      };
    },
    dispose() { for (const texture of [dirt, dirtNormal, dirtRough, dirtAo, dry, bushMap, ...bladeMaps]) texture.dispose(); },
  };
}
