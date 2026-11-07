/** Persistent house flames and a long-lived, wind-carried wildfire smoke plume. */
import * as THREE from 'three';
import { mulberry32 } from '@cot/fx/particles.ts';
import { BURNING_HOUSE, HOUSE_FIRE_SCALE, houseGroundHeight } from './firefighterScene.ts';
import { houseSurfaceWorld } from './houseFireLayout.ts';
import { createVolumetricFire } from './volumetricFire.ts';
import { LATE_FX_LAYER } from '@cot/fx/layers.ts';
import type { Frame } from './protocol.ts';

export async function createBrushFires(scene: THREE.Scene) {
  const smokeTexture = await new THREE.TextureLoader().loadAsync('/fx/particles-smoke.png');
  const volume = createVolumetricFire(scene);
  const group = new THREE.Group(); group.name = 'Burning left-front house / rolling wildfire smoke'; scene.add(group);
  const clock = { value: 0 };
  // The original atlas provides silhouettes. A shared, seamless density field
  // adds moving internal folds and self-shading instead of a flat dark ribbon.
  const noiseSize = 128, noisePixels = new Float32Array(noiseSize * noiseSize);
  const noiseRandom = mulberry32(0x51A0CE);
  for (const [cells, weight] of [[4, .57], [8, .29], [16, .14]]) {
    const lattice = Array.from({ length: cells * cells }, () => noiseRandom());
    for (let y = 0; y < noiseSize; y++) for (let x = 0; x < noiseSize; x++) {
      const gx = x / noiseSize * cells, gy = y / noiseSize * cells;
      const ix = Math.floor(gx), iy = Math.floor(gy);
      const fx = gx - ix, fy = gy - iy;
      const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
      const at = (dx: number, dy: number) => lattice[(iy + dy) % cells * cells + (ix + dx) % cells];
      noisePixels[y * noiseSize + x] += THREE.MathUtils.lerp(
        THREE.MathUtils.lerp(at(0, 0), at(1, 0), sx),
        THREE.MathUtils.lerp(at(0, 1), at(1, 1), sx), sy) * weight;
    }
  }
  const turbulence = new THREE.DataTexture(Uint8Array.from(noisePixels, value => Math.round(value * 255)),
    noiseSize, noiseSize, THREE.RedFormat);
  turbulence.wrapS = turbulence.wrapT = THREE.RepeatWrapping;
  turbulence.magFilter = turbulence.minFilter = THREE.LinearFilter;
  turbulence.needsUpdate = true;
  function pool(texture: THREE.Texture) {
    const quad = new THREE.PlaneGeometry(1, 1);
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.index = quad.index;
    geometry.setAttribute('position', quad.getAttribute('position'));
    geometry.setAttribute('uv', quad.getAttribute('uv'));
    const origins = new THREE.InstancedBufferAttribute(new Float32Array(4096 * 4), 4);
    const variations = new THREE.InstancedBufferAttribute(new Float32Array(4096 * 4), 4);
    geometry.setAttribute('aOrigin', origins);
    geometry.setAttribute('aVariation', variations); geometry.instanceCount = 0;
    const material = new THREE.ShaderMaterial({
      uniforms: { uTime: clock, uMap: { value: texture }, uScale: { value: HOUSE_FIRE_SCALE },
        uTurbulence: { value: turbulence } },
      vertexShader: `
        attribute vec4 aOrigin;
        attribute vec4 aVariation;
        uniform float uTime;
        uniform float uScale;
        varying vec2 vUv;
        varying float vAge;
        varying float vLayer;
        varying float vSeed;
        void main() {
          vec3 p = aOrigin.xyz;
          float age, size, rotation;

          // Staggered 32–44 s lifetimes give a continuous, already established
          // plume. Strong initial buoyancy bends into a long downwind trail.
          float lifetime = mix(32.0, 44.0, aVariation.x);
          age = fract(uTime / lifetime + aOrigin.w);
          float t = age * lifetime;
          float seed = aVariation.z;
          float uplift = 55.0 * (1.0 - exp(-t / 18.0)) + t * 0.25;
          p.y += uplift + sin(t * 0.45 + seed * 31.0) * (1.0 + age * 3.0);
          p.x += 7.2 * t * mix(0.4, 1.0, smoothstep(0.0, 8.0, t));
          p.z += 2.4 * t;
          // Counter-rotating eddies keep the outline billowing, not a row of
          // identical expanding circles travelling in a perfectly straight line.
          p.x += sin(t * 0.7 + seed * 19.0) * (2.0 + age * 24.0);
          p.z += cos(t * 0.45 + seed * 13.0) * (2.0 + age * 26.0);
          size = mix(2.4, 9.5, pow(age, 0.7)) * uScale * mix(0.55, 1.2, aVariation.y);
          rotation = seed * 6.283185 + sin(t * 0.16 + seed * 7.0) * 0.4 +
            t * mix(-0.04, 0.04, aVariation.y);

          vAge = age;
          vLayer = aVariation.w;
          vSeed = aVariation.z;
          vec2 corner = mat2(cos(rotation), -sin(rotation), sin(rotation), cos(rotation)) * position.xy;
          vec4 mv = viewMatrix * vec4(p, 1.0);
          mv.xy += corner * size;
          gl_Position = projectionMatrix * mv;
          vUv = uv;
        }`,
      fragmentShader: `
        uniform sampler2D uMap;
        uniform sampler2D uTurbulence;
        uniform float uTime;
        varying vec2 vUv;
        varying float vAge;
        varying float vLayer;
        varying float vSeed;
        void main() {
          float frame = floor(vAge * 15.0);
          vec2 tile = vec2(mod(frame, 4.0), floor(frame / 4.0));
          vec4 texel = texture2D(uMap, (tile + vUv) / 4.0);
          float alpha;
          vec3 color;

          // Retain the dense soot until the last fifth of the long flight;
          // only then let it disperse into softer grey edges.
          float fade = smoothstep(0.0, 0.045, vAge) * (1.0 - smoothstep(0.8, 1.0, vAge));
          float edge = 1.0 - smoothstep(0.35, 0.53, length(vUv - vec2(0.5)));
          vec2 flow = vUv * 1.7 + vec2(vAge * 0.8, -vAge * 1.1) + vSeed * 13.0;
          float broad = texture2D(uTurbulence, flow).r;
          float fine = texture2D(uTurbulence, flow * 2.8 + broad * 0.4).r;
          float fold = smoothstep(0.2, 0.8, broad * 0.75 + fine * 0.25);
          float density = mix(0.35, 1.1, fold);
          alpha = (1.0 - exp(-texel.a * density * mix(0.30, 0.66, vLayer) * 2.1)) * fade * edge;
          float lighting = (1.0 - fold) * 0.65 + vUv.y * 0.25;
          color = mix(vec3(0.006, 0.008, 0.010), vec3(0.18, 0.185, 0.19), lighting);
          color *= mix(0.55, 1.0, vAge) * mix(1.0, 0.6, vLayer);

          if (alpha < 0.006) discard;
          gl_FragColor = vec4(color, alpha);
        }`,
      transparent: true, depthWrite: false, depthTest: true,
      blending: THREE.NormalBlending,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false; mesh.renderOrder = 23; mesh.layers.set(LATE_FX_LAYER); group.add(mesh);
    return { geometry, material, origins, variations, quad };
  }
  const smoke = pool(smokeTexture);
  let siteSignature = '';
  return {
    setTime(time: number) { clock.value = time; volume.setTime(time); },
    bindDepth(depth: THREE.DepthTexture, width: number, height: number) { volume.bindDepth(depth, width, height); },
    reset() { siteSignature = ''; smoke.geometry.instanceCount = 0; volume.sync(false); },
    sync(frame: Frame) {
      const sites = frame.units.filter(unit => unit.team === 'ENEMY' && unit.alive);
      const signature = sites.map(unit => `${unit.id}:${unit.position.join(',')}`).join('|');
      if (signature === siteSignature) return;
      siteSignature = signature;
      const rng = mulberry32(frame.manifest.seed);
      volume.sync(sites.length > 0);
      const count = sites.length ? 176 : 0;
      for (let i = 0; i < count; i++) {
        const core = i % 3 === 0 ? 1 : 0;
        const x = (rng() * 2 - 1) * (core ? 2 : 9), z = (rng() * 2 - 1) * 11;
        const point = houseSurfaceWorld(x, z);
        const sourceHeight = Math.abs(x) < 4.3 && Math.abs(z) < 7 ? 7.6 - Math.abs(x) / 4.3 * 3 : .5;
        smoke.origins.setXYZW(i, point.x, houseGroundHeight() + sourceHeight, point.z, rng());
        smoke.variations.setXYZW(i, rng(), rng(), rng(), core);
      }
      smoke.geometry.instanceCount = count;
      smoke.origins.needsUpdate = smoke.variations.needsUpdate = true;
    },
    state() { return { ...volume.state(), fire_volumes: 1, smoke_particles: smoke.geometry.instanceCount,
      smoke_scale: HOUSE_FIRE_SCALE, burning_houses: 1,
      fire_source: BURNING_HOUSE, smoke_style: 'dense-rolling-charcoal-plume',
      smoke_lifetime_s: [32, 44], smoke_wind_mps: [7.2, 2.4], smoke_fade_after_fraction: .8,
      smoke_plume_travel_m: 330 }; },
    dispose() {
      smoke.geometry.dispose(); smoke.quad.dispose(); smoke.material.dispose(); volume.dispose();
      smokeTexture.dispose(); turbulence.dispose(); group.removeFromParent();
    },
  };
}
