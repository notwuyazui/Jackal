/** A depth-clipped, turbulent fire volume rooted in the house and its ground. */
import * as THREE from 'three';
import { LATE_FX_LAYER } from '@cot/fx/layers.ts';
import { mulberry32 } from '@cot/fx/particles.ts';
import { BURNING_HOUSE, houseGroundHeight } from './firefighterScene.ts';

export const FIRE_HALF_EXTENT = { x: 32, z: 36 };
export const FIRE_HEIGHT = 34;

export function createVolumetricFire(scene: THREE.Scene) {
  const size = 64, rng = mulberry32(0xF1A6E), pixels = new Uint8Array(size ** 3);
  for (let i = 0; i < pixels.length; i++) pixels[i] = Math.floor(rng() * 256);
  const noise = new THREE.Data3DTexture(pixels, size, size, size);
  noise.format = THREE.RedFormat; noise.type = THREE.UnsignedByteType;
  noise.minFilter = noise.magFilter = THREE.LinearFilter;
  noise.wrapS = noise.wrapT = noise.wrapR = THREE.RepeatWrapping;
  noise.unpackAlignment = 1; noise.needsUpdate = true;
  const clock = { value: 0 };
  const uniforms = {
    uTime: clock, uNoise: { value: noise }, uDepth: { value: null as THREE.DepthTexture | null },
    uResolution: { value: new THREE.Vector2(1, 1) },
    uNear: { value: .25 }, uFar: { value: 16000 }, uInverseWorld: { value: new THREE.Matrix4() },
  };
  const material = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3, uniforms,
    vertexShader: `
      out vec3 vLocal;
      out vec3 vWorld;
      void main() {
        vLocal = position;
        vec4 world = modelMatrix * vec4(position, 1.0);
        vWorld = world.xyz;
        gl_Position = projectionMatrix * viewMatrix * world;
      }`,
    fragmentShader: `
      precision highp float;
      precision highp sampler3D;
      uniform sampler3D uNoise;
      uniform sampler2D uDepth;
      uniform float uTime, uNear, uFar;
      uniform vec2 uResolution;
      uniform mat4 uInverseWorld;
      in vec3 vLocal;
      in vec3 vWorld;
      out vec4 outColor;
      float noise3(vec3 p) { return texture(uNoise, p).r; }
      float turbulence(vec3 p) {
        return noise3(p) * 0.65 + noise3(p * 2.03 + 0.21) * 0.27 + noise3(p * 4.07 + 0.47) * 0.08;
      }
      float houseDistance(vec2 p) {
        vec2 q = abs(p) - vec2(5.0, 8.5);
        return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
      }
      vec2 fireAt(vec3 p) {
        // Coherent upward advection, two interacting eddies, and a gentler
        // downwind lean break the fire into rolling sheets and detached wisps.
        vec3 moving = vec3(p.x * 0.018, (p.y - uTime * 5.6) * 0.010, p.z * 0.018);
        vec2 curl = vec2(noise3(moving * 0.53 + 0.17), noise3(moving * 0.61 + 0.73)) - 0.5;
        vec2 warped = p.xz - vec2(0.07, 0.14) * p.y - curl * (1.0 + p.y * 0.12);
        float n = turbulence(moving + vec3(curl.x, 0.0, curl.y) * 0.11);
        float edgeNoise = noise3(vec3(p.xz * 0.015, 0.31));
        float radius = length(warped / vec2(31.0, 35.0));
        float fuel = 1.0 - smoothstep(0.78, 1.0, radius + (edgeNoise - 0.5) * 0.12);
        float distance = max(0.0, houseDistance(warped));
        float height = mix(11.5, 32.0, exp(-distance * distance / 20.0));
        // Ground ring burns continuously; the house is the taller fuel source.
        height *= 0.9 + noise3(moving * 0.37 + 0.43) * 0.22;
        float h = p.y / height;
        float envelope = max(0.0, 1.0 - h);
        float sheets = smoothstep(0.36, 0.77, n * 0.72 + envelope * 0.46);
        float density = sheets * sqrt(envelope) * fuel;
        density *= smoothstep(0.0, 0.22, p.y) * (1.0 - smoothstep(0.96, 1.0, h));
        float heat = clamp(envelope * 0.22 + sheets * 0.3 + n * 0.5, 0.0, 1.0);
        return vec2(density, heat);
      }
      vec3 flameColor(float heat) {
        vec3 edge = vec3(1.25, 0.055, 0.0015);
        vec3 orange = vec3(3.0, 0.46, 0.013);
        vec3 yellow = vec3(4.4, 1.25, 0.09);
        vec3 color = mix(edge, orange, smoothstep(0.13, 0.58, heat));
        color = mix(color, yellow, smoothstep(0.58, 0.9, heat));
        return mix(color, vec3(5.5, 3.0, 0.65), smoothstep(0.95, 1.0, heat) * 0.18);
      }
      void main() {
        vec3 origin = (uInverseWorld * vec4(cameraPosition, 1.0)).xyz;
        vec3 ray = normalize(vLocal - origin);
        vec3 inverseRay = 1.0 / ray;
        vec3 t0 = (vec3(-32.0, 0.0, -36.0) - origin) * inverseRay;
        vec3 t1 = (vec3(32.0, 34.0, 36.0) - origin) * inverseRay;
        vec3 entry = min(t0, t1), exit = max(t0, t1);
        float first = max(0.0, max(max(entry.x, entry.y), entry.z));
        float last = min(min(exit.x, exit.y), exit.z);
        // Stop each ray at the already rendered house/tree/terrain surface.
        // Fire behind a wall cannot shine through the wall as an overlay.
        float depth = texture(uDepth, gl_FragCoord.xy / uResolution).r;
        float viewZ = (uNear * uFar) / ((uFar - uNear) * depth - uFar);
        vec3 worldRay = normalize(vWorld - cameraPosition);
        float forward = max(0.001, -(viewMatrix * vec4(worldRay, 0.0)).z);
        last = min(last, -viewZ / forward);
        if (last <= first) discard;
        float stepSize = clamp((last - first) / 88.0, 0.2, 1.1);
        // Screen-stable interleaving suppresses marching bands without random
        // frame-to-frame shimmer in the recording.
        float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
        float t = first + jitter * stepSize;
        vec3 radiance = vec3(0.0);
        float transmittance = 1.0;
        for (int i = 0; i < 96; i++) {
          if (t >= last || transmittance < 0.025) break;
          vec2 fire = fireAt(origin + ray * t);
          float density = fire.x * smoothstep(0.0, 0.45, last - t);
          float absorbed = 1.0 - exp(-density * stepSize * 0.48);
          radiance += transmittance * absorbed * flameColor(fire.y);
          transmittance *= 1.0 - absorbed;
          t += stepSize;
        }
        float alpha = 1.0 - transmittance;
        if (alpha < 0.008) discard;
        outColor = vec4(radiance / max(alpha, 0.001), alpha);
      }`,
    transparent: true, depthWrite: false, depthTest: false,
    blending: THREE.NormalBlending, toneMapped: false, side: THREE.FrontSide,
  });
  const geometry = new THREE.BoxGeometry(64, FIRE_HEIGHT, 72);
  geometry.translate(0, FIRE_HEIGHT / 2, 0);
  const volume = new THREE.Mesh(geometry, material);
  volume.name = 'Turbulent, depth-clipped house and ground fire volume';
  volume.layers.set(LATE_FX_LAYER); volume.renderOrder = 22; volume.frustumCulled = false;
  volume.position.set(BURNING_HOUSE.x, 0, BURNING_HOUSE.z);
  volume.rotation.y = BURNING_HOUSE.yaw; volume.visible = false;
  scene.add(volume);
  volume.onBeforeRender = (_renderer, _scene, camera) => {
    const perspective = camera as THREE.PerspectiveCamera;
    uniforms.uNear.value = perspective.near; uniforms.uFar.value = perspective.far;
  };

  // Small glowing embers leave the burning fuel, climb and dim independently.
  const emberCount = 160, positions = new Float32Array(emberCount * 3), seeds = new Float32Array(emberCount * 4);
  for (let i = 0; i < emberCount; i++) {
    const angle = rng() * Math.PI * 2, radius = Math.sqrt(rng());
    positions.set([Math.cos(angle) * radius * 14, .2, Math.sin(angle) * radius * 18], i * 3);
    seeds.set([rng(), rng(), rng(), rng()], i * 4);
  }
  const emberGeometry = new THREE.BufferGeometry();
  emberGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  emberGeometry.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 4));
  const emberMaterial = new THREE.ShaderMaterial({
    uniforms: { uTime: clock }, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, toneMapped: false,
    vertexShader: `
      attribute vec4 aSeed;
      uniform float uTime;
      varying float vAge;
      void main() {
        float age = fract(uTime / (4.0 + aSeed.x * 5.0) + aSeed.y);
        vAge = age;
        vec3 p = position;
        p.y += age * (9.0 + aSeed.z * 24.0);
        p.x += age * age * 3.3 + sin(age * 11.0 + aSeed.w * 21.0) * age * 1.5;
        p.z += age * age * 6.8;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = clamp((1.0 - age * 0.7) * (180.0 + aSeed.w * 220.0) / -mv.z, 0.5, 2.2);
      }`,
    fragmentShader: `
      varying float vAge;
      void main() {
        float glow = 1.0 - smoothstep(0.05, 0.5, length(gl_PointCoord - vec2(0.5)));
        float alpha = glow * smoothstep(0.0, 0.06, vAge) * (1.0 - smoothstep(0.55, 1.0, vAge));
        gl_FragColor = vec4(mix(vec3(3.0, 1.0, 0.08), vec3(0.8, 0.04, 0.002), vAge), alpha);
      }`,
  });
  const embers = new THREE.Points(emberGeometry, emberMaterial);
  embers.name = 'Wind-carried burning embers'; embers.layers.set(LATE_FX_LAYER); embers.renderOrder = 24;
  embers.position.copy(volume.position); embers.rotation.y = volume.rotation.y; embers.visible = false;
  scene.add(embers);

  return {
    setTime(time: number) { clock.value = time; },
    bindDepth(depth: THREE.DepthTexture, width: number, height: number) {
      uniforms.uDepth.value = depth; uniforms.uResolution.value.set(width, height);
    },
    sync(active: boolean) {
      volume.position.y = embers.position.y = houseGroundHeight();
      volume.updateMatrixWorld(true); uniforms.uInverseWorld.value.copy(volume.matrixWorld).invert();
      volume.visible = embers.visible = active;
    },
    state() { return { fire_style: 'raymarched-turbulent-volume', fire_height_m: FIRE_HEIGHT,
      fire_footprint_m: [64, 72], fire_size_multiplier: 2.3, fire_ray_steps: 96,
      fire_emission: 'whole-house-and-surrounding-ground', fire_embers: emberCount }; },
    dispose() {
      volume.removeFromParent(); embers.removeFromParent(); geometry.dispose(); material.dispose(); noise.dispose();
      emberGeometry.dispose(); emberMaterial.dispose();
    },
  };
}
