/** Draped charred earth, glowing seams and ash around the burning building. */
import * as THREE from 'three';
import { houseSurfaceWorld } from './houseFireLayout.ts';
import { FIRE_GROUND_SIZE } from './firefighterScene.ts';

export function createBurningGround(groundAt: (x: number, z: number) => number) {
  const geometry = new THREE.PlaneGeometry(...FIRE_GROUND_SIZE, 72, 80);
  const position = geometry.getAttribute('position');
  for (let i = 0; i < position.count; i++) {
    const point = houseSurfaceWorld(position.getX(i), -position.getY(i));
    position.setXYZ(i, point.x, groundAt(point.x, point.z) + .055, point.z);
  }
  geometry.computeVertexNormals();
  const clock = { value: 0 };
  const material = new THREE.MeshStandardMaterial({ color: '#252019', roughness: 1,
    transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2,
    side: THREE.DoubleSide, metalness: 0 });
  material.onBeforeCompile = shader => {
    shader.uniforms.uBurnTime = clock;
    shader.vertexShader = 'varying vec2 vScorchUv;\n' + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n vScorchUv = uv;');
    shader.fragmentShader = `
      varying vec2 vScorchUv;
      uniform float uBurnTime;
      float burnHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      float burnNoise(vec2 p) {
        vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
        return mix(mix(burnHash(i), burnHash(i + vec2(1.0, 0.0)), f.x),
          mix(burnHash(i + vec2(0.0, 1.0)), burnHash(i + vec2(1.0)), f.x), f.y);
      }
      ` + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace('#include <map_fragment>', `
      #include <map_fragment>
      vec2 p = (vScorchUv - 0.5) * 2.0;
      float broad = burnNoise(p * 5.0), grain = burnNoise(p * 95.0);
      float radius = length(p) + (broad - 0.5) * 0.16;
      float scorch = 1.0 - smoothstep(0.81, 1.0, radius);
      diffuseColor.a *= scorch;
      diffuseColor.rgb *= 0.45 + grain * 0.9;
      diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.12, 0.105, 0.09),
        smoothstep(0.62, 0.82, burnNoise(p * 22.0)) * 0.4);`);
    shader.fragmentShader = shader.fragmentShader.replace('#include <emissivemap_fragment>', `
      #include <emissivemap_fragment>
      float seam = smoothstep(0.72, 0.91, burnNoise(p * 18.0));
      seam *= 1.0 - smoothstep(0.67, 0.88, burnNoise(p * 21.0 + 3.7));
      float pulse = 0.7 + 0.3 * sin(uBurnTime * 2.8 + p.x * 19.0 + p.y * 23.0);
      totalEmissiveRadiance += vec3(1.6, 0.13, 0.006) * seam * pulse * scorch * 0.7;`);
  };
  material.customProgramCacheKey = () => 'burning-charred-ground-v1';
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'Scorched earth and glowing embers around the house'; mesh.receiveShadow = true;
  return { mesh, setTime(time: number) { clock.value = time; },
    dispose() { mesh.removeFromParent(); geometry.dispose(); material.dispose(); } };
}
