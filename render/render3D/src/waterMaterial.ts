/** Thin refractive water, moving surface ripples and irregular aerated wisps. */
import * as THREE from 'three';

const SPREAD_SECONDS = 1.2;
const FAR_SPREAD_SECONDS = 6;
export function waterSpread(age: number) {
  const u = THREE.MathUtils.clamp(age / SPREAD_SECONDS, 0, 1);
  const far = THREE.MathUtils.clamp(age / FAR_SPREAD_SECONDS, 0, 1);
  return .14 + .86 * u * u * (3 - 2 * u) + .5 * far * far * (3 - 2 * far);
}

export function createWaterMaterials(clock: { value: number },
  options: { near?: number; far?: number; sunDirection?: THREE.Vector3 } = {}) {
  const shared = {
    uWaterTime: clock,
    uSpreadTime: { value: SPREAD_SECONDS },
    uFarSpreadTime: { value: FAR_SPREAD_SECONDS },
    uSceneColor: { value: null as THREE.Texture | null },
    uSceneDepth: { value: null as THREE.DepthTexture | null },
    uHasScene: { value: 0 },
    uResolution: { value: new THREE.Vector2(1, 1) },
    uNearFar: { value: new THREE.Vector2(options.near ?? .25, options.far ?? 16000) },
    uSunDirection: { value: (options.sunDirection ?? new THREE.Vector3(.45, .65, .4)).clone().normalize() },
  };
  const vertexShader = /* glsl */`
    uniform float uWaterTime;
    uniform float uSpreadTime;
    uniform float uFarSpreadTime;
    attribute float aWaterAge;
    varying vec3 vWaterWorld;
    varying vec3 vWaterNormal;
    varying float vViewDepth;
    varying float vWaterAge;
    void main() {
      vec3 p = position;
      vec3 n = normal;
      mat4 instance = mat4(1.0);
      #ifdef USE_INSTANCING
        instance = instanceMatrix;
        mat3 im = mat3(instanceMatrix);
        n /= max(vec3(dot(im[0], im[0]), dot(im[1], im[1]), dot(im[2], im[2])), vec3(0.00001));
        n = im * n;
      #endif
      vec3 seed = (modelMatrix * instance * vec4(p, 1.0)).xyz;
      // The near jet opens quickly, then keeps widening and breaking up as it
      // travels. World-space ripples meet across adjacent arc sections.
      float flow = sin(dot(seed, vec3(4.1, 3.2, 2.7)) - uWaterTime * 19.0);
      float fine = sin(dot(seed, vec3(9.3, -6.8, 8.1)) - uWaterTime * 29.0);
      float spread = mix(.14, 1.0, smoothstep(0.0, uSpreadTime, aWaterAge))
        + .5 * smoothstep(0.0, uFarSpreadTime, aWaterAge);
      float broken = smoothstep(.06, 2.0, aWaterAge);
      p.xz *= spread * (.86 + flow * .035 + fine * .025);
      p.x += sin(dot(seed, vec3(.7,1.3,.8)) - uWaterTime * 13.0) * .05 * spread * broken;
      p.z += cos(dot(seed, vec3(1.1,.6,1.3)) - uWaterTime * 16.0) * .05 * spread * broken;
      vec4 world = modelMatrix * instance * vec4(p, 1.0);
      vec4 view = viewMatrix * world;
      vWaterWorld = world.xyz;
      vWaterNormal = normalize(mat3(modelMatrix) * n);
      vViewDepth = -view.z;
      vWaterAge = aWaterAge;
      gl_Position = projectionMatrix * view;
    }`;
  const fragmentShader = /* glsl */`
    uniform float uWaterTime;
    uniform float uOpacity;
    uniform sampler2D uSceneColor;
    uniform sampler2D uSceneDepth;
    uniform float uHasScene;
    uniform vec2 uResolution;
    uniform vec2 uNearFar;
    uniform vec3 uSunDirection;
    varying vec3 vWaterWorld;
    varying vec3 vWaterNormal;
    varying float vViewDepth;
    varying float vWaterAge;
    float hash(vec3 p) {
      p = fract(p * .3183099 + vec3(.13, .37, .71));
      p *= 17.0;
      return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
    }
    float noise(vec3 p) {
      vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
      return mix(mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x),
                     mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
                 mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
                     mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
    }
    float viewDepth(float z) {
      return uNearFar.x * uNearFar.y / max(.00001, uNearFar.y - z * (uNearFar.y - uNearFar.x));
    }
    void main() {
      vec3 flow = vWaterWorld * 2.5 - vec3(uWaterTime * 11.0, uWaterTime * 4.3, uWaterTime * 7.7);
      float broad = noise(flow);
      float fine = noise(flow * 2.8 + broad * 1.4);
      float aeration = smoothstep(.04, 1.35, vWaterAge);
      vec3 ripple = vec3(sin(flow.y * 2.1 + fine * 4.0), cos(flow.z * 2.4 + broad * 3.0), sin(flow.x * 1.7));
      vec3 N = normalize(vWaterNormal + ripple * .14);
      vec3 V = normalize(cameraPosition - vWaterWorld);
      float ndv = abs(dot(N, V));
      // Air/water IOR 1.333 gives a 2.04% normal-incidence reflection.
      float fresnel = .0204 + .9796 * pow(1.0 - ndv, 5.0);
      float sun = pow(max(0.0, dot(reflect(-uSunDirection, N), V)), 140.0);
      vec3 skyReflection = mix(vec3(.11,.17,.13), vec3(.70,.83,.94), smoothstep(-.1,.75,N.y));
      vec2 uv = gl_FragCoord.xy / uResolution;
      vec3 screenNormal = mat3(viewMatrix) * N;
      vec2 offset = screenNormal.xy * (.0015 + .0012 * broad) * (1.0 - fresnel * .5);
      vec2 refractedUV = clamp(uv + offset, vec2(.001), vec2(.999));
      float soft = 1.0;
      vec3 background = vec3(.39,.48,.45);
      if (uHasScene > .5) {
        float shiftedZ = viewDepth(texture2D(uSceneDepth, refractedUV).x);
        if (shiftedZ < vViewDepth + .05) refractedUV = uv;
        background = texture2D(uSceneColor, refractedUV).rgb;
        float opaqueZ = viewDepth(texture2D(uSceneDepth, uv).x);
        soft = smoothstep(0.0, .32, opaqueZ - vViewDepth);
      }
      vec3 color;
      float alpha;
      #if WATER_FOAM == 1
        // Broken, flowing white wisps replace the old solid white core.
        float wisps = smoothstep(.43, .78, broad * .70 + fine * .35);
        alpha = uOpacity * wisps * .65 * (1.0 - smoothstep(.12, 1.0, vWaterAge));
        color = vec3(.86,.93,.96) * (.86 + fine * .16) + sun * .8;
      #else
        vec3 transmitted = background * vec3(.975,.995,1.0);
        color = mix(transmitted, skyReflection, fresnel * .72);
        float froth = smoothstep(.48, .82, broad * .65 + fine * .5) * aeration;
        color = mix(color, vec3(.85,.93,.96), .08 + aeration * .22 + froth * .30);
        color += sun * (1.8 + fresnel * 2.0);
        alpha = uOpacity * (.58 + fresnel * .6 + froth * .6);
        #if WATER_DROPLET == 1
          alpha = uOpacity * (.55 + fresnel * .65);
          color = mix(color, skyReflection, .18) + sun * 2.0;
          color = mix(color, vec3(.84,.92,.96), aeration * .25);
        #else
          // Gradually tear the outer sheet into irregular water ribbons;
          // detached ballistic droplets carry the widening spray around it.
          float ribbons = smoothstep(.28, .64, broad * .65 + fine * .45);
          alpha *= mix(1.0, ribbons, smoothstep(.08, .65, vWaterAge))
            * (1.0 - smoothstep(.12, .8, vWaterAge));
        #endif
      #endif
      alpha = clamp(alpha * soft, 0.0, .85);
      if (alpha < .008) discard;
      gl_FragColor = vec4(color, alpha);
    }`;
  function material(kind: 'jet' | 'foam' | 'droplet', opacity: number) {
    return new THREE.ShaderMaterial({ vertexShader, fragmentShader,
      defines: { WATER_FOAM: kind === 'foam' ? 1 : 0, WATER_DROPLET: kind === 'droplet' ? 1 : 0 },
      uniforms: { ...shared, uOpacity: { value: opacity } },
      transparent: true, opacity, depthWrite: false, depthTest: true, toneMapped: false });
  }
  const head = material('droplet', .34), jet = material('jet', .46);
  const foam = material('foam', .27), spray = material('droplet', .62);
  return { head, jet, foam, spray,
    bind(depth: THREE.DepthTexture, width: number, height: number, color?: THREE.Texture) {
      shared.uSceneDepth.value = depth; shared.uSceneColor.value = color ?? null;
      shared.uResolution.value.set(width, height); shared.uHasScene.value = color ? 1 : 0;
    },
    dispose() { for (const value of [head, jet, foam, spray]) value.dispose(); },
  };
}
