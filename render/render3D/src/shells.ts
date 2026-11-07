/** Water droplets/trails plus continuous brush fires; no combat effects. */
import * as THREE from 'three';
import { createBrushFires } from './brushFire.ts';
import { createWaterArc, sampleWaterArc, WATER_GRAVITY, type WaterArc, type WaterSample } from './waterTrajectory.ts';
import type { Frame } from './protocol.ts';

export async function createShellPresentation(scene: THREE.Scene) {
  const fires = await createBrushFires(scene);
  let sceneDepth: THREE.DepthTexture | null = null;
  const capacity = 2048;
  const trailSegments = 3;
  const widthScale = 9.0;
  const waterClock = { value: 0 };
  const headGeometry = new THREE.SphereGeometry(1, 10, 6);
  const trailGeometry = new THREE.CylinderGeometry(1, 1, 1, 10, 1, true);
  const headMaterial = new THREE.MeshPhysicalMaterial({ color: 0x9cdef3,
    metalness: 0, roughness: 0.09, clearcoat: .9, ior: 1.333,
    transparent: true, opacity: 0.3, depthWrite: false });
  const trailMaterial = new THREE.MeshPhysicalMaterial({ color: 0x78cbe7,
    metalness: 0, roughness: 0.1, clearcoat: .9, ior: 1.333, specularIntensity: .65,
    transparent: true, opacity: 0.4, depthWrite: false });
  const coreMaterial = new THREE.MeshBasicMaterial({ color: 0xd2f3ff,
    transparent: true, opacity: 0.3, depthWrite: false, toneMapped: false });
  const sprayMaterial = new THREE.MeshPhysicalMaterial({ color: 0xc9efff,
    metalness: 0, roughness: .12, clearcoat: .8, ior: 1.333,
    transparent: true, opacity: .42, depthWrite: false });
  for (const material of [headMaterial, trailMaterial, coreMaterial]) {
    material.onBeforeCompile = shader => {
      shader.uniforms.uWaterTime = waterClock;
      shader.vertexShader = 'varying vec3 vWaterWorld;\n' + shader.vertexShader;
      shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', `
        #include <begin_vertex>
        vec4 jetPoint = vec4(transformed, 1.0);
        #ifdef USE_INSTANCING
          jetPoint = instanceMatrix * jetPoint;
        #endif
        vWaterWorld = (modelMatrix * jetPoint).xyz;`);
      shader.fragmentShader = 'varying vec3 vWaterWorld; uniform float uWaterTime;\n' + shader.fragmentShader;
      shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>', `
        #include <color_fragment>
        float ripple = 0.5 + 0.5 * sin(dot(vWaterWorld, vec3(4.0, 3.5, 2.7)) - uWaterTime * 19.0);
        float fine = 0.5 + 0.5 * sin(dot(vWaterWorld, vec3(11.0, -8.0, 7.0)) - uWaterTime * 31.0);
        diffuseColor.rgb *= 0.87 + ripple * 0.16 + fine * 0.05;
        diffuseColor.a *= 0.94 + fine * 0.06;`);
    };
    material.customProgramCacheKey = () => 'flowing-aerated-water-v1';
  }
  const heads = new THREE.InstancedMesh(headGeometry, headMaterial, capacity);
  const trails = new THREE.InstancedMesh(trailGeometry, trailMaterial, capacity * trailSegments);
  const cores = new THREE.InstancedMesh(trailGeometry, coreMaterial, capacity * trailSegments);
  const spray = new THREE.InstancedMesh(headGeometry, sprayMaterial, capacity * 2);
  const water = new THREE.Group(); water.name = 'Translucent broad water jet / separated three-second packets';
  water.add(heads, trails, cores, spray); scene.add(water);
  for (const mesh of [heads, trails, cores, spray]) {
    mesh.count = 0; mesh.frustumCulled = false;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  }
  const dummy = new THREE.Object3D(), up = new THREE.Vector3(0, 1, 0);
  const direction = new THREE.Vector3(), position = new THREE.Vector3();
  const tailStart = new THREE.Vector3(), tailEnd = new THREE.Vector3();
  const sideways = new THREE.Vector3(), crossStream = new THREE.Vector3();
  const sample: WaterSample = { x: 0, y: 0, z: 0, vy: 0 };
  const arcs = new Map<number, WaterArc>();
  const live: Array<{ id: number; x: number; y: number; height_m: number;
    tail_height_m: number; vertical_speed_mps: number; tracer: string }> = [];

  return {
    bindSceneDepth(depth: THREE.DepthTexture, width: number, height: number) { sceneDepth = depth; fires.bindDepth(depth, width, height); },
    pinTime(value: number) { fires.setTime(value); waterClock.value = value; },
    reset(_seed: number) {
      fires.reset(); arcs.clear(); live.length = 0;
      heads.count = trails.count = cores.count = spray.count = 0;
    },
    sync(next: Frame, planeHeight: number) {
      fires.sync(next); live.length = 0;
      const seen = new Set<number>();
      let index = 0, trailIndex = 0, sprayIndex = 0;
      for (const bullet of next.bullets) {
        if (index >= capacity) break;
        let arc = arcs.get(bullet.id);
        if (!arc) {
          arc = createWaterArc(bullet, next, planeHeight);
          arcs.set(bullet.id, arc);
        }
        seen.add(bullet.id);
        const age = Math.max(0, arc.lifetime - bullet.lifetime);
        sampleWaterArc(arc, age, sample);
        // Missed jets fall into the ground instead of continuing beneath it.
        if (sample.y <= 0.08) continue;
        position.set(sample.x, sample.y, sample.z);
        const headVy = sample.vy;
        direction.set(arc.vx, headVy, arc.vz).normalize();
        dummy.quaternion.setFromUnitVectors(up, direction);
        // Shallow, irregular beads blend into the translucent trailing volume.
        const ripple = .94 + .06 * Math.sin(bullet.id * 2.4);
        dummy.position.copy(position); dummy.scale.set(0.07 * widthScale * ripple, 0.24, 0.07 * widthScale * ripple);
        dummy.updateMatrix(); heads.setMatrixAt(index, dummy.matrix);
        // Match the 0.1 s emission interval: a small overlap joins the fluid,
        // without the broad alternating opaque bands of the previous trails.
        const tailAge = Math.max(0, age - 0.105);
        sampleWaterArc(arc, tailAge, sample);
        const tailHeight = sample.y;
        tailStart.set(sample.x, sample.y, sample.z);
        for (let segment = 1; segment <= trailSegments; segment++) {
          sampleWaterArc(arc, tailAge + (age - tailAge) * segment / trailSegments, sample);
          tailEnd.set(sample.x, sample.y, sample.z);
          const length = tailStart.distanceTo(tailEnd);
          direction.subVectors(tailEnd, tailStart).normalize();
          dummy.quaternion.setFromUnitVectors(up, direction);
          dummy.position.copy(tailStart).add(tailEnd).multiplyScalar(0.5);
          dummy.scale.set(0.105 * widthScale, length, 0.105 * widthScale); dummy.updateMatrix(); trails.setMatrixAt(trailIndex, dummy.matrix);
          dummy.scale.set(0.038 * widthScale, length, 0.038 * widthScale); dummy.updateMatrix(); cores.setMatrixAt(trailIndex, dummy.matrix);
          tailStart.copy(tailEnd); trailIndex++;
        }
        // Fine detached droplets at the edge give the jet an aerated, wet
        // silhouette instead of a stack of large projectile-shaped beads.
        direction.set(arc.vx, headVy, arc.vz).normalize();
        sideways.set(-arc.vz, 0, arc.vx).normalize();
        crossStream.crossVectors(direction, sideways).normalize();
        for (let drop = 0; drop < 2; drop++) {
          const phase = bullet.id * 2.39996 + drop * 2.8;
          const radius = widthScale * (.115 + .035 * Math.sin(phase + age * 5));
          dummy.position.copy(position).addScaledVector(sideways, Math.cos(phase) * radius)
            .addScaledVector(crossStream, Math.sin(phase) * radius);
          dummy.quaternion.setFromUnitVectors(up, direction);
          const size = widthScale * (.012 + .005 * (1 + Math.sin(phase * 3)));
          dummy.scale.set(size, size * 1.8, size); dummy.updateMatrix();
          spray.setMatrixAt(sprayIndex++, dummy.matrix);
        }
        live.push({ id: bullet.id, x: bullet.position[0], y: bullet.position[1], height_m: position.y,
          tail_height_m: tailHeight, vertical_speed_mps: headVy, tracer: 'parabolic-blue-water' });
        index++;
      }
      for (const id of arcs.keys()) if (!seen.has(id)) arcs.delete(id);
      heads.count = index; trails.count = cores.count = trailIndex; spray.count = sprayIndex;
      for (const mesh of [heads, trails, cores, spray]) mesh.instanceMatrix.needsUpdate = true;
    },
    render() {},
    effectState() { return { flash_particles: 0, ...fires.state(), trajectory: 'parabolic', water_max_gravity_mps2: WATER_GRAVITY,
      water_gravity_mode: 'range-adjusted', water_arc_height_above_chord_m: 18,
      water_width_scale: widthScale, landing_mode: 'continuous-fire-front',
      water_style: 'flowing-water-with-fine-spray', water_opacity: trailMaterial.opacity,
      water_spray_droplets: spray.count,
      soft_depth_bound: sceneDepth instanceof THREE.DepthTexture }; },
    receipts(_manifest: Frame['manifest']) { return live; },
    dispose() {
      fires.dispose(); water.removeFromParent();
      for (const mesh of [heads, trails, cores, spray]) mesh.dispose();
      headGeometry.dispose(); trailGeometry.dispose();
      headMaterial.dispose(); trailMaterial.dispose(); coreMaterial.dispose(); sprayMaterial.dispose(); arcs.clear();
    },
  };
}
