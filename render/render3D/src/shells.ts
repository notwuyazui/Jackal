/** Water droplets/trails plus continuous brush fires; no combat effects. */
import * as THREE from 'three';
import { createBrushFires } from './brushFire.ts';
import { createWaterArc, sampleWaterArc, WATER_GRAVITY, WATER_ARC_RISE_M, type WaterArc, type WaterSample } from './waterTrajectory.ts';
import { createWaterMaterials, waterSpread } from './waterMaterial.ts';
import { LATE_FX_LAYER } from '@cot/fx/layers.ts';
import type { Frame } from './protocol.ts';

export async function createShellPresentation(scene: THREE.Scene,
  options: { near?: number; far?: number; sunDirection?: THREE.Vector3 } = {}) {
  const fires = await createBrushFires(scene);
  let sceneDepth: THREE.DepthTexture | null = null;
  const capacity = 2048;
  const trailSegments = 6;
  const sprayPerShell = 48;
  const sprayHalfAngleDeg = .9;
  const widthScale = 6.75;
  const random = (seed: number) => { const value = Math.sin(seed) * 43758.5453; return value - Math.floor(value); };
  const waterClock = { value: 0 };
  const headGeometry = new THREE.SphereGeometry(1, 16, 10);
  const sprayGeometry = new THREE.SphereGeometry(1, 10, 6);
  const trailGeometry = new THREE.CylinderGeometry(1, 1, 1, 28, 1, true);
  const headAge = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1);
  const trailAge = new THREE.InstancedBufferAttribute(new Float32Array(capacity * trailSegments), 1);
  const sprayAge = new THREE.InstancedBufferAttribute(new Float32Array(capacity * sprayPerShell), 1);
  headGeometry.setAttribute('aWaterAge', headAge); trailGeometry.setAttribute('aWaterAge', trailAge);
  sprayGeometry.setAttribute('aWaterAge', sprayAge);
  for (const attribute of [headAge, trailAge, sprayAge]) attribute.setUsage(THREE.DynamicDrawUsage);
  const materials = createWaterMaterials(waterClock, options);
  const { head: headMaterial, jet: trailMaterial, foam: coreMaterial, spray: sprayMaterial } = materials;
  const heads = new THREE.InstancedMesh(headGeometry, headMaterial, capacity);
  const trails = new THREE.InstancedMesh(trailGeometry, trailMaterial, capacity * trailSegments);
  const cores = new THREE.InstancedMesh(trailGeometry, coreMaterial, capacity * trailSegments);
  const spray = new THREE.InstancedMesh(sprayGeometry, sprayMaterial, capacity * sprayPerShell);
  const water = new THREE.Group(); water.name = 'Translucent broad water jet / separated three-second packets';
  water.add(heads, trails, cores, spray); scene.add(water);
  for (const mesh of [heads, trails, cores, spray]) {
    mesh.count = 0; mesh.frustumCulled = false;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.layers.set(LATE_FX_LAYER); mesh.renderOrder = 25;
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
    bindSceneDepth(depth: THREE.DepthTexture, width: number, height: number, color?: THREE.Texture) {
      sceneDepth = depth; fires.bindDepth(depth, width, height); materials.bind(depth, width, height, color);
    },
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
        dummy.position.copy(position); dummy.scale.set(0.0075 * widthScale * ripple, 0.12, 0.0075 * widthScale * ripple);
        dummy.updateMatrix(); heads.setMatrixAt(index, dummy.matrix);
        headAge.setX(index, age);
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
          trailAge.setX(trailIndex, tailAge + (age - tailAge) * (segment - .5) / trailSegments);
          tailStart.copy(tailEnd); trailIndex++;
        }
        // Fine detached droplets at the edge give the jet an aerated, wet
        // silhouette instead of a stack of large projectile-shaped beads.
        // Give each drop a small, fixed radial launch velocity. Its separation
        // grows with flight time while it retains the central arc's gravity.
        direction.set(arc.vx, arc.launchVy, arc.vz).normalize();
        sideways.set(-arc.vz, 0, arc.vx).normalize();
        crossStream.crossVectors(direction, sideways).normalize();
        const radialSpeed = Math.hypot(arc.vx, arc.launchVy, arc.vz) * Math.tan(THREE.MathUtils.degToRad(sprayHalfAngleDeg));
        for (let drop = 0; drop < sprayPerShell; drop++) {
          const seed = bullet.id * 127.1 + drop * 311.7;
          const phase = random(seed) * Math.PI * 2;
          const dropAge = Math.max(0, age - .105 * (drop + random(seed + 73.7)) / sprayPerShell);
          sampleWaterArc(arc, dropAge, sample);
          const radialFraction = Math.sqrt(random(seed + 19.1));
          const sideSpeed = Math.cos(phase) * radialSpeed * radialFraction;
          const crossSpeed = Math.sin(phase) * radialSpeed * radialFraction;
          const radius = .025 * waterSpread(dropAge) + radialSpeed * dropAge;
          dummy.position.set(sample.x, sample.y, sample.z)
            .addScaledVector(sideways, Math.cos(phase) * radius * radialFraction)
            .addScaledVector(crossStream, Math.sin(phase) * radius * radialFraction);
          direction.set(arc.vx, sample.vy, arc.vz).addScaledVector(sideways, sideSpeed)
            .addScaledVector(crossStream, crossSpeed).normalize();
          dummy.quaternion.setFromUnitVectors(up, direction);
          const size = widthScale * (.0025 + .0018 * (1 + Math.sin(phase * 3)));
          dummy.scale.set(size, size * 4.0, size); dummy.updateMatrix();
          sprayAge.setX(sprayIndex, dropAge); spray.setMatrixAt(sprayIndex++, dummy.matrix);
        }
        live.push({ id: bullet.id, x: bullet.position[0], y: bullet.position[1], height_m: position.y,
          tail_height_m: tailHeight, vertical_speed_mps: headVy, tracer: 'parabolic-blue-water' });
        index++;
      }
      for (const id of arcs.keys()) if (!seen.has(id)) arcs.delete(id);
      heads.count = index; trails.count = cores.count = trailIndex; spray.count = sprayIndex;
      for (const mesh of [heads, trails, cores, spray]) mesh.instanceMatrix.needsUpdate = true;
      for (const attribute of [headAge, trailAge, sprayAge]) attribute.needsUpdate = true;
    },
    render() {},
    effectState() { return { flash_particles: 0, ...fires.state(), trajectory: 'parabolic', water_max_gravity_mps2: WATER_GRAVITY,
      water_gravity_mode: 'range-adjusted', water_arc_height_above_chord_m: WATER_ARC_RISE_M,
      water_width_scale: widthScale, landing_mode: 'continuous-fire-front',
      water_style: 'refractive-turbulent-water-with-aerated-spray', water_opacity: trailMaterial.opacity,
      water_refraction: true, water_ior: 1.333,
      water_spread_mode: 'progressive-cone-with-ribbon-breakup', water_spray_half_angle_deg: sprayHalfAngleDeg,
      water_spray_droplets: spray.count,
      soft_depth_bound: sceneDepth instanceof THREE.DepthTexture }; },
    receipts(_manifest: Frame['manifest']) { return live; },
    dispose() {
      fires.dispose(); water.removeFromParent();
      for (const mesh of [heads, trails, cores, spray]) mesh.dispose();
      headGeometry.dispose(); trailGeometry.dispose(); sprayGeometry.dispose();
      materials.dispose(); arcs.clear();
    },
  };
}
