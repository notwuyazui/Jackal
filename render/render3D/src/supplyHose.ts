/** A charged hose paid out from a fixed supply point behind the crew. */
import * as THREE from 'three';
import { center, roadRelief, routes } from './roads.ts';

const RADIUS = .047;
const SIDES = 10;
const REAR_LENGTH = 34;

export function hoseGroundHeight(x: number, z: number) {
  return Math.max(0, ...routes.map(route => roadRelief(
    (route.across ? z : x) - center(route, route.across ? x : z), route.width)));
}

function hoseTube(name: string, segments: number, material: THREE.Material) {
  const geometry = new THREE.BufferGeometry();
  const position = new THREE.BufferAttribute(new Float32Array((segments + 1) * (SIDES + 1) * 3), 3);
  const normal = new THREE.BufferAttribute(new Float32Array(position.count * 3), 3);
  position.setUsage(THREE.DynamicDrawUsage); normal.setUsage(THREE.DynamicDrawUsage);
  const uv = new Float32Array(position.count * 2), indices: number[] = [];
  for (let ring = 0; ring <= segments; ring++) for (let side = 0; side <= SIDES; side++) {
    const vertex = ring * (SIDES + 1) + side;
    uv[vertex * 2] = ring / segments * 24; uv[vertex * 2 + 1] = side / SIDES;
    if (ring < segments && side < SIDES) {
      const next = vertex + SIDES + 1;
      indices.push(vertex, vertex + 1, next, vertex + 1, next + 1, next);
    }
  }
  geometry.setAttribute('position', position); geometry.setAttribute('normal', normal);
  geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2)); geometry.setIndex(indices);
  const mesh = new THREE.Mesh(geometry, material); mesh.name = name;
  mesh.castShadow = true; mesh.receiveShadow = true; mesh.frustumCulled = false;
  return { mesh, geometry, position, normal, segments,
    centers: Array.from({ length: segments + 1 }, () => new THREE.Vector3()) };
}

export function createSupplyHose(root: THREE.Group, material: THREE.Material,
  groundAt: (x: number, z: number) => number = hoseGroundHeight) {
  const group = new THREE.Group(); group.name = 'ground-laid trailing supply hose'; root.add(group);
  const ground = hoseTube('hose laid out behind the firefighter', 192, material);
  ground.mesh.matrixAutoUpdate = false;
  const lifted = hoseTube('hose from the ground to the two-handed nozzle', 48, material);
  group.add(ground.mesh, lifted.mesh);
  const feed = new THREE.Vector3(), drop = new THREE.Vector3(), previousDrop = new THREE.Vector3();
  const groundedDrop = new THREE.Vector3(), lastNozzle = new THREE.Vector3(Infinity, 0, 0);
  const lastTrailRoot = new THREE.Vector3(), history: THREE.Vector3[] = [];
  const forward = new THREE.Vector3(), right = new THREE.Vector3();
  const inverse = new THREE.Matrix4(), centerPoint = new THREE.Vector3();
  const lastGroundPose = new THREE.Matrix4();
  const tangent = new THREE.Vector3(), sideAxis = new THREE.Vector3(), vertical = new THREE.Vector3();
  const radial = new THREE.Vector3(), vertex = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
  let initialized = false, length = 0, groundLength = 0, clock = 0;
  let lastGroundUpdate = -Infinity, lastYaw = Infinity;

  function seat(point: THREE.Vector3) {
    point.y = groundAt(point.x, point.z) + RADIUS + .012;
    return point;
  }

  function updateTube(tube: ReturnType<typeof hoseTube>, curve: THREE.CatmullRomCurve3, onGround: boolean) {
    curve.updateArcLengths();
    for (let ring = 0; ring <= tube.segments; ring++) {
      curve.getPointAt(ring / tube.segments, tube.centers[ring]);
      const point = tube.centers[ring];
      const floor = groundAt(point.x, point.z) + RADIUS + .012;
      point.y = onGround ? floor : Math.max(point.y, floor);
    }
    for (let ring = 0; ring <= tube.segments; ring++) {
      centerPoint.copy(tube.centers[ring]);
      tangent.subVectors(tube.centers[Math.min(tube.segments, ring + 1)],
        tube.centers[Math.max(0, ring - 1)]).normalize();
      sideAxis.crossVectors(up, tangent);
      if (sideAxis.lengthSq() < 1e-6) sideAxis.copy(right);
      sideAxis.normalize(); vertical.crossVectors(tangent, sideAxis).normalize();
      for (let side = 0; side <= SIDES; side++) {
        const angle = side / SIDES * Math.PI * 2;
        radial.copy(sideAxis).multiplyScalar(Math.cos(angle)).addScaledVector(vertical, Math.sin(angle));
        vertex.copy(centerPoint).addScaledVector(radial, RADIUS).applyMatrix4(inverse);
        const index = ring * (SIDES + 1) + side;
        tube.position.setXYZ(index, vertex.x, vertex.y, vertex.z);
        radial.transformDirection(inverse); tube.normal.setXYZ(index, radial.x, radial.y, radial.z);
      }
    }
    tube.position.needsUpdate = tube.normal.needsUpdate = true;
  }

  return {
    group,
    update(nozzleRearWorld: THREE.Vector3, bodyYaw: number, time = clock + 1 / 60) {
      clock = time;
      root.updateMatrixWorld(true); inverse.copy(root.matrixWorld).invert();
      forward.set(Math.sin(bodyYaw), 0, Math.cos(bodyYaw));
      right.set(Math.cos(bodyYaw), 0, -Math.sin(bodyYaw));
      drop.copy(root.position).addScaledVector(forward, -.85).addScaledVector(right, .55); seat(drop);
      if (!initialized) {
        initialized = true;
        feed.copy(root.position).addScaledVector(forward, -REAR_LENGTH).addScaledVector(right, .55); seat(feed);
        for (const [distance, bend] of [[0, 0], [8, .65], [17, -.7], [25, .45], [30, -.25]]) {
          history.push(seat(feed.clone().addScaledVector(forward, distance).addScaledVector(right, bend)));
        }
        previousDrop.copy(drop); lastTrailRoot.copy(root.position);
      }
      if (root.position.distanceTo(lastTrailRoot) >= .65) {
        history.push(previousDrop.clone()); lastTrailRoot.copy(root.position);
        // Preserve the supply end when a long recording needs fewer old knots.
        if (history.length > 256) history.splice(5, 1);
      }
      previousDrop.copy(drop);
      // A long hose on the ground need not be rebuilt at display frequency.
      // Its lifted section stays attached to the last laid-down ground point.
      const groundChanged = lastGroundUpdate === -Infinity ||
        (time - lastGroundUpdate >= .05 && drop.distanceToSquared(groundedDrop) > .0001);
      if (groundChanged) {
        const groundCurve = new THREE.CatmullRomCurve3([...history, drop.clone()], false, 'centripetal');
        updateTube(ground, groundCurve, true);
        lastGroundPose.copy(root.matrixWorld);
        groundedDrop.copy(drop); groundLength = groundCurve.getLength(); lastGroundUpdate = time;
      }
      // Cached ground vertices belong to the last update's body transform.
      // Cancel subsequent body motion instead of dragging that entire hose
      // segment between its lower-frequency geometry updates.
      ground.mesh.matrix.multiplyMatrices(inverse, lastGroundPose);
      ground.mesh.matrixWorldNeedsUpdate = true;
      if (!groundChanged && nozzleRearWorld.distanceToSquared(lastNozzle) < 1e-8 && Math.abs(bodyYaw - lastYaw) < 1e-5) return;
      lastNozzle.copy(nozzleRearWorld); lastYaw = bodyYaw;
      const hipSide = root.position.clone().addScaledVector(right, .48).addScaledVector(forward, -.12);
      hipSide.y = .70;
      const lowered = groundedDrop.clone().lerp(hipSide, .42); lowered.y = .25;
      const handLead = nozzleRearWorld.clone().lerp(hipSide, .35);
      const raisedCurve = new THREE.CatmullRomCurve3([groundedDrop.clone(), lowered, hipSide, handLead, nozzleRearWorld.clone()],
        false, 'centripetal');
      updateTube(lifted, raisedCurve, false);
      length = groundLength + raisedCurve.getLength();
    },
    state() {
      return { feed: feed.clone(), length, radius: RADIUS, trailKnots: history.length,
        groundCenters: ground.centers, liftedCenters: lifted.centers };
    },
  };
}
