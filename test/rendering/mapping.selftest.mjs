import assert from 'node:assert/strict';
import * as THREE from '../../render/render3D/node_modules/three/build/three.module.js';
import { toWorld, toYaw, toHeading, relativeTurretYaw, effectiveParameters } from '../../render/render3D/src/mapping.ts';
import { createColliders } from '../../render/render3D/src/colliders.ts';

const close = (a, b, epsilon = 1e-9) => assert.ok(Math.abs(a - b) < epsilon, String(a) + ' != ' + b);
const angleError = (a, b) => ((a - b + 540) % 360) - 180;
assert.deepEqual(toWorld(220, 320, 960, 640, 0.25), { x: -65, z: 0 });
assert.deepEqual(toWorld(0, 0, 960, 640, 0.25), { x: -120, z: -80 });
assert.deepEqual(toWorld(960, 640, 960, 640, 0.25), { x: 120, z: 80 });
for (let angle = 0; angle < 360; angle += 1.25) {
  close(angleError(toHeading(toYaw(angle)), angle), 0);
  const direction = new THREE.Vector3(0, 0, 1).applyAxisAngle(new THREE.Vector3(0, 1, 0), toYaw(angle));
  // Jackal's 0 degrees points up the screen; increasing degrees turns right.
  close(direction.x, Math.sin(angle * Math.PI / 180));
  close(direction.z, -Math.cos(angle * Math.PI / 180));
}
for (let hull = 0; hull < 360; hull += 37) {
  for (let turret = 0; turret < 360; turret += 17) {
    const parent = new THREE.Group(), child = new THREE.Group();
    parent.rotation.y = toYaw(hull);
    parent.scale.setScalar(0.8);
    child.rotation.y = relativeTurretYaw(hull, turret);
    parent.add(child); parent.updateMatrixWorld(true);
    const bearing = new THREE.Vector3(0, 0, 1).transformDirection(child.matrixWorld);
    close(angleError(toHeading(Math.atan2(bearing.x, bearing.z)), turret), 0);
  }
}
const parameters = effectiveParameters({
  max_speed: 50, max_acceleration: 100, min_acceleration: -100, hull_deg_s: 100,
  turret_deg_s: 150, max_health: 100, cooldown: 1.5,
  shell_speed: 300, shell_lifetime: 1.2, shell_base_damage: 8,
}, 0.25);
assert.equal(parameters.topSpeedKmh, 45);
assert.equal(parameters.accelerationMps2, 25);
assert.equal(parameters.decelerationMps2, -25);
assert.equal(parameters.turretTraverseDegS, 150);
assert.equal(parameters.reloadS, 1.5);
assert.equal(parameters.shell.speedMps, 75);
assert.equal(parameters.shell.speedMps * parameters.shell.lifetimeS, 90);
assert.equal(parameters.shell.ricochet, false);
assert.equal(parameters.shell.splashDamage, 0);
const colliders = createColliders({ meters_per_pixel: 0.25, map: { size: [960, 640] } }, 3);
// Pygame's odd-height rectangle is centered at y=320.5, regardless of the
// tank's floating-point position or hull bearing. Movement bounds can differ.
const unit = { position: [220.2, 320.1], heading: 0,
  hitbox: [212, 309, 16, 23], collision_box: [211, 310, 18, 20] };
for (const heading of [0, 37, 90, 179, 270]) {
  const receipt = colliders.pose({ ...unit, heading });
  assert.deepEqual(receipt.hitbox, unit.hitbox);
  assert.deepEqual(receipt.collision_box, unit.collision_box);
  assert.deepEqual(receipt.collider_y, [0, 3]);
}
colliders.dispose();
console.log('PASS: coordinate axes, complete yaw cycle, uniformly scaled turret bearings, Jackal parameters');
