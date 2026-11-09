import assert from 'node:assert/strict';
import { createFirefighter } from '../../render/render3D/src/firefighter.ts';
import * as THREE from '../../render/render3D/node_modules/three/build/three.module.js';
import { FIREFIGHTER_MODEL_SCALE } from '../../render/render3D/src/firefighterDimensions.ts';

const parameters = { id: 2, size: [32, 46], max_speed: 50, max_acceleration: 100,
  min_acceleration: -100, hull_deg_s: 100, turret_deg_s: 75, max_health: 100,
  cooldown: .1, shell_speed: 135, shell_lifetime: 18, shell_base_damage: 1 };
const manifest = { units: [parameters], meters_per_pixel: .25, map: { size: [960, 640] } };
const model = createFirefighter(parameters, 'PLAYER', manifest);
let x = 480, y = 320, previous = null;
let plantedPairs = 0, maxSlide = 0, maxIK = 0, maxTwist = 0, minSole = Infinity;
let movingKneeMin = Infinity, movingKneeMax = -Infinity;
let spreadSamples = 0, forwardSamples = 0;
const liftoffs = [];
const knee = model.root.getObjectByName('articulated knee');
for (let tick = 0; tick < 1200; tick++) {
  const t = tick / 60;
  let vx = 0, vz = 0;
  if (t >= 1 && t < 5) vz = 2.4;
  else if (t >= 5 && t < 6) vz = 2.4 * (6 - t);
  else if (t >= 9 && t < 12) vx = 1.2;
  else if (t >= 13 && t < 16) vz = -1.2;
  x += vx / 60 / .25; y += vz / 60 / .25;
  const aim = t < 7 ? 180 : t < 8 ? 180 + (t - 7) * 60 : 240;
  const receipt = model.pose({ id: 2, team: 'PLAYER', position: [x, y], heading: 180,
    turret_heading: aim, speed: Math.hypot(vx, vz) / .25, angular_speed: 0,
    alive: true, health: 100, max_health: 100 }, t);
  maxIK = Math.max(maxIK, receipt.foot_ik_error_m);
  maxTwist = Math.max(maxTwist, Math.abs(receipt.torso_twist_deg));
  for (const side of ['left', 'right']) {
    const foot = receipt[`${side}_foot_world`];
    minSole = Math.min(minSole, foot[1] - .12932 * FIREFIGHTER_MODEL_SCALE);
    if (previous?.[`${side}_foot_contact`] && receipt[`${side}_foot_contact`]) {
      plantedPairs++;
      maxSlide = Math.max(maxSlide, Math.hypot(...foot.map((v, i) => v - previous[`${side}_foot_world`][i])));
    }
    if (previous?.[`${side}_foot_contact`] && !receipt[`${side}_foot_contact`]) liftoffs.push(side);
  }
  if (t > 2 && t < 5) {
    forwardSamples++;
    spreadSamples += Math.abs(receipt.left_foot_world[2] - receipt.right_foot_world[2]) > .3;
  }
  if (Math.hypot(vx, vz) > .5) {
    movingKneeMin = Math.min(movingKneeMin, knee.rotation.x);
    movingKneeMax = Math.max(movingKneeMax, knee.rotation.x);
  }
  previous = receipt;
}
console.log({ plantedPairs, maxSlide, maxIK, maxTwist, minSole,
  kneeRange: movingKneeMax - movingKneeMin });
assert.ok(plantedPairs > 300, 'walk, stop, turn and sidestep all contain grounded support');
assert.ok(maxSlide < .012, 'a planted boot must not slide in world coordinates');
assert.ok(maxIK < .015, 'feet stay within anatomical leg reach');
assert.ok(minSole > -.006, 'boots do not penetrate the ground');
assert.ok(movingKneeMax - movingKneeMin > .3, 'moving knees articulate');
assert.ok(maxTwist < 80, 'torso does not twist through an impossible tank-turret angle');
assert.ok(liftoffs.length > 20 && liftoffs.every((side, i) => i === 0 || side !== liftoffs[i - 1]),
  'left/right steps alternate rather than synchronizing into a bunny hop');
assert.ok(spreadSamples / forwardSamples > .25, 'running feet separate fore/aft through the stride');
assert.equal(previous.locomotion_mode, 'planted-idle');
assert.ok(previous.left_foot_contact && previous.right_foot_contact, 'both boots settle after stopping');
const standingBounds = new THREE.Box3().setFromObject(model.body);
assert.ok(Math.abs(standingBounds.max.y - standingBounds.min.y - 1.8) < .005,
  'actual standing geometry from boot soles to helmet crown is 1.8 metres');
model.dispose();
console.log('PASS: world-planted feet, leg reach, soles, acceleration/braking, turning, side/back steps and idle');
