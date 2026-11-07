import assert from 'node:assert/strict';
import { advanceFirefighterGait, createGaitState, sampleFootTarget, solveLegPose }
  from '../../render/render3D/src/firefighterGait.ts';

const close = (actual, expected, epsilon = 1e-7) =>
  assert.ok(Math.abs(actual - expected) < epsilon, `${actual} != ${expected}`);

const idle = createGaitState();
for (let i = 0; i < 120; i++) advanceFirefighterGait(idle, 0, 0, 1 / 60);
close(idle.phase, 0);
close(idle.blend, 0);

const moving = createGaitState();
let sample;
for (let i = 0; i < 120; i++) sample = advanceFirefighterGait(moving, 3, 0, 1 / 60);
assert.ok(moving.blend > .999, 'moving character fully enters the gait');
assert.ok(sample.stepRateHz > 3.3 && sample.stepRateHz < 3.7, '3 m/s is a plausible jog cadence');
close(moving.travelledM, 6, 1e-6);

// During the planted part of the cycle, local boot travel cancels root travel.
const duty = .38;
const phaseA = Math.PI * 2 * .08;
const phaseB = Math.PI * 2 * .20;
const footA = sampleFootTarget(phaseA, 0, sample);
const footB = sampleFootTarget(phaseB, 0, sample);
assert.ok(footA.contact && footB.contact);
const rootTravel = sample.strideLengthM * (.20 - .08);
close(footB.z - footA.z, -rootTravel);

// Forward kinematics must reconstruct the requested planted-boot target.
const pose = solveLegPose(footA.z, .82, .45, .45);
assert.ok(pose.reachable, 'normal stance stays within human leg reach');
const q1 = -pose.hip, q2 = -pose.knee;
const z = .45 * Math.sin(q1) + .45 * Math.sin(q1 + q2);
const down = .45 * Math.cos(q1) + .45 * Math.cos(q1 + q2);
close(z, pose.targetZ); close(down, pose.targetDown);

// Turning in place also steps instead of rotating on rigid, motionless legs.
const turning = createGaitState();
for (let i = 0; i < 60; i++) sample = advanceFirefighterGait(turning, 0, 35, 1 / 60);
assert.ok(turning.phase > 0 && turning.blend > .99 && sample.stepRateHz > 0);

console.log('PASS: distance-linked gait, planted-foot cancellation, leg IK and turning steps');
