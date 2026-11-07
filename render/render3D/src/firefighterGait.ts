const TAU = Math.PI * 2;

export interface FirefighterGaitState {
  phase: number;
  blend: number;
  travelledM: number;
}

export interface FirefighterGaitSample {
  moving: boolean;
  locomotionMps: number;
  strideLengthM: number;
  cycleRateHz: number;
  stepRateHz: number;
  runBlend: number;
}

export interface FootTarget {
  z: number;
  lift: number;
  contact: boolean;
  cycle: number;
}

export interface LegPose {
  hip: number;
  knee: number;
  ankle: number;
  reachable: boolean;
  targetZ: number;
  targetDown: number;
}

const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value));
const smooth = (value: number) => {
  const x = clamp(value, 0, 1);
  return x * x * (3 - 2 * x);
};

export function createGaitState(): FirefighterGaitState {
  return { phase: 0, blend: 0, travelledM: 0 };
}

/**
 * Advance from actual linear/angular movement rather than wall-clock time.
 * One full phase is a left-to-left stride, so translation and foot cadence
 * cannot drift apart during acceleration or braking.
 */
export function advanceFirefighterGait(state: FirefighterGaitState,
  speedMps: number, angularDegS: number, dt: number): FirefighterGaitSample {
  const linear = Math.abs(speedMps);
  // A turn moves each planted foot around the pelvis even when forward speed
  // is nearly zero. The 0.24 m radius approximates the boot-to-centre offset.
  const turning = Math.abs(angularDegS) * Math.PI / 180 * .24;
  const locomotion = Math.max(linear, turning);
  const moving = locomotion > .055;
  const safeDt = clamp(dt, 0, .05);
  const response = moving ? 11 : 14;
  state.blend += ((moving ? 1 : 0) - state.blend) * (1 - Math.exp(-response * safeDt));
  if (!moving && state.blend < .002) state.blend = 0;

  // Slow walking uses short strides; the staged 3 m/s approach becomes a
  // believable jog with a roughly 1.7 m full stride and 3.5 steps per second.
  const strideLengthM = clamp(.68 + locomotion * .35, .68, 1.72);
  const cycleRateHz = moving ? locomotion / strideLengthM : 0;
  if (moving && safeDt > 0) {
    state.phase = (state.phase + TAU * cycleRateHz * safeDt) % TAU;
    state.travelledM += linear * safeDt;
  }
  return { moving, locomotionMps: locomotion, strideLengthM, cycleRateHz,
    stepRateHz: cycleRateHz * 2, runBlend: smooth((linear - 1.15) / 1.25) };
}

/**
 * A planted boot travels backward relative to the pelvis at exactly the
 * character's forward speed. The swing uses a smooth return and toe lift.
 */
export function sampleFootTarget(phase: number, sideOffset: number,
  sample: FirefighterGaitSample): FootTarget {
  const cycle = ((phase / TAU + sideOffset) % 1 + 1) % 1;
  const duty = .62 + (.38 - .62) * sample.runBlend;
  const travel = Math.min(.66, sample.strideLengthM * duty);
  if (cycle < duty) {
    const u = cycle / duty;
    return { z: travel * (.5 - u), lift: 0, contact: true, cycle };
  }
  const u = (cycle - duty) / (1 - duty);
  const eased = smooth(u);
  return { z: travel * (-.5 + eased),
    lift: Math.sin(Math.PI * u) * (.09 + .055 * sample.runBlend),
    contact: false, cycle };
}

/** Solve a sagittal two-bone leg. Angles are Three.js local X rotations. */
export function solveLegPose(targetZ: number, targetDown: number,
  upperLength = .45, lowerLength = .45): LegPose {
  const maximum = upperLength + lowerLength - .002;
  const minimum = Math.abs(upperLength - lowerLength) + .002;
  const requested = Math.hypot(targetZ, targetDown);
  const reachable = requested >= minimum && requested <= maximum;
  const distance = clamp(requested, minimum, maximum);
  const scale = requested > 1e-6 ? distance / requested : 1;
  const z = targetZ * scale, down = targetDown * scale;
  const cosKnee = clamp((distance * distance - upperLength * upperLength - lowerLength * lowerLength)
    / (2 * upperLength * lowerLength), -1, 1);
  // Negative mathematical knee angle chooses the human-like forward knee.
  const q2 = -Math.acos(cosKnee);
  const q1 = Math.atan2(z, down)
    - Math.atan2(lowerLength * Math.sin(q2), upperLength + lowerLength * Math.cos(q2));
  const hip = -q1;
  const knee = -q2;
  return { hip, knee, ankle: q1 + q2, reachable, targetZ: z, targetDown: down };
}
