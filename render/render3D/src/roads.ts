/** Shared visual road footprint in CoT world metres; Jackal collision remains flat. */
export const routes = [
  { across: true, offset: -28, phase: 0.5, bend: 5.0, width: 9.2 },
  { across: false, offset: -37, phase: 1.4, bend: 7.0, width: 8.6 },
  { across: false, offset: 68, phase: 3.1, bend: 5.0, width: 8.2 },
];

export function center(route: typeof routes[number], along: number) {
  return route.offset + Math.sin(along / 45 + route.phase) * route.bend;
}

export function distanceToRoad(x: number, z: number) {
  return Math.min(...routes.map(route => Math.abs((route.across ? z : x) -
    center(route, route.across ? x : z))));
}

export function onRoad(x: number, z: number, shoulder = 0) {
  return routes.some(route => Math.abs((route.across ? z : x) -
    center(route, route.across ? x : z)) < route.width / 2 + shoulder);
}

/** The road mesh rises above the flat Jackal ground and cuts two real troughs. */
export function roadRelief(across: number, width: number) {
  const radius = Math.abs(across);
  const feather = Math.max(0, Math.min(1, (width / 2 - radius) / 1.15));
  const crown = Math.exp(-Math.pow(radius / 1.22, 2)) * 0.065;
  const rut = Math.exp(-Math.pow((radius - 1.55) / 0.54, 2)) * 0.138;
  const shoulder = Math.exp(-Math.pow((radius - (width / 2 - 0.65)) / 0.56, 2)) * 0.047;
  return feather * (0.155 + crown - rut - shoulder);
}
