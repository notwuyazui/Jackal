// Pure coordinate/parameter conversions, shared by the live view and tests.
export const radians = Math.PI / 180;
export function toWorld(x: number, y: number, width: number, height: number, scale: number) {
  return { x: (x - width / 2) * scale, z: (y - height / 2) * scale };
}
export function toYaw(degrees: number) { return Math.PI - degrees * radians; }
export function toHeading(yaw: number) { return ((Math.PI - yaw) / radians % 360 + 360) % 360; }
export function relativeTurretYaw(hull: number, turret: number) {
  return toYaw(turret) - toYaw(hull);
}
export function effectiveParameters(p: {
  max_speed: number; max_acceleration: number; min_acceleration: number;
  hull_deg_s: number; turret_deg_s: number; max_health: number; cooldown: number;
  shell_speed: number; shell_lifetime: number; shell_base_damage: number;
}, scale: number) {
  return {
    topSpeedKmh: p.max_speed * scale * 3.6,
    reverseSpeedKmh: p.max_speed * scale * 3.6,
    accelerationMps2: p.max_acceleration * scale,
    decelerationMps2: p.min_acceleration * scale,
    hullTraverseDegS: p.hull_deg_s, turretTraverseDegS: p.turret_deg_s,
    hp: p.max_health, reloadS: p.cooldown,
    shell: { type: 'HE', source: 'normal_shell', speedMps: p.shell_speed * scale,
      lifetimeS: p.shell_lifetime, damage: p.shell_base_damage,
      ricochet: false, penetrationTest: false, splashDamage: 0 },
  };
}
