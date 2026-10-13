import * as THREE from 'three';
import { toWorld } from './mapping.ts';
import type { Manifest, UnitFrame } from './protocol.ts';

/** Extrude the actual Pygame rectangles. Jackal does not rotate these boxes. */
export function createColliders(manifest: Manifest, height: number, debug = false) {
  const group = new THREE.Group();
  group.name = 'Jackal axis-aligned hit / movement cuboids';
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const hitMaterial = new THREE.MeshBasicMaterial({ color: '#ffd080', wireframe: true });
  const movementMaterial = new THREE.MeshBasicMaterial({ color: '#7fe5ff', wireframe: true });
  const hit = new THREE.Mesh(geometry, hitMaterial);
  const movement = new THREE.Mesh(geometry, movementMaterial);
  hit.name = 'damage hitbox';
  movement.name = 'movement collision box';
  hit.visible = movement.visible = debug;
  group.add(hit, movement);
  const bounds = new THREE.Box3();
  const scale = manifest.meters_per_pixel;
  const [width, mapHeight] = manifest.map.size;

  function update(mesh: THREE.Mesh, rect: number[]) {
    const [left, top, w, h] = rect;
    const center = toWorld(left + w / 2, top + h / 2, width, mapHeight, scale);
    mesh.position.set(center.x, height / 2, center.z);
    mesh.scale.set(w * scale, height, h * scale);
    mesh.updateMatrixWorld(true);
    bounds.setFromObject(mesh);
    return [bounds.min.x / scale + width / 2, bounds.min.z / scale + mapHeight / 2,
      (bounds.max.x - bounds.min.x) / scale, (bounds.max.z - bounds.min.z) / scale];
  }

  return {
    group,
    pose(unit: UnitFrame) {
      return { hitbox: update(hit, unit.hitbox), collision_box: update(movement, unit.collision_box),
        collider_y: [bounds.min.y, bounds.max.y] };
    },
    dispose() {
      group.removeFromParent(); geometry.dispose(); hitMaterial.dispose(); movementMaterial.dispose();
    },
  };
}
