import * as THREE from 'three';

/** Keep every tree transform/color while making off-camera woods cullable. */
export function partitionForest(forest: THREE.Object3D, cellSize = 160) {
  const pools: THREE.InstancedMesh[] = [];
  forest.traverse(object => { if (object instanceof THREE.InstancedMesh) pools.push(object); });
  const matrix = new THREE.Matrix4(), position = new THREE.Vector3(), color = new THREE.Color();
  let count = 0;
  for (const pool of pools) {
    const cells = new Map<string, number[]>();
    for (let i = 0; i < pool.count; i++) {
      pool.getMatrixAt(i, matrix); position.setFromMatrixPosition(matrix);
      const key = `${Math.floor(position.x / cellSize)},${Math.floor(position.z / cellSize)}`;
      const indices = cells.get(key);
      if (indices) indices.push(i); else cells.set(key, [i]);
    }
    for (const [cell, indices] of cells) {
      const chunk = new THREE.InstancedMesh(pool.geometry, pool.material, indices.length);
      chunk.name = `${pool.name} / canopy cell ${cell}`;
      chunk.userData = { ...pool.userData };
      chunk.castShadow = pool.castShadow; chunk.receiveShadow = pool.receiveShadow;
      chunk.visible = pool.visible; chunk.layers.mask = pool.layers.mask;
      chunk.renderOrder = pool.renderOrder;
      chunk.matrix.copy(pool.matrix); chunk.matrixAutoUpdate = false;
      chunk.frustumCulled = true;
      for (let i = 0; i < indices.length; i++) {
        pool.getMatrixAt(indices[i], matrix); chunk.setMatrixAt(i, matrix);
        if (pool.instanceColor) { pool.getColorAt(indices[i], color); chunk.setColorAt(i, color); }
      }
      chunk.instanceMatrix.needsUpdate = true;
      if (chunk.instanceColor) chunk.instanceColor.needsUpdate = true;
      chunk.computeBoundingSphere(); chunk.computeBoundingBox();
      pool.parent!.add(chunk); count++;
    }
    pool.removeFromParent(); pool.dispose();
  }
  return count;
}
