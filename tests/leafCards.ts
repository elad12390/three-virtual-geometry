import * as THREE from 'three';

/**
 * A palm-like crown of separate leaf cards: `fronds` fronds, each a row of bent three-quad leaflets on both sides
 * (like the leaflets of a palm frond). Every leaflet is its own piece, so edge collapses alone stall on it.
 */
export function leafCards(fronds = 12, leafletsPerSide = 36, seed = 3): THREE.BufferGeometry {
  let s = seed;
  const rand = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  for (let f = 0; f < fronds; f++) {
    const yaw = (f / fronds) * Math.PI * 2 + rand() * 0.3;
    const dir = new THREE.Vector3(Math.cos(yaw), 0, Math.sin(yaw));
    const side = new THREE.Vector3(-dir.z, 0, dir.x);
    for (let i = 0; i < leafletsPerSide; i++) {
      const t = (i + 0.5) / leafletsPerSide;
      const base = dir.clone().multiplyScalar(t * 4.5).add(new THREE.Vector3(0, 8 + 1.2 * t - 2.2 * t * t, 0));
      for (const sign of [-1, 1]) {
        const out = side.clone().multiplyScalar(sign).addScaledVector(dir, 0.5).normalize();
        const length = 1.1 * (1 - 0.6 * Math.abs(t - 0.4)) * (0.85 + 0.3 * rand());
        const first = positions.length / 3;
        for (let k = 0; k <= 3; k++) {
          const u = k / 3;
          const p = base.clone().addScaledVector(out, u * length);
          p.y -= 0.25 * u * u * length; // the leaflet droops toward its tip
          for (const w of [-0.033, 0.033]) {
            const q = p.clone().addScaledVector(dir, w);
            positions.push(q.x, q.y, q.z);
            uvs.push(u, w > 0 ? 1 : 0);
          }
        }
        for (let k = 0; k < 3; k++) {
          const a = first + k * 2;
          indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
        }
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  g.setIndex(indices);
  g.computeVertexNormals();
  return g;
}
