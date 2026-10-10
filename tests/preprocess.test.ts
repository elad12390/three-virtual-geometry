import { expect, test } from 'vitest';
import * as THREE from 'three';
import { buildVirtualMesh, fromBufferGeometry, partitionMeshlets, MAX_MESHLET_TRIANGLES } from '../src/index';
import { ERROR_INFINITY, MESHLET_BOUNDS_STRIDE, MESHLET_INFO_STRIDE } from '../src/core/constants';
import { selectCut, verifyCutCoverage } from '../src/index';
import { leafCards } from './leafCards';

test('DAG builder invariants', async () => {
  const assert = (cond: unknown, msg: string) => {
    expect(cond, msg).toBeTruthy();
  };

  async function checkDag(name: string, geometry: THREE.BufferGeometry, opts = {}) {
    const src = fromBufferGeometry(geometry);
    const d = await buildVirtualMesh(src, opts);
    assert(d.meshletCount > 0, `${name}: produces meshlets`);
    assert(d.stats.lodLevels > 1, `${name}: builds more than one LOD level`);
    assert(d.stats.leafTriangles === src.indices.length / 3, `${name}: leaf level covers every input triangle`);
    let monotonic = true;
    let sized = true;
    let rootsSeen = 0;
    for (let i = 0; i < d.meshletCount; i++) {
      const err = d.meshletBounds[i * MESHLET_BOUNDS_STRIDE + 8];
      const parentErr = d.meshletBounds[i * MESHLET_BOUNDS_STRIDE + 9];
      if (parentErr < err) monotonic = false;
      const tris = d.meshletInfo[i * MESHLET_INFO_STRIDE];
      if (tris === 0 || tris > MAX_MESHLET_TRIANGLES) sized = false;
      if (parentErr >= ERROR_INFINITY) rootsSeen++;
    }
    assert(monotonic, `${name}: parent error >= own error for every meshlet`);
    assert(sized, `${name}: every meshlet has 1..${MAX_MESHLET_TRIANGLES} triangles`);
    assert(rootsSeen === d.stats.rootMeshlets, `${name}: root count matches stats`);

    // The compact meshlet encoding (local vertex list + packed local triangles) must decode to the
    // same triangles as the flat index list.
    let decodeOk = true;
    for (let i = 0; i < d.meshletCount && decodeOk; i++) {
      const tris = d.meshletInfo[i * MESHLET_INFO_STRIDE];
      const firstIndex = d.meshletInfo[i * MESHLET_INFO_STRIDE + 1];
      const vertexOffset = d.meshletInfo[i * MESHLET_INFO_STRIDE + 3];
      for (let t = 0; t < tris; t++) {
        const packed = d.meshletTriangles[firstIndex / 3 + t];
        for (let k = 0; k < 3; k++) {
          const local = (packed >> (8 * k)) & 255;
          if (d.meshletVertices[vertexOffset + local] !== d.indices[firstIndex + t * 3 + k]) decodeOk = false;
        }
      }
    }
    assert(decodeOk, `${name}: packed meshlet triangles decode to the original index list`);
    return d;
  }

  function noisySphere() {
    const g = new THREE.IcosahedronGeometry(1, 40);
    const p = g.attributes.position;
    const v = new THREE.Vector3();
    for (let i = 0; i < p.count; i++) {
      v.fromBufferAttribute(p, i);
      const n = 1 + 0.1 * Math.sin(v.x * 9) * Math.cos(v.y * 7);
      p.setXYZ(i, v.x * n, v.y * n, v.z * n);
    }
    return g;
  }

  function terrain() {
    const g = new THREE.PlaneGeometry(100, 100, 64, 64).rotateX(-Math.PI / 2);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) p.setY(i, 4 * Math.sin(p.getX(i) * 0.2) * Math.cos(p.getZ(i) * 0.13));
    return g;
  }

  await checkDag('noisy sphere', noisySphere());
  await checkDag('terrain', terrain());
  await checkDag('terrain (prune)', terrain(), { prune: true });

  // Leaf cards: separate pieces, simplified piece by piece and thinned (aggregate.ts).
  const leaves = await checkDag('leaf cards', leafCards());
  assert((leaves.stats.aggregateGroups ?? 0) > 0, 'leaf cards: groups are simplified piece by piece');
  const level1 = leaves.levelRanges[3];
  let level1Triangles = 0;
  for (let i = leaves.levelRanges[2]; i < leaves.levelRanges[2] + level1; i++) level1Triangles += leaves.meshletInfo[i * MESHLET_INFO_STRIDE];
  assert(level1Triangles <= leaves.stats.leafTriangles * 0.6, `leaf cards: the first level halves the triangles (${level1Triangles} of ${leaves.stats.leafTriangles})`);
  for (const distance of [5, 20, 60, 200, 800]) {
    const view = { viewMatrix: new THREE.Matrix4().makeTranslation(0, -8, -distance).elements, projScale: 1000, near: 0.1 };
    assert(verifyCutCoverage(leaves, selectCut(leaves, view, 1)).length === 0, `leaf cards: complete cut at ${distance} m`);
  }

  const groups = partitionMeshlets(
    Array.from({ length: 40 }, (_, i) => ({ boundaryEdges: [i, i + 1], center: [i, 0, 0] as [number, number, number] })),
    8
  );
  assert(groups.flat().length === 40 && new Set(groups.flat()).size === 40, 'partitionMeshlets: every meshlet in exactly one group');
  assert(groups.every((g) => g.length <= 8), 'partitionMeshlets: groups respect the size limit');
}, 300_000);
