import { expect, test } from 'vitest';
import * as THREE from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { fromBufferGeometry } from '../src/index';

test('fromBufferGeometry matches three.js mergeVertices', async () => {
  const assert = (cond: unknown, msg: string) => {
    expect(cond, msg).toBeTruthy();
  };

  /** The previous fromBufferGeometry, built on three's string-keyed mergeVertices. */
  function reference(geometry: THREE.BufferGeometry) {
    let g = geometry.clone();
    for (const name of Object.keys(g.attributes)) if (name !== 'position') g.deleteAttribute(name);
    g = mergeVertices(g, 1e-6);
    g.computeVertexNormals();
    return { positions: g.attributes.position.array, normals: g.attributes.normal.array, indices: g.index!.array };
  }

  const same = (a: ArrayLike<number>, b: ArrayLike<number>) => a.length === b.length && Array.prototype.every.call(a, (v: number, i: number) => v === b[i]);

  const cases: [string, THREE.BufferGeometry][] = [
    ['sphere', new THREE.SphereGeometry(1, 48, 24)],
    ['box', new THREE.BoxGeometry(2, 1, 3, 4, 4, 4)],
    ['torus knot (non-indexed)', new THREE.TorusKnotGeometry(5, 1.2, 64, 12).toNonIndexed()],
    ['large coordinates', new THREE.CylinderGeometry(800, 900, 2000, 32, 8).translate(1500, -300, 4000)],
  ];
  for (const [name, geometry] of cases) {
    const ref = reference(geometry);
    const got = fromBufferGeometry(geometry, { uvs: false });
    assert(same(got.positions, ref.positions), `fromBufferGeometry ${name}: same welded positions as mergeVertices`);
    assert(same(got.indices, ref.indices), `fromBufferGeometry ${name}: same indices as mergeVertices`);
    assert(same(got.normals, ref.normals), `fromBufferGeometry ${name}: same normals as mergeVertices`);
  }
}, 300_000);
