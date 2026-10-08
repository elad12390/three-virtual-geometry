import { expect, test } from 'vitest';
import * as THREE from 'three';
import {
  buildVirtualMesh,
  buildVirtualMeshCached,
  decodeVirtualMesh,
  encodeVirtualMesh,
  fromBufferGeometry,
  loadVirtualMesh,
  virtualMeshCacheKey,
  VG_FORMAT_VERSION,
  type VirtualMeshData,
} from '../src/index';
import { makeConifer, makeRock } from '../examples/demo/assets';

test('baked file format round-trips and rejects bad input', async () => {
  const assert = (cond: unknown, msg: string) => {
    expect(cond, msg).toBeTruthy();
  };

  const sameBytes = (a: ArrayBufferView, b: ArrayBufferView) => {
    if (a.byteLength !== b.byteLength) return false;
    const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    const y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
    return true;
  };

  /** Every field identical: typed arrays by type and bytes, everything else by value. Same key order. */
  function identical(a: VirtualMeshData, b: VirtualMeshData): string | null {
    const ra = a as unknown as Record<string, unknown>;
    const rb = b as unknown as Record<string, unknown>;
    if (Object.keys(ra).join() !== Object.keys(rb).join()) return `keys differ: ${Object.keys(ra)} vs ${Object.keys(rb)}`;
    for (const key of Object.keys(ra)) {
      const x = ra[key];
      const y = rb[key];
      if (ArrayBuffer.isView(x)) {
        if (!ArrayBuffer.isView(y) || x.constructor !== y.constructor || !sameBytes(x, y)) return `field ${key} differs`;
      } else if (!Object.is(x, y) && JSON.stringify(x) !== JSON.stringify(y)) {
        return `field ${key} differs`;
      }
    }
    return null;
  }

  const headerOf = (bytes: Uint8Array) => {
    const view = new DataView(bytes.buffer, bytes.byteOffset);
    return JSON.parse(new TextDecoder().decode(bytes.subarray(16, 16 + view.getUint32(8, true))));
  };

  async function rejects(promise: Promise<unknown>, pattern: RegExp, msg: string) {
    try {
      await promise;
    } catch (e) {
      assert(pattern.test((e as Error).message), `${msg} (${(e as Error).message})`);
      return;
    }
    assert(false, `${msg}: expected an error`);
  }

  function sphere() {
    return fromBufferGeometry(new THREE.SphereGeometry(1, 96, 64));
  }

  const meshes: [string, VirtualMeshData][] = [
    ['sphere', await buildVirtualMesh(sphere())],
    ['rock with colors', await buildVirtualMesh(makeRock(11, 32))],
    ['conifer (voxel LODs)', await buildVirtualMesh(makeConifer(5))],
  ];
  assert(meshes[1][1].colors !== null, 'rock has vertex colors');
  assert(meshes[2][1].stats.voxelLevels > 0, 'conifer build creates voxel LOD levels');

  for (const [name, data] of meshes) {
    for (const options of [{}, { deflate: false }, { compress: false }, { keepIndices: true }]) {
      const label = `${name} ${JSON.stringify(options)}`;
      const bytes = await encodeVirtualMesh(data, options);
      const back = await decodeVirtualMesh(bytes);
      const diff = identical(data, back);
      assert(diff === null, `${label}: round trip is exact${diff ? ` (${diff})` : ''}`);
      const header = headerOf(bytes);
      const indicesField = header.fields.find((f: { name: string }) => f.name === 'indices');
      if (options.keepIndices) assert(indicesField.encoding !== 'meshlet-indices', `${label}: indices stored`);
      else assert(indicesField.encoding === 'meshlet-indices' && indicesField.byteLength === 0, `${label}: indices dropped and rebuilt identically`);
    }
  }

  // ------------------------------------------------------------------ container
  const [, rock] = meshes[1];
  const bytes = await encodeVirtualMesh(rock);
  const view = new DataView(bytes.buffer, bytes.byteOffset);
  assert(new TextDecoder().decode(bytes.subarray(0, 4)) === 'VGEO', 'magic bytes VGEO');
  assert(view.getUint32(4, true) === VG_FORMAT_VERSION, 'format version in the prefix');
  const header = headerOf(bytes);
  assert(header.fields.every((f: { offset: number }) => f.offset % 4 === 0) && bytes.length % 4 === 0, 'payload blobs are 4-byte aligned');
  assert(
    header.fields.some((f: { encoding: string }) => f.encoding === 'meshopt-vertex') && header.fields.some((f: { encoding: string }) => f.encoding === 'meshlet-triangles'),
    'meshoptimizer codecs are used'
  );
  const raw = Object.values(rock).reduce((acc, v) => acc + (ArrayBuffer.isView(v) ? v.byteLength : 0), 0);
  assert(bytes.length * 2.5 < raw, `encoded is much smaller than the arrays (${(bytes.length / 1024).toFixed(0)} KB vs ${(raw / 1024).toFixed(0)} KB)`);
  assert(identical(rock, await loadVirtualMesh(bytes.buffer)) === null, 'loadVirtualMesh decodes an ArrayBuffer');
  const unaligned = new Uint8Array(bytes.length + 1);
  unaligned.set(bytes, 1);
  assert(identical(rock, await decodeVirtualMesh(unaligned.subarray(1))) === null, 'decodes from an unaligned view');

  // Fields are discovered, not listed: new typed arrays, nulls and non-finite numbers survive.
  const extended = { ...rock, uvs: new Float32Array(rock.positions.length / 3 * 2).map((_, i) => i * 0.25), tangents: null, extra: { scale: Infinity, tag: 'x' } };
  const extendedBack = await decodeVirtualMesh(await encodeVirtualMesh(extended as unknown as VirtualMeshData));
  assert(identical(extended as unknown as VirtualMeshData, extendedBack) === null, 'unknown fields (typed array, null, object with Infinity) round-trip');

  // ------------------------------------------------------------------ rejection
  const patched = (fn: (b: Uint8Array) => void) => {
    const copy = bytes.slice();
    fn(copy);
    return copy;
  };
  await rejects(decodeVirtualMesh(new Uint8Array(0)), /not a three-virtual-geometry/, 'empty input rejected');
  await rejects(decodeVirtualMesh(new TextEncoder().encode('{"glTF": "not this"}')), /not a three-virtual-geometry/, 'foreign file rejected');
  await rejects(decodeVirtualMesh(patched((b) => new DataView(b.buffer).setUint32(4, VG_FORMAT_VERSION + 1, true))), /unsupported .*version/, 'newer format version rejected');
  await rejects(decodeVirtualMesh(patched((b) => (b[b.length - 100] ^= 0x10))), /corrupt/, 'flipped payload byte rejected');
  await rejects(decodeVirtualMesh(bytes.slice(0, bytes.length - 64)), /truncated/, 'truncated file rejected');
  await rejects(decodeVirtualMesh(bytes.slice(0, 40)), /truncated/, 'file cut inside the header rejected');

  // ------------------------------------------------------------------ cache (Node has no IndexedDB: plain build)
  const src = sphere();
  const cached = await buildVirtualMeshCached(src, { onProgress: () => {} });
  const direct = await buildVirtualMesh(src);
  cached.stats.buildMs = direct.stats.buildMs;
  assert(identical(cached, direct) === null, 'buildVirtualMeshCached without IndexedDB builds the same data');
  const key = await virtualMeshCacheKey(src, { prune: true });
  assert(key === (await virtualMeshCacheKey(src, { prune: true, onProgress: () => {} })), 'cache key ignores callbacks');
  assert(key !== (await virtualMeshCacheKey(src, { prune: false })), 'cache key depends on build options');
  const moved = { ...src, positions: src.positions.slice() };
  moved.positions[0] += 1e-6;
  assert(key !== (await virtualMeshCacheKey(moved, { prune: true })), 'cache key depends on source contents');
}, 300_000);
