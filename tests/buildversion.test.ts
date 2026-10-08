import { expect, test } from 'vitest';
/**
 * Guards the build cache: cached builds are keyed by VG_BUILD_VERSION, so any change to what
 * buildVirtualMesh produces must bump it, or users keep loading stale builds. This test fingerprints the
 * output for fixed inputs; when the fingerprint changes, bump the version and record the new value here.
 */
import { createHash } from 'node:crypto';
import * as THREE from 'three';
import { buildVirtualMesh, fromBufferGeometry, VG_BUILD_VERSION } from '../src/index';
import { makeBroadleaf } from '../examples/demo/assets';

test('build output matches the recorded fingerprint for VG_BUILD_VERSION', async () => {
  const RECORDED = { version: 3, fingerprint: '66ea358be060a22e' };

  const assert = (cond: unknown, msg: string) => {
    expect(cond, msg).toBeTruthy();
  };

  const hash = createHash('sha256');
  for (const source of [fromBufferGeometry(new THREE.TorusKnotGeometry(1, 0.3, 160, 24)), makeBroadleaf(9)]) {
    const data = (await buildVirtualMesh(source)) as unknown as Record<string, unknown>;
    for (const key of Object.keys(data).sort()) {
      const value = data[key];
      hash.update(key);
      if (ArrayBuffer.isView(value)) hash.update(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
      else if (key !== 'stats') hash.update(JSON.stringify(value));
    }
  }
  const fingerprint = hash.digest('hex').slice(0, 16);

  if (VG_BUILD_VERSION === RECORDED.version) {
    assert(
      fingerprint === RECORDED.fingerprint,
      `build output unchanged for VG_BUILD_VERSION ${VG_BUILD_VERSION} (got ${fingerprint}; if the change is intended, ` +
        `bump VG_BUILD_VERSION in src/core/io/cache.ts and record the new version and fingerprint in this test)`
    );
  } else {
    assert(false, `VG_BUILD_VERSION is ${VG_BUILD_VERSION}: record { version: ${VG_BUILD_VERSION}, fingerprint: '${fingerprint}' } here`);
  }
}, 300_000);
