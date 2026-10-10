import { expect, test } from 'vitest';
import * as THREE from 'three';
import { buildVirtualMesh, fromBufferGeometry, instanceMeshletRange, LOD_FADE_STEPS, selectBlendCut, selectCut, verifyCutCoverage, verifyCutOverlap } from '../src/index';

test('LOD cut covers every leaf exactly once (no holes, no overlaps)', async () => {
  const assert = (cond: unknown, msg: string) => {
    expect(cond, msg).toBeTruthy();
  };

  // Seeded RNG so the test is deterministic.
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

  function terrain() {
    const g = new THREE.PlaneGeometry(400, 400, 96, 96).rotateX(-Math.PI / 2);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) p.setY(i, 30 * Math.sin(p.getX(i) * 0.05) * Math.cos(p.getZ(i) * 0.04));
    return g;
  }

  const data = await buildVirtualMesh(fromBufferGeometry(terrain()));
  assert(data.replacementGroup.length === data.meshletCount, 'DAG links: one replacement group per meshlet');

  // Views: aerial and low, several positions, thresholds in pixels.
  const views = [];
  for (let k = 0; k < 12; k++) {
    const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 5000);
    camera.position.set((rand() - 0.5) * 300, 2 + rand() * 80, (rand() - 0.5) * 300);
    camera.lookAt((rand() - 0.5) * 200, 0, (rand() - 0.5) * 200);
    camera.updateMatrixWorld();
    views.push({ camera, projScale: camera.projectionMatrix.elements[5] * 1080 * 0.5 });
  }

  let worst = 0;
  for (const { camera, projScale } of views) {
    for (const threshold of [0.5, 1, 3]) {
      const selected = selectCut(data, { viewMatrix: camera.matrixWorldInverse.elements, projScale, near: camera.near }, threshold);
      const holes = verifyCutCoverage(data, selected);
      worst = Math.max(worst, holes.length);
      assert(holes.length === 0, `no holes at threshold ${threshold}px (camera ${camera.position.toArray().map((v) => v.toFixed(0)).join(',')})`);
      const view = { viewMatrix: camera.matrixWorldInverse.elements, projScale, near: camera.near };
      const [start, end] = instanceMeshletRange(data, view, threshold);
      let outside = 0;
      let inCut = 0;
      for (let i = 0; i < data.meshletCount; i++) {
        if (!selected[i]) continue;
        inCut++;
        if (i < start || i >= end) outside++;
      }
      assert(outside === 0, `level range [${start}, ${end}) contains all ${inCut} selected meshlets (threshold ${threshold}px)`);
    }
  }
  console.log(`worst hole count across views: ${worst}`);

  // Far away, the level range should shrink to a few coarse levels (that is the point of the instance pass).
  {
    const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 50000);
    camera.position.set(0, 2000, 8000);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const view = { viewMatrix: camera.matrixWorldInverse.elements, projScale: camera.projectionMatrix.elements[5] * 540, near: 0.1 };
    const [start, end] = instanceMeshletRange(data, view, 1);
    assert(end - start < data.meshletCount / 4, `far instance tests only ${end - start} of ${data.meshletCount} meshlets`);
  }
}, 300_000);

test('LOD blend: every pixel step of the band shows a complete cut, without overlaps', async () => {
  const assert = (cond: unknown, msg: string) => {
    expect(cond, msg).toBeTruthy();
  };
  const g = new THREE.PlaneGeometry(400, 400, 96, 96).rotateX(-Math.PI / 2);
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) p.setY(i, 30 * Math.sin(p.getX(i) * 0.05) * Math.cos(p.getZ(i) * 0.04));
  const data = await buildVirtualMesh(fromBufferGeometry(g));

  let blended = 0;
  for (const [x, y, z] of [
    [0, 40, 260],
    [150, 6, -40],
    [-300, 200, 300],
  ]) {
    const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 5000);
    camera.position.set(x, y, z);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const view = { viewMatrix: camera.matrixWorldInverse.elements, projScale: camera.projectionMatrix.elements[5] * 540, near: 0.1 };
    for (const [threshold, blend] of [
      [1, 2],
      [0.7, 1.5],
      [2, 3],
    ]) {
      const { lo, hi } = selectBlendCut(data, view, threshold, blend);
      // The band contains the plain cuts at both of its ends.
      for (const t of [threshold, threshold * blend * 0.999]) {
        const plain = selectCut(data, view, t);
        for (let i = 0; i < data.meshletCount; i++) assert(!plain[i] || hi[i] > 0, `plain cut at ${t}px is inside the blend selection`);
      }
      for (let step = 0; step < LOD_FADE_STEPS; step++) {
        const selected = new Uint8Array(data.meshletCount);
        for (let i = 0; i < data.meshletCount; i++) selected[i] = lo[i] <= step && step < hi[i] ? 1 : 0;
        assert(verifyCutCoverage(data, selected).length === 0, `no holes at step ${step} (${threshold}px x${blend})`);
        assert(verifyCutOverlap(data, selected).length === 0, `no overlaps at step ${step} (${threshold}px x${blend})`);
      }
      for (let i = 0; i < data.meshletCount; i++) if (hi[i] > 0 && (lo[i] > 0 || hi[i] < LOD_FADE_STEPS)) blended++;
    }
  }
  assert(blended > 0, 'some meshlets are partly blended');
  // Blend 1 is the plain cut.
  const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 5000);
  camera.position.set(0, 40, 260);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  const view = { viewMatrix: camera.matrixWorldInverse.elements, projScale: camera.projectionMatrix.elements[5] * 540, near: 0.1 };
  const plain = selectCut(data, view, 1);
  const { lo, hi } = selectBlendCut(data, view, 1, 1);
  for (let i = 0; i < data.meshletCount; i++) assert((plain[i] === 1) === (hi[i] > 0) && (hi[i] === 0 || (lo[i] === 0 && hi[i] === LOD_FADE_STEPS)), 'blend 1 = plain cut');
}, 300_000);
