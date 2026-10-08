import { expect, test } from 'vitest';
import * as THREE from 'three';
import { buildVirtualMesh, fromBufferGeometry, instanceMeshletRange, selectCut, verifyCutCoverage } from '../src/index';

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
  assert(data.replacementStart.length === data.meshletCount + 1, 'DAG links: one CSR row per meshlet');

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
