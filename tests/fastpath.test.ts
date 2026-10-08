import { expect, test } from 'vitest';
import * as THREE from 'three';
import { buildVirtualMesh, fromBufferGeometry, instanceGuaranteedRange, selectCut } from '../src/index';
import { FAST_PATH_MAX_MESHLETS, MESHLET_BOUNDS_STRIDE } from '../src/core/constants';

test('fast path only emits meshlets the full test would select', async () => {
  const assert = (cond: unknown, msg: string) => {
    expect(cond, msg).toBeTruthy();
  };

  // Seeded RNG so the test is deterministic.
  let seed = 11;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

  function terrain() {
    const g = new THREE.PlaneGeometry(400, 400, 96, 96).rotateX(-Math.PI / 2);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) p.setY(i, 30 * Math.sin(p.getX(i) * 0.05) * Math.cos(p.getZ(i) * 0.04));
    return g;
  }

  const data = await buildVirtualMesh(fromBufferGeometry(terrain()));
  const B = data.meshletBounds;
  const dist = (a: ArrayLike<number>, b: ArrayLike<number>) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

  // Invariants the fast path relies on.
  {
    const C = data.cullBoundsSphere;
    let ok = true;
    for (let i = 0; i < data.meshletCount; i++) {
      const b = i * MESHLET_BOUNDS_STRIDE;
      if (dist(B.subarray(b + 12, b + 15), C) + B[b + 15] > C[3]) ok = false;
    }
    assert(ok, 'cullBoundsSphere encloses every meshlet culling sphere');

    const L = data.lodBoundsSphere;
    ok = true;
    for (let i = 0; i < data.meshletCount; i++) {
      const b = i * MESHLET_BOUNDS_STRIDE;
      if (dist(B.subarray(b, b + 3), L) + B[b + 3] > L[3]) ok = false;
      if (data.meshletBounds[b + 9] < 1e29 && dist(B.subarray(b + 4, b + 7), L) + B[b + 7] > L[3]) ok = false;
    }
    assert(ok, 'lodBoundsSphere encloses every LOD and parent sphere');

    const levels = data.levelRanges.length / 2;
    ok = true;
    for (let l = 0; l < levels; l++) {
      let maxOwn = 0;
      let minParent = Infinity;
      for (let i = data.levelRanges[l * 2]; i < data.levelRanges[l * 2] + data.levelRanges[l * 2 + 1]; i++) {
        maxOwn = Math.max(maxOwn, B[i * MESHLET_BOUNDS_STRIDE + 8]);
        minParent = Math.min(minParent, B[i * MESHLET_BOUNDS_STRIDE + 9]);
      }
      if (data.levelGuardErrors[l * 2] !== maxOwn || data.levelGuardErrors[l * 2 + 1] !== minParent) ok = false;
    }
    assert(ok, 'levelGuardErrors are the per-level max own error and min parent error');
  }

  // Conservativeness: whenever the fast path fires for an instance, the per-meshlet LOD test (the baseline cut,
  // ignoring frustum culling) selects exactly the meshlets of the fast range, and nothing else.
  let fast = 0;
  let fastSingle = 0;
  let instancesChecked = 0;
  let violations = 0;
  for (let k = 0; k < 40; k++) {
    // Meshlet errors are in object units (the root's own error is ~270 here), so single coarse meshlets only
    // pass the pixel threshold from very far away. Cover that range, plus a few closer cameras.
    const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1e8);
    const dir = new THREE.Vector3(rand() - 0.5, 0.05 + rand() * 0.6, rand() - 0.5).normalize();
    camera.position.copy(dir.multiplyScalar(k < 30 ? Math.pow(10, 4.3 + rand() * 2) : 100 * Math.pow(10, rand() * 2)));
    camera.lookAt((rand() - 0.5) * 200, 0, (rand() - 0.5) * 200);
    camera.updateMatrixWorld();
    const projScale = camera.projectionMatrix.elements[5] * 1080 * 0.5;

    for (let j = 0; j < 5; j++) {
      const s = [1, 0.5, 2, 3][j % 4] * (0.75 + rand() * 0.5);
      const model = new THREE.Matrix4().makeTranslation((rand() - 0.5) * 600, (rand() - 0.5) * 60, (rand() - 0.5) * 600).scale(new THREE.Vector3(s, s, s));
      const modelView = camera.matrixWorldInverse.clone().multiply(model);
      for (const threshold of [0.5, 1, 3]) {
        instancesChecked++;
        const view = { viewMatrix: modelView.elements, projScale, near: camera.near, scale: s };
        const range = instanceGuaranteedRange(data, view, threshold);
        if (!range) continue;
        const [start, end] = range;
        fast++;
        if (end - start === 1) fastSingle++;
        if (end <= start || end - start > FAST_PATH_MAX_MESHLETS) violations++;
        const selected = selectCut(data, view, threshold);
        let inRange = 0;
        let total = 0;
        for (let i = 0; i < data.meshletCount; i++) {
          if (!selected[i]) continue;
          total++;
          if (i >= start && i < end) inRange++;
        }
        if (inRange !== end - start || total !== inRange) violations++;
      }
    }
  }

  // Targeted cases: for every meshlet, put the camera on the line through the bound's centre and the meshlet
  // sphere, so the meshlet's own (or parent's) pixel error crosses the threshold right at the edge of the distance
  // band. A bound that is too tight (wrong end of the band, too big a margin) fails here, not in random views.
  {
    const probe = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1e8);
    const projScale = probe.projectionMatrix.elements[5] * 1080 * 0.5;
    const C = new THREE.Vector3(data.lodBoundsSphere[0], data.lodBoundsSphere[1], data.lodBoundsSphere[2]);
    const dirOf = (x: number, y: number, z: number) => {
      const v = new THREE.Vector3(x, y, z).sub(C);
      return v.lengthSq() > 0 ? v.normalize() : new THREE.Vector3(rand() - 0.5, rand() - 0.5, rand() - 0.5).normalize();
    };
    for (let i = 0; i < data.meshletCount; i++) {
      const b = i * MESHLET_BOUNDS_STRIDE;
      const own = { c: [B[b], B[b + 1], B[b + 2]], r: B[b + 3], e: B[b + 8] };
      const parent = { c: [B[b + 4], B[b + 5], B[b + 6]], r: B[b + 7], e: B[b + 9] };
      for (const threshold of [0.5, 1, 3]) {
        for (const s of [1, 2.5]) {
          for (let k = 0; k <= 32; k++) {
            const f = 0.996 + k * 0.00025; // sweep across the distance band edge (band is ~0.2% wide here)
            for (const mode of ['own', 'parent'] as const) {
              const m = mode === 'own' ? own : parent;
              if (!(m.e > 0 && m.e < 1e29)) continue;
              const offset = dist(m.c, C.toArray());
              // Own: camera beyond the meshlet, pixel = e * s * ps / (D - s*(offset + r)) = t at the near edge.
              // Parent: camera on the opposite side, pixel = e * s * ps / (D + s*(offset - r)) = t at the far edge.
              const dirSign = mode === 'own' ? 1 : -1;
              const u = dirOf(m.c[0], m.c[1], m.c[2]).multiplyScalar(dirSign);
              const Dc = mode === 'own' ? (m.e * s * projScale) / threshold + s * (offset + m.r) : (m.e * s * projScale) / threshold - s * (offset - m.r);
              if (!(Dc > 0)) continue;
              const T = new THREE.Vector3((rand() - 0.5) * 600, (rand() - 0.5) * 60, (rand() - 0.5) * 600);
              const model = new THREE.Matrix4().makeTranslation(T.x, T.y, T.z).scale(new THREE.Vector3(s, s, s));
              const centerWorld = C.clone().applyMatrix4(model);
              const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1e8);
              camera.position.copy(centerWorld).addScaledVector(u, Dc * f);
              camera.lookAt(centerWorld);
              camera.updateMatrixWorld();
              const modelView = camera.matrixWorldInverse.clone().multiply(model);
              instancesChecked++;
              const view = { viewMatrix: modelView.elements, projScale, near: camera.near, scale: s };
              const range = instanceGuaranteedRange(data, view, threshold);
              if (!range) continue;
              const [start, end] = range;
              fast++;
              if (end - start === 1) fastSingle++;
              if (end <= start || end - start > FAST_PATH_MAX_MESHLETS) violations++;
              const selected = selectCut(data, view, threshold);
              let inRange = 0;
              let total = 0;
              for (let j = 0; j < data.meshletCount; j++) {
                if (!selected[j]) continue;
                total++;
                if (j >= start && j < end) inRange++;
              }
              if (inRange !== end - start || total !== inRange) violations++;
            }
          }
        }
      }
    }
  }
  console.log(`fast path fired for ${fast} of ${instancesChecked} instance/threshold cases (${fastSingle} single-meshlet)`);
  assert(fast > 0 && fastSingle > 0, 'the fast path fires for far instances (test is not vacuous)');
  assert(violations === 0, 'every fast-path range equals the per-meshlet LOD cut for that instance');
}, 300_000);
