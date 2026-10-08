import * as THREE from 'three/webgpu';
import { buildVirtualMeshCached } from '../../src/index';
import { VirtualMesh } from '../../src/index';
import { fromBufferGeometry } from '../../src/source';
import type { DemoApp } from './app';

/** Minimal scene: one dense rock instanced on a grid. Open with `?scene=test`. */
export async function createTestScene(app: DemoApp) {
  const { scene, camera, controls } = app;
  scene.background = new THREE.Color(0x202630);
  scene.add(new THREE.HemisphereLight(0xbfd8ff, 0x403020, 1.5));
  const sun = new THREE.DirectionalLight(0xffffff, 2.5);
  sun.position.set(3, 5, 2);
  scene.add(sun);

  const rock = new THREE.IcosahedronGeometry(1, 80);
  const p = rock.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const n = 1 + 0.12 * Math.sin(v.x * 9) * Math.cos(v.y * 7) * Math.sin(v.z * 11) + 0.03 * Math.sin(v.x * 40 + v.y * 31);
    p.setXYZ(i, v.x * n, v.y * n * 0.8, v.z * n);
  }

  app.progress('Building meshlet DAG…', 0);
  const data = await buildVirtualMeshCached(fromBufferGeometry(rock), { onProgress: (f) => app.progress('Building meshlet DAG…', f) });
  console.log('vg build', data.stats, 'meshlets', data.meshletCount);

  const grid = 30;
  const matrices = new Float32Array(grid * grid * 16);
  const m = new THREE.Matrix4();
  for (let x = 0; x < grid; x++) {
    for (let z = 0; z < grid; z++) {
      m.makeRotationY((x * 7 + z * 13) % 6.28).setPosition((x - grid / 2) * 3, 0, (z - grid / 2) * 3);
      m.toArray(matrices, (x * grid + z) * 16);
    }
  }

  const material = new THREE.MeshStandardNodeMaterial({ color: 0x9a8f80, roughness: 0.9 });
  scene.add(new VirtualMesh(app.vg, data, material, { matrices }));

  camera.position.set(0, 12, 50);
  controls.target.set(0, 0, 0);
  app.extraStats = `asset: ${(data.stats.leafTriangles / 1000).toFixed(0)}k tris, ${data.meshletCount} meshlets, ${data.stats.lodLevels} LODs, built in ${data.stats.buildMs.toFixed(0)} ms`;
}
