/**
 * Scalability stress test: `?scene=stress&count=N` (default 100k).
 * N mixed instances (rocks, conifers, broadleaf trees, bushes) on a flat ground at constant density,
 * so the field grows with N. The camera stands at the edge looking across it: close-up detail in front,
 * thousands of distant objects towards the horizon. Used to find how many instances fit in a frame budget.
 */
import * as THREE from 'three/webgpu';
import { buildVirtualMeshCached, VirtualMesh } from '../../src/index';
import type { DemoApp } from './app';
import { makeBroadleaf, makeBush, makeConifer, makeRock } from './assets';
import { mulberry32 } from './noise';

/** WebGPU's default maxStorageBufferBindingSize is 128 MB; one mat4 per instance is 64 bytes. */
const MAX_PER_MESH = 1_000_000;
const SPACING = 7; // meters between instances

export async function createStressScene(app: DemoApp) {
  const params = new URLSearchParams(location.search);
  const count = Math.min(4 * MAX_PER_MESH, Math.max(1, Number(params.get('count') ?? 100_000)));
  const { scene, camera, controls, vg } = app;

  const specs = [
    { name: 'rock', make: () => makeRock(11, 60), scale: [1.5, 5], color: 0x8a8274 },
    { name: 'conifer', make: () => makeConifer(5), scale: [0.7, 1.3], color: 0xffffff, side: THREE.DoubleSide },
    { name: 'broadleaf', make: () => makeBroadleaf(9), scale: [0.7, 1.2], color: 0xffffff, side: THREE.DoubleSide },
    { name: 'bush', make: () => makeBush(3), scale: [0.8, 1.8], color: 0xffffff, side: THREE.DoubleSide },
  ] as const;

  const side = Math.ceil(Math.sqrt(count)) * SPACING;
  const rand = mulberry32(7);
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const p = new THREE.Vector3();
  const s = new THREE.Vector3();
  const e = new THREE.Euler();

  let fullDetail = 0;
  for (let k = 0; k < specs.length; k++) {
    const spec = specs[k];
    app.progress(`Building ${spec.name}…`, k / specs.length);
    await new Promise((r) => setTimeout(r, 0));
    const data = await buildVirtualMeshCached(spec.make());

    const n = Math.min(MAX_PER_MESH, Math.floor(count / specs.length) + (k < count % specs.length ? 1 : 0));
    const matrices = new Float32Array(n * 16);
    for (let i = 0; i < n; i++) {
      const sc = spec.scale[0] + rand() * (spec.scale[1] - spec.scale[0]);
      e.set(0, rand() * Math.PI * 2, 0);
      q.setFromEuler(e);
      p.set(rand() * side, 0, -rand() * side);
      m.compose(p, q, s.setScalar(sc));
      m.toArray(matrices, i * 16);
    }
    const material = new THREE.MeshStandardNodeMaterial({ color: spec.color, roughness: 0.85, side: 'side' in spec ? spec.side : THREE.FrontSide });
    const mesh = new VirtualMesh(vg, data, material, { matrices });
    mesh.name = spec.name;
    scene.add(mesh);
    fullDetail += data.stats.leafTriangles * n;
  }

  scene.background = new THREE.Color(0xa8c4dc);
  scene.fog = new THREE.FogExp2(0xa8c4dc, 0.6 / side);
  scene.add(new THREE.HemisphereLight(0xdfeaff, 0x4d4030, 2.2));
  const sun = new THREE.DirectionalLight(0xfff0dc, 2.5);
  sun.position.set(1, 2, 1);
  scene.add(sun);
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(side * 3, side * 3).rotateX(-Math.PI / 2),
    new THREE.MeshStandardNodeMaterial({ color: 0x4f6b32, roughness: 1 })
  );
  ground.position.set(side / 2, 0, -side / 2);
  scene.add(ground);

  camera.near = 0.5;
  camera.far = side * 3;
  camera.updateProjectionMatrix();
  camera.position.set(side * 0.1, 25, side * 0.05);
  controls.target.set(side * 0.6, 0, -side * 0.6);

  app.extraStats = `stress: ${count.toLocaleString()} instances, ${(fullDetail / 1e9).toFixed(2)}B full-detail triangles`;
}
