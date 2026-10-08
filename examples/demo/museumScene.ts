/**
 * Test scene for the museum architecture: builds the museum and puts a simple stand-in of the right size on every
 * slot (bronze capsules for statues, rock-like blobs for specimens, giants for heroes) to judge composition.
 * Open with `?scene=museum`; `&tour` flies the camera through it, `&arms=north` builds one gallery only.
 */
import * as THREE from 'three/webgpu';
import { color, float, mix, mx_noise_float, positionWorld, smoothstep } from 'three/tsl';
import { buildVirtualMeshCached, fromBufferGeometry } from '../../src/index';
import type { DemoApp } from './app';
import { buildMuseum, type ExhibitSlot } from './museum';
import { mulberry32 } from './noise';
import { makeTour } from './tour';

export async function createMuseumScene(app: DemoApp) {
  const { scene, camera, controls, vg, renderer } = app;
  const params = new URLSearchParams(location.search);
  const museum = await buildMuseum(app);
  const random = mulberry32(5);

  // Weathered bronze for statues, pale stone for specimens.
  const bronze = new THREE.MeshStandardNodeMaterial();
  const weathering = mx_noise_float(positionWorld.mul(0.8)).mul(0.5).add(0.5);
  const patina = smoothstep(0.52, 0.8, weathering);
  bronze.colorNode = mix(color(0x8a5f36), color(0x3f6a58), patina);
  bronze.metalnessNode = mix(float(0.9), float(0.2), patina);
  bronze.roughnessNode = mix(float(0.34), float(0.75), patina);
  const rockMaterial = new THREE.MeshStandardNodeMaterial({ color: 0x9a8f80, roughness: 0.9 });

  const standIns = {
    statue: () => {
      const g = new THREE.CapsuleGeometry(0.42, 2.0, 12, 24);
      g.translate(0, 1.4 + 0.1, 0);
      return g;
    },
    hero: () => {
      const g = new THREE.CapsuleGeometry(1.1, 5.4, 12, 24);
      g.translate(0, 1.1 + 2.7, 0);
      return g;
    },
    specimen: () => {
      const g = new THREE.DodecahedronGeometry(2, 5);
      const p = g.attributes.position;
      const v = new THREE.Vector3();
      for (let i = 0; i < p.count; i++) {
        v.fromBufferAttribute(p, i);
        const n = 1 + 0.12 * Math.sin(v.x * 3) * Math.cos(v.y * 2.5) * Math.sin(v.z * 3.5);
        p.setXYZ(i, v.x * n, v.y * n * 0.5 + 0.9, v.z * n);
      }
      g.computeVertexNormals();
      return g;
    },
  };
  const matrices: Record<ExhibitSlot['kind'], number[]> = { statue: [], hero: [], specimen: [] };
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  for (const slot of museum.slots) {
    const k = slot.kind === 'statue' ? 0.9 + random() * 0.2 : 1;
    q.setFromAxisAngle(up, slot.yaw);
    m.compose(slot.position, q, s.setScalar(k));
    matrices[slot.kind].push(...m.elements);
  }
  let count = 0;
  for (const kind of ['statue', 'hero', 'specimen'] as const) {
    if (!matrices[kind].length || params.has('empty')) continue;
    app.progress(`Placing stand-in ${kind}s…`, 0.9);
    const data = await buildVirtualMeshCached(fromBufferGeometry(standIns[kind]()));
    const mesh = vg.createMesh(data, kind === 'specimen' ? rockMaterial : bronze, { matrices: new Float32Array(matrices[kind]) });
    mesh.castShadow = mesh.receiveShadow = true;
    scene.add(mesh);
    count += matrices[kind].length / 16;
  }

  const keys = museum.tour;
  const { poseAt, duration: tourSeconds } = makeTour(keys, () => 1.0);
  poseAt(0, controls.target, camera.position);
  controls.minDistance = 0.3;
  app.benchPoses = museum.benchPoses;

  const tour = { active: params.has('tour'), time: 0 };
  const stopTour = () => (tour.active = false);
  renderer.domElement.addEventListener('pointerdown', stopTour);
  renderer.domElement.addEventListener('wheel', stopTour, { passive: true });
  app.onFrame = (dt) => {
    if (tour.active) {
      tour.time = (tour.time + dt) % tourSeconds;
      poseAt(tour.time, controls.target, camera.position);
    }
    museum.onFrame(dt);
  };
  const recordSeconds = museum.recordSeconds;
  Object.assign(window, { scans: { tour, tourSeconds, recordSeconds }, museum });

  const folder = app.gui.addFolder('Museum');
  folder.add({ 'fly-through': () => ((tour.active = !tour.active), tour.active && (tour.time = 0)) }, 'fly-through');
  const byKind = (kind: string) => museum.slots.filter((slot) => slot.kind === kind).length;
  app.extraStats = `museum: ${byKind('statue')} statue slots, ${byKind('specimen')} specimen slots, ${byKind('hero')} hero slots; ${count} stand-ins`;
}
