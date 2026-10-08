/**
 * glTF import: a parking lot of textured glTF cars (Kenney Car Kit, CC0), converted with one call:
 *
 *   await vg.add(lot);
 *
 * Every car is a plain clone of a loaded glTF scene, so the cars share geometry and materials: the import
 * turns each (geometry, material) pair into one VirtualMesh with one instance per car.
 * Open with `?scene=import`. `&model=<url of a .glb>` imports that model instead (on a grid, `&count=` copies).
 * `&compare`: converted objects (left) next to the same objects rendered by three.js (right, kept out of the
 * import by `filter`): a glTF car, a sphere with color/normal/roughness maps, an alpha-tested leaf card.
 */
import * as THREE from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import type { DemoApp } from './app';

const KIT = 'models/kenney-car-kit/';
const CARS = ['sedan', 'police', 'taxi', 'firetruck', 'van', 'suv', 'truck', 'ambulance', 'garbage-truck', 'race'];

export async function createImportScene(app: DemoApp) {
  const { scene, camera, controls, vg, renderer } = app;
  const params = new URLSearchParams(location.search);
  scene.background = new THREE.Color(0xa9c3d9);
  scene.fog = new THREE.Fog(0xa9c3d9, 150, 700);
  scene.add(new THREE.HemisphereLight(0xdfeeff, 0x5a5048, 2.2));
  const sun = new THREE.DirectionalLight(0xfff2e0, 2.8);
  sun.position.set(-60, 100, 40);
  sun.castShadow = true;
  sun.shadow.mapSize.set(4096, 4096);
  Object.assign(sun.shadow.camera, { left: -90, right: 90, top: 90, bottom: -90, near: 1, far: 400 });
  sun.shadow.camera.updateProjectionMatrix();
  sun.shadow.bias = -0.0005;
  scene.add(sun);
  renderer.shadowMap.enabled = true;
  // Keep casters just outside the view in the cut, so their shadows still land in view.
  vg.shadowSweep.value.copy(sun.position).normalize().multiplyScalar(-60);

  const loader = new GLTFLoader();
  const lot = new THREE.Group();
  let rng = 7;
  const random = () => ((rng = (rng * 16807) % 2147483647) / 2147483647);

  const model = params.get('model');
  const reference = new Set<THREE.Object3D>();
  if (params.has('compare')) {
    const objects = new THREE.Group();
    let offset = 1.3;
    if (model) {
      const gltf = await loader.loadAsync(model);
      const box = new THREE.Box3().setFromObject(gltf.scene);
      gltf.scene.position.y = -box.min.y;
      offset = box.getSize(new THREE.Vector3()).x * 0.6;
      objects.add(gltf.scene);
    } else {
      const car = await loader.loadAsync(`${KIT}firetruck.glb`);
      objects.add(car.scene);
      const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.7, 96, 48), texturedMaterial());
      sphere.position.set(0, 0.8, 3);
      const leaf = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 1.6), cutoutMaterial());
      leaf.position.set(0, 1.2, 5.2);
      leaf.lookAt(sun.position.clone().normalize().add(leaf.position)); // face the sun: its holes show in the shadow
      objects.add(sphere, leaf);
    }
    const converted = objects.clone();
    converted.position.x = -offset;
    const original = objects.clone();
    original.position.x = offset;
    original.traverse((o) => reference.add(o));
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(12, 14).rotateX(-Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0x8a8f96 }));
    floor.position.z = 2.5;
    lot.add(converted, original, floor);
    camera.position.set(0, 4, 10);
    controls.target.set(0, 0.6, 2.5);
  } else if (model) {
    app.progress(`Loading ${model}…`, 0);
    const gltf = await loader.loadAsync(model);
    const box = new THREE.Box3().setFromObject(gltf.scene);
    const size = box.getSize(new THREE.Vector3());
    const spacing = Math.max(size.x, size.z) * 1.4;
    const side = Math.ceil(Math.sqrt(Number(params.get('count') ?? 100)));
    for (let x = 0; x < side; x++) {
      for (let z = 0; z < side; z++) {
        const copy = gltf.scene.clone();
        copy.position.set((x - side / 2) * spacing, -box.min.y, (z - side / 2) * spacing);
        copy.rotation.y = random() * Math.PI * 2;
        lot.add(copy);
      }
    }
    camera.position.set(spacing * 1.5, size.y * 1.5, spacing * 2.5);
    controls.target.set(0, size.y * 0.5, 0);
  } else {
    app.progress('Loading glTF cars…', 0);
    const cars = await Promise.all(CARS.map((name) => loader.loadAsync(`${KIT}${name}.glb`)));
    const side = Math.max(1, Math.round(Math.sqrt(Number(params.get('count') ?? 2500))));
    const pitch = new THREE.Vector2(3.2, 6);
    for (let x = 0; x < side; x++) {
      for (let z = 0; z < side; z++) {
        if (random() < 0.12) continue; // a few empty spots
        const car = cars[Math.floor(random() * cars.length)].scene.clone();
        car.position.set((x - side / 2) * pitch.x + (random() - 0.5) * 0.3, 0, (z - side / 2) * pitch.y + (random() - 0.5) * 0.4);
        car.rotation.y = (z % 2 ? Math.PI : 0) + (random() - 0.5) * 0.15;
        lot.add(car);
      }
    }
    // Asphalt: an ordinary three.js mesh, converted along with the cars.
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(side * pitch.x + 40, side * pitch.y + 40).rotateX(-Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0x3a3d42, roughness: 0.95 }));
    ground.name = 'ground';
    lot.add(ground);
    camera.position.set(14, 7, 22);
    controls.target.set(0, 0.5, 0);
  }
  // Flags set on the source meshes carry over to the VirtualMeshes.
  lot.traverse((o) => {
    o.castShadow = o.receiveShadow = true;
  });
  scene.add(lot);

  // ---- the one call ----
  const t0 = performance.now();
  const imported = await vg.add(lot, {
    filter: (mesh) => !reference.has(mesh),
    onProgress: async (f) => {
      app.progress('Building virtual geometry…', f);
      await new Promise((r) => setTimeout(r, 0)); // let the progress bar repaint
    },
  });
  const ms = performance.now() - t0;
  console.log('vg import', imported.stats, imported.groups.length, 'meshes', `${ms.toFixed(0)} ms`, imported.skipped);

  const { sourceMeshes, instances, uniqueGeometries } = imported.stats;
  app.extraStats = `import: ${sourceMeshes.toLocaleString()} glTF meshes -> ${imported.meshes.length} VirtualMeshes (${instances.toLocaleString()} instances, ${uniqueGeometries} unique geometries) in ${(ms / 1000).toFixed(1)} s`;
}

/** 256x256 RGBA DataTexture from a per-texel function returning [r, g, b, a] in 0..1. */
function dataTexture(texel: (u: number, v: number) => number[], colorSpace: THREE.ColorSpace = THREE.NoColorSpace) {
  const size = 256;
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const c = texel((x + 0.5) / size, (y + 0.5) / size);
      for (let k = 0; k < 4; k++) data[(y * size + x) * 4 + k] = Math.round(Math.max(0, Math.min(1, c[k])) * 255);
    }
  }
  const t = new THREE.DataTexture(data, size, size);
  t.colorSpace = colorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.generateMipmaps = true;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.needsUpdate = true;
  return t;
}

/** Checker color map, bumpy normal map, striped roughness map: shows wrong UVs or tangents at a glance. */
function texturedMaterial() {
  const height = (u: number, v: number) => Math.sin(u * Math.PI * 16) * Math.sin(v * Math.PI * 8);
  const map = dataTexture((u, v) => ((Math.floor(u * 8) + Math.floor(v * 4)) % 2 ? [0.9, 0.75, 0.2, 1] : [0.15, 0.35, 0.8, 1]), THREE.SRGBColorSpace);
  const normalMap = dataTexture((u, v) => {
    const e = 1 / 256;
    const dx = (height(u + e, v) - height(u - e, v)) / (2 * e);
    const dy = (height(u, v + e) - height(u, v - e)) / (2 * e);
    const n = new THREE.Vector3(-dx * 0.01, -dy * 0.01, 1).normalize();
    return [n.x * 0.5 + 0.5, n.y * 0.5 + 0.5, n.z * 0.5 + 0.5, 1];
  });
  const roughnessMap = dataTexture((u) => [0, Math.floor(u * 6) % 2 ? 0.15 : 0.9, 0, 1]);
  return new THREE.MeshStandardMaterial({ map, normalMap, normalScale: new THREE.Vector2(1, 1), roughnessMap, metalness: 0.2 });
}

/** Alpha-tested card with round holes: the holes must also show in its shadow. */
function cutoutMaterial() {
  const map = dataTexture((u, v) => {
    const fu = (u * 5) % 1 - 0.5;
    const fv = (v * 5) % 1 - 0.5;
    return [0.3, 0.6 + v * 0.3, 0.2, fu * fu + fv * fv < 0.09 ? 0 : 1];
  }, THREE.SRGBColorSpace);
  return new THREE.MeshStandardMaterial({ map, alphaTest: 0.5, side: THREE.DoubleSide });
}
