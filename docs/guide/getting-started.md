# Getting started

## Install

```bash
npm install three-virtual-geometry three
```

Requirements:

- three.js r180 or newer, using the WebGPU renderer (`three/webgpu`).
- A browser with WebGPU: Chrome or Edge 113+, Safari 26+, Firefox 141+ (Windows).

## Your first scene

Load a glTF model and hand it to `VirtualGeometry`. That is the whole integration:

```ts
import * as THREE from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VirtualGeometry } from 'three-virtual-geometry';

const renderer = new THREE.WebGPURenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
await renderer.init();

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 5000);
camera.position.set(0, 2, 6);
new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2));

// 1. One VirtualGeometry per app.
const vg = new VirtualGeometry();

// 2. Load a model and convert it. Materials, textures and instancing are handled for you.
const gltf = await new GLTFLoader().loadAsync('/models/level.glb');
await vg.add(gltf.scene);
scene.add(gltf.scene);

// 3. Render as usual. Level of detail and culling run automatically before each render.
renderer.setAnimationLoop(() => renderer.render(scene, camera));
```

What `vg.add` does:

- Every static `Mesh` and `InstancedMesh` becomes a `VirtualMesh`. Skinned and morphing meshes, points, lines and
  sprites are left alone and keep rendering normally.
- Meshes that share geometry and material are merged into **one** VirtualMesh with one instance per copy.
- Materials are converted to node materials and keep their textures (color, normal, roughness, metalness,
  emissive, AO, alpha), as well as shadows, vertex colors and instance colors.
- The cluster hierarchy of each unique geometry is built once and cached in the browser, so the next page load is
  fast.

## Your own geometry

Any `BufferGeometry` works. Build its cluster hierarchy once, then create a mesh with as many instances as you like:

```ts
import { buildVirtualMeshCached, fromBufferGeometry } from 'three-virtual-geometry';

const data = await buildVirtualMeshCached(fromBufferGeometry(myGeometry));

const matrices = new Float32Array(1000 * 16); // one 4x4 matrix per instance
const m = new THREE.Matrix4();
for (let i = 0; i < 1000; i++) {
  m.makeTranslation(Math.random() * 100, 0, Math.random() * 100);
  m.toArray(matrices, i * 16);
}

const rocks = vg.createMesh(data, new THREE.MeshStandardNodeMaterial({ color: 0x8a8274 }), { matrices });
scene.add(rocks);
```

Dense meshes are the point: a million-triangle scan or sculpt costs about the same per frame as a low-poly one,
once it is far enough away to cover few pixels.

## Hide the first-frame shader compile (optional)

Pipelines compile in the background by default, so meshes appear a moment after the first frame instead of the page
freezing. Behind a loading screen, compile everything up front instead:

```ts
await vg.compileAsync(renderer, scene, camera);
hideLoadingScreen();
```

## Next steps

- [Importing models](/guide/import): options of `vg.add`, materials and textures.
- [Meshes and instances](/guide/meshes): moving instances, draw distance, colors.
- [Settings and levers](/guide/settings): quality, performance budget, shadows, occlusion, debug views.
- [Set it up with your AI agent](/guide/ai-agents): a prompt you can paste into Claude Code, Cursor or Copilot.
