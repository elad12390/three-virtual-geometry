# Getting started

three-virtual-geometry makes three.js draw detailed models cheaply. You give it meshes at full detail, as many and
as dense as you like. Every frame, it works out on the GPU which parts of which meshes are visible and how much
detail each part needs to look right at its size on screen, about one triangle per pixel, and draws only that.
There are no LOD models to author and no draw distances to tune. You keep using three.js as usual.

This page takes you from nothing to a running scene. If you don't use npm or a bundler, see
[Use it from a CDN](/guide/cdn) instead: a single HTML file is enough.

## Requirements

| What | Version | Why |
| --- | --- | --- |
| three.js | r180 (`0.180.0`) or newer | The library builds on three.js's WebGPU renderer and node materials. It is tested with `0.186.1`. |
| Renderer | `WebGPURenderer` from `three/webgpu` | The level-of-detail selection and culling run in WebGPU compute shaders. The classic `WebGLRenderer` can't run them. |
| Browser | Chrome or Edge 113+, Safari 26+, Firefox 141+ on Windows | These have WebGPU. On Linux and Android, Chrome's WebGPU support depends on the GPU and driver. |
| Page | served over `https://`, or from `http://localhost` | WebGPU is only available in secure contexts. |

To check a browser, open its console and type `navigator.gpu`. If it prints `undefined`, the browser or device has
no WebGPU. [webgpureport.org](https://webgpureport.org) shows the details.

::: warning No WebGL fallback
`WebGPURenderer` can fall back to WebGL 2 on devices without WebGPU, but virtual geometry cannot: its meshes need
compute shaders and storage buffers. Check `navigator.gpu` and show a message (or a simpler scene) when it is
missing. See [Without WebGPU](#without-webgpu) below.
:::

## Install

```bash
npm install three-virtual-geometry three
```

That installs the library and three.js. The only other dependency, [meshoptimizer](https://github.com/zeux/meshoptimizer)
(mesh simplification, as WebAssembly), comes along automatically. TypeScript types are included.

The package is a standard ES module and works with every modern bundler: Vite, webpack 5, Rollup, esbuild,
Parcel, and the frameworks built on them.

## Your first scene

Load a glTF model and hand it to `VirtualGeometry`. This is a complete program:

```ts
import * as THREE from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VirtualGeometry } from 'three-virtual-geometry';

// A normal three.js WebGPU setup.
const renderer = new THREE.WebGPURenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
await renderer.init(); // required: sets up the GPU before anything is rendered

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 5000);
camera.position.set(0, 2, 6);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2));

// 1. One VirtualGeometry for the whole app.
const vg = new VirtualGeometry();

// 2. Load a model and convert it. Materials, textures and instancing are handled for you.
const gltf = await new GLTFLoader().loadAsync('/models/level.glb');
await vg.add(gltf.scene);
scene.add(gltf.scene);

// 3. Render as usual. Level of detail and culling run automatically before each render.
renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});
```

Step by step:

1. **`new VirtualGeometry()`** creates the shared state: settings, GPU buffers and the per-frame work. Create one for
   the whole app and use it for every model.
2. **`await vg.add(gltf.scene)`** converts the model in place. Call it before adding the model to the scene. It
   resolves when the model's meshes are built and ready to draw.
3. **`renderer.render(scene, camera)`** works unchanged. Just before each render of a scene that contains virtual
   geometry, the GPU selects the detail and culls hidden geometry for the camera you render with. You don't call
   anything extra.

### What `vg.add` does

- Every static `Mesh` and `InstancedMesh` under the object becomes a `VirtualMesh`. Skinned meshes, meshes with
  morph targets, points, lines and sprites are left alone and keep rendering normally.
- Meshes that share a geometry and a material become **one** VirtualMesh with one instance per copy, so a forest
  of 10,000 copies of a tree is one draw call.
- Materials are converted to node materials and keep their textures: color, normal, roughness, metalness, emissive,
  ambient occlusion and alpha. Shadows, vertex colors and instance colors keep working too.
- Your scene graph stays as it was: names, transforms, `userData` and child objects are preserved. The original
  meshes stop rendering, and the converted ones are added under the same object.
- The cluster hierarchy of each unique geometry is built once and cached in the browser (IndexedDB).

All options, the returned object and what isn't converted are described in [Importing models](/guide/import).

### First load and later loads

The first time a page sees a model, the library builds its cluster hierarchy in the browser. That takes a fraction
of a second for typical game assets and longer for dense scans: about 0.3 seconds for a 50,000-triangle mesh, and
several seconds for a mesh with millions of triangles. The result is cached, so later visits load it 15 to 25 times
faster. To show progress during the first load:

```ts
await vg.add(gltf.scene, {
  onProgress: (fraction) => (progressBar.style.width = `${fraction * 100}%`),
});
```

To skip building in the browser entirely, bake the models ahead of time with the `vg-bake` command: see
[Baking and caching](/guide/caching).

## Your own geometry

Any `BufferGeometry` works, including procedural ones. Build its cluster hierarchy once, then create a mesh with as
many instances as you like:

```ts
import { buildVirtualMeshCached, fromBufferGeometry } from 'three-virtual-geometry';

// Once per geometry (cached in the browser).
const data = await buildVirtualMeshCached(fromBufferGeometry(myGeometry));

// One 4x4 matrix per instance: 16 numbers each, in three.js's Matrix4 order.
const count = 1000;
const matrices = new Float32Array(count * 16);
const m = new THREE.Matrix4();
for (let i = 0; i < count; i++) {
  m.makeTranslation(Math.random() * 100, 0, Math.random() * 100);
  m.toArray(matrices, i * 16);
}

const rocks = vg.createMesh(data, new THREE.MeshStandardNodeMaterial({ color: 0x8a8274 }), { matrices });
rocks.castShadow = true;
scene.add(rocks);
```

Dense meshes are the point: a million-triangle scan or sculpt costs about the same per frame as a low-poly one once
it covers only part of the screen, because only the detail visible on screen is drawn.
[Meshes and instances](/guide/meshes) covers moving instances, per-instance colors and draw distances.

## Hide the first-frame shader compile (optional)

The first time a material is drawn, the browser compiles its GPU programs. By default this happens in the
background: the page never freezes, and meshes appear a moment after the first frame. If you have a loading screen,
compile everything behind it instead, so the first frame is complete:

```ts
await vg.compileAsync(renderer, scene, camera);
hideLoadingScreen();
```

## Check that it works

Read the statistics once the scene is running:

```ts
const stats = await vg.readStats(renderer);
console.log(stats.drawnTriangles, 'of', stats.fullDetailTriangles, 'triangles drawn');
```

- `drawnTriangles` should be far below `fullDetailTriangles` in any large scene. That gap is the work saved.
- `visibleInstances` counts instances that survived culling; `occludedInstances` counts those hidden behind others.
- `overflow` should be `false`. If it is `true`, see [Settings and levers](/guide/settings#draw-capacity).

For a visual check, color every cluster differently:

```ts
import { VG_DEBUG_MODES } from 'three-virtual-geometry';
vg.debugMode.value = VG_DEBUG_MODES.meshlets; // back to normal: VG_DEBUG_MODES.shaded
```

Near objects show many small clusters and distant ones a few large ones. That is the level of detail at work.

## Moving from WebGLRenderer

If your project uses the classic renderer, the switch is mostly mechanical:

| Before | After |
| --- | --- |
| `import * as THREE from 'three'` | `import * as THREE from 'three/webgpu'` |
| `new THREE.WebGLRenderer()` | `new THREE.WebGPURenderer()`, then `await renderer.init()` before the first render |
| `MeshStandardMaterial` and other classic materials | keep them: `vg.add` converts them. For meshes you build with `vg.createMesh`, use the node versions (`MeshStandardNodeMaterial`, ...) or [`toNodeMaterial`](/api#materials). |
| `ShaderMaterial` | rewrite it as a node material. Classic shader materials don't work with the WebGPU renderer. |
| post-processing with `EffectComposer` | three.js's node-based `THREE.RenderPipeline` (named `PostProcessing` before r183) |
| `requestAnimationFrame(loop)` | still works; `renderer.setAnimationLoop(loop)` is the recommended equivalent |

Addons (`three/addons/...`) keep working. See three.js's [WebGPU guide](https://threejs.org/manual/#en/webgpurenderer)
for the details of the renderer itself.

## Without WebGPU

Detect WebGPU before creating anything, and decide what to show without it:

```ts
if (!navigator.gpu) {
  showMessage('This page needs WebGPU: use Chrome or Edge 113+, or Safari 26+.');
} else {
  startApp();
}
```

`navigator.gpu` can exist on a device whose GPU is blocklisted. To be certain, also check
`await navigator.gpu.requestAdapter()`, which returns `null` when no GPU is usable.

## Server-side rendering frameworks

three-virtual-geometry renders in the browser only: it needs WebGPU, and caches builds in IndexedDB. In frameworks
that render pages on the server, such as Next.js, Nuxt, SvelteKit or Astro, create the renderer and the
`VirtualGeometry` on the client only, for example inside an effect or in the framework's client-only component.
Importing the package on the server works (it doesn't touch browser APIs until you use it), so shared modules can
import its types and functions.

## Next steps

- [Use it from a CDN](/guide/cdn): no npm, no bundler, one HTML file.
- [Importing models](/guide/import): every option of `vg.add`, materials and textures.
- [Meshes and instances](/guide/meshes): moving instances, draw distance, colors.
- [Settings and levers](/guide/settings): quality, performance budget, shadows, occlusion, debug views.
- [Baking and caching](/guide/caching): skip the first-load build.
- [Set it up with your AI agent](/guide/ai-agents): a prompt for Claude Code, Cursor or Copilot.
