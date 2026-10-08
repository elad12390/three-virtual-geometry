# Use it from a CDN (no build step)

You don't need npm, a bundler or a build step. Every version of three-virtual-geometry published to npm is also
served by the public CDNs [jsDelivr](https://www.jsdelivr.com/package/npm/three-virtual-geometry) and
[unpkg](https://unpkg.com/browse/three-virtual-geometry/), so a single HTML file can use it directly. This page shows
both ways to do that, explains every line, and lists the mistakes that commonly break it.

## Two builds: pick one

The package ships two single-file builds for the browser:

| | All-in-one | Minimal |
| --- | --- | --- |
| File | `dist/three-virtual-geometry.all.min.js` | `dist/three-virtual-geometry.min.js` |
| Contains | three-virtual-geometry, **three.js** (WebGPU build and TSL), meshoptimizer, `GLTFLoader`, `OrbitControls`, `RoomEnvironment` | three-virtual-geometry and meshoptimizer. **No three.js.** |
| You add | nothing | three.js, through an import map |
| Setup | one `import` line | an import map with 4 entries |
| Download size | 1.4 MB (about 400 kB compressed) | 230 kB (about 80 kB compressed), plus three.js |
| Other three.js addons (`DRACOLoader`, `KTX2Loader`, `TransformControls`, ...) | not available | all of them, from the same CDN |
| Choose it when | you want the quickest start: a prototype, a demo, a single page | you already use three.js on the page, or need other addons, or want to pick the three.js version |

Both builds are standard JavaScript modules (ES modules), loaded with `<script type="module">`, which every browser
with WebGPU supports. They contain exactly the same three-virtual-geometry code as the npm package.

::: tip Live examples
Both examples below run on this site. Open them and use your browser's **View source**: each is the exact file shown
on this page, with nothing hidden.
[All-in-one example](/examples/cdn-all.html){target="_blank"} ·
[Minimal example](/examples/cdn-minimal.html){target="_blank"}
:::

## All-in-one: one import

Save this as `index.html`, serve the folder with any static web server (see [Serving the page](#serving-the-page)),
and open it in a WebGPU browser. It loads a model, places 400 copies of it, and converts them with one call:

<<< @/public/examples/cdn-all.html

The whole setup is this one line:

```js
import { THREE, VirtualGeometry, GLTFLoader, OrbitControls, RoomEnvironment }
  from 'https://cdn.jsdelivr.net/npm/three-virtual-geometry@0.1/dist/three-virtual-geometry.all.min.js';
```

What the all-in-one build exports:

| Export | What it is | In an npm project you'd write |
| --- | --- | --- |
| `THREE` | all of three.js's WebGPU build: `THREE.Scene`, `THREE.WebGPURenderer`, `THREE.MeshStandardNodeMaterial`, ... | `import * as THREE from 'three/webgpu'` |
| `TSL` | three.js Shading Language functions, for custom node materials: `TSL.texture`, `TSL.color`, ... | `import * as TSL from 'three/tsl'` |
| `GLTFLoader` | loads `.glb` and `.gltf` models | `import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'` |
| `OrbitControls` | mouse and touch camera controls | `import { OrbitControls } from 'three/addons/controls/OrbitControls.js'` |
| `RoomEnvironment` | a neutral environment for lighting PBR materials | `import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'` |
| everything else | the full three-virtual-geometry API: `VirtualGeometry`, `buildVirtualMeshCached`, `fromBufferGeometry`, `vgUv`, ... ([API reference](/api)) | `import { ... } from 'three-virtual-geometry'` |

Two rules for the all-in-one build:

- **Use its `THREE`, and don't load three.js a second time.** The build contains three.js. If the page also loads
  three.js from somewhere else, there are two copies, and objects from one are not recognized by the other: they
  silently don't render, or fail with errors deep inside the renderer. Need an addon that isn't included? Use the
  minimal build.
- **The default file.** `https://cdn.jsdelivr.net/npm/three-virtual-geometry@0.1` (no file path) also serves the
  all-in-one build. The explicit file name is clearer, so the examples use it.

## Minimal: bring your own three.js

The minimal build leaves three.js out. It imports `three/webgpu` and `three/tsl` by name, and an **import map** tells
the browser which file those names mean. Use it when the page already uses three.js, when you need addons that the
all-in-one build doesn't include, or when you want to choose the three.js version yourself.

<<< @/public/examples/cdn-minimal.html

There are two parts to it.

1. **An import map** (`<script type="importmap">`). It tells the browser which URL to load when code says
   `import ... from 'three'` or `from 'three-virtual-geometry'`. This is a standard browser feature, supported by
   every browser that has WebGPU. It must come before any module script.
2. **A module script** (`<script type="module">`). This is your application code. It is the same code you would
   write with npm and a bundler; only where the packages come from is different.

### The import map, line by line

```json
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.186.1/build/three.webgpu.js",
    "three/webgpu": "https://cdn.jsdelivr.net/npm/three@0.186.1/build/three.webgpu.js",
    "three/tsl": "https://cdn.jsdelivr.net/npm/three@0.186.1/build/three.tsl.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.186.1/examples/jsm/",
    "three-virtual-geometry": "https://cdn.jsdelivr.net/npm/three-virtual-geometry@0.1/dist/three-virtual-geometry.min.js"
  }
}
```

| Entry | Needed by | Why it points there |
| --- | --- | --- |
| `three` | three.js addons (loaders, controls) and your code | three.js addons import from `'three'`. Pointing it at the **WebGPU build**, the same file as `three/webgpu`, gives the whole page a single copy of three.js. three.js's own WebGPU examples map it the same way. |
| `three/webgpu` | three-virtual-geometry and your code | The WebGPU renderer and node materials. |
| `three/tsl` | three-virtual-geometry | Three.js Shading Language functions, used to build the GPU programs. It loads `three/webgpu` itself, through the same map. |
| `three/addons/` | your code | The prefix for addons: `three/addons/loaders/GLTFLoader.js` becomes `.../examples/jsm/loaders/GLTFLoader.js`. The trailing `/` is required on both sides. |
| `three-virtual-geometry` | your code | The minimal build of this library (meshoptimizer is inside it). |

If your own code doesn't import `'three'` or any `three/addons/...`, you can leave those two entries out. Keeping them
costs nothing: the browser only downloads what is actually imported.

### Choosing versions

**Pin an exact version of three.js.** A URL without a version (`three/build/...`) always loads the newest release, so
your page could break on its own when a new version comes out.

| Package | Supported versions | In the examples |
| --- | --- | --- |
| `three` | r180 (`0.180.0`) or newer | `0.186.1`, the version this release is tested with and the one inside the all-in-one build |
| `three-virtual-geometry` | any published version | `0.1`, the newest `0.1.x` |

For three-virtual-geometry, a partial version like `@0.1` picks up bug fixes automatically and never jumps to `0.2`,
which may change the API. Use an exact version such as `@0.1.0` when the page must never change. All versions are
listed on [npm](https://www.npmjs.com/package/three-virtual-geometry?activeTab=versions).

**Every `three` entry must use the same version and the same CDN.** The browser treats two different URLs as two
different modules. If `three` and `three/webgpu` point to different versions, or one to jsDelivr and the other to
unpkg, the page loads two copies of three.js, with the symptoms described above.

## Using unpkg instead of jsDelivr

Both CDNs serve the same files from npm. Swap the host and keep the paths:

```js
// All-in-one
import { THREE, VirtualGeometry } from 'https://unpkg.com/three-virtual-geometry@0.1/dist/three-virtual-geometry.all.min.js';
```

```json
{
  "imports": {
    "three": "https://unpkg.com/three@0.186.1/build/three.webgpu.js",
    "three/webgpu": "https://unpkg.com/three@0.186.1/build/three.webgpu.js",
    "three/tsl": "https://unpkg.com/three@0.186.1/build/three.tsl.js",
    "three/addons/": "https://unpkg.com/three@0.186.1/examples/jsm/",
    "three-virtual-geometry": "https://unpkg.com/three-virtual-geometry@0.1/dist/three-virtual-geometry.min.js"
  }
}
```

Use the plain file URLs shown here. Avoid URLs that rewrite a package's imports for you, such as jsDelivr's `/+esm`
URLs or esm.sh without `?external=three`: they point the library's `import ... from 'three/webgpu'` at their own
copy of three.js instead of the one in your import map, which gives you two copies of three.js.

## Geometry made in code

Everything from the npm guides works the same way. A field of 10,000 instanced rocks built from a `BufferGeometry`,
written for the all-in-one build (with the minimal build, import `THREE` from `'three/webgpu'` instead):

```html
<script type="module">
  import { THREE, VirtualGeometry, buildVirtualMeshCached, fromBufferGeometry }
    from 'https://cdn.jsdelivr.net/npm/three-virtual-geometry@0.1/dist/three-virtual-geometry.all.min.js';

  // ... renderer, scene and camera as in the examples above ...
  const vg = new VirtualGeometry();

  // A dense source mesh: about 85,000 triangles. Built once, then cached in the browser.
  const rock = new THREE.IcosahedronGeometry(1, 64);
  const data = await buildVirtualMeshCached(fromBufferGeometry(rock));

  // One 4x4 matrix per instance (16 numbers each).
  const count = 10_000;
  const matrices = new Float32Array(count * 16);
  const m = new THREE.Matrix4();
  for (let i = 0; i < count; i++) {
    m.makeTranslation((Math.random() - 0.5) * 500, 0, (Math.random() - 0.5) * 500);
    m.toArray(matrices, i * 16);
  }
  scene.add(vg.createMesh(data, new THREE.MeshStandardNodeMaterial({ color: 0x8a8274 }), { matrices }));
</script>
```

See [Meshes and instances](/guide/meshes) for moving instances, colors and draw distances.

## Serving the page

Opening the file by double-clicking it (a `file://` address) **does not work**. Browsers block module scripts and
model downloads on `file://` pages, and WebGPU needs a secure context: a page served over `https://`, or from
`http://localhost`.

Any static server works. Run one of these in the folder that contains `index.html`, then open
`http://localhost:8080`:

```bash
npx serve -l 8080            # Node.js
python3 -m http.server 8080  # Python
```

To publish the page, upload it to any static host (GitHub Pages, Netlify, Cloudflare Pages, S3, your own server).
It must be served over HTTPS.

## Loading your own models

Models are downloaded by the page, so the same rules apply as for any web request:

- **Same site:** put `model.glb` next to `index.html` and load `'./model.glb'`. This always works.
- **Another site:** that server must allow cross-origin requests (send an `Access-Control-Allow-Origin` header).
  GitHub Pages, jsDelivr and most CDNs do. If it doesn't, the browser console shows a CORS error and the model
  doesn't load.
- **Draco or meshopt-compressed glTF:** configure the loader as in three.js (`DRACOLoader`, or
  `GLTFLoader.setMeshoptDecoder`). These addons aren't in the all-in-one build, so use the minimal build for them.
  three-virtual-geometry sees the decoded meshes either way.

## First load and caching

The first time a page sees a model, three-virtual-geometry builds its cluster hierarchy in the browser: a fraction of
a second for small models, up to tens of seconds for scenes with billions of triangles. The result is stored in the
browser's IndexedDB, so the next visit loads it in a moment. This works the same from a CDN as from npm. Use
`onProgress` (as in the examples) to show progress, and see [Baking and caching](/guide/caching) to ship pre-built
files instead.

The CDN files themselves are cached by the browser and by the CDN. A URL with an exact version (like `@0.1.0`) never
changes, so browsers can keep it for a long time.

## Self-hosting the files

To avoid depending on a CDN (an intranet, an offline app, strict security rules), copy the files to your own server
and load them from there. Get them from the npm package (`npm pack three-virtual-geometry`, or `node_modules` after
`npm install`):

| Build | Files to copy | Then |
| --- | --- | --- |
| All-in-one | `dist/three-virtual-geometry.all.min.js` (and `.map` for debugging) | `import { THREE, VirtualGeometry } from '/vendor/three-virtual-geometry.all.min.js'` |
| Minimal | `dist/three-virtual-geometry.min.js`, plus three.js's whole `build/` folder (`three.webgpu.js` loads `three.core.js` next to it) and `examples/jsm/` if you use addons | point the import map entries at your copies |

Keep three.js's folder structure as it is: its files import each other by relative paths.

## Content Security Policy

If your site sends a `Content-Security-Policy` header, allow:

- `script-src`: the CDN host (`https://cdn.jsdelivr.net` or `https://unpkg.com`) and `'wasm-unsafe-eval'`, because
  meshoptimizer compiles WebAssembly.
- `connect-src`: every host you load models from.
- An inline import map or inline module script needs a nonce or hash in `script-src`, like any inline script.

## The npm entry point

The package's main entry, `dist/index.js`, is meant for bundlers: it imports `three/webgpu`, `three/tsl` and
`meshoptimizer` by name and is not minified. You can load it with an import map too (add a `meshoptimizer` entry
pointing to `https://cdn.jsdelivr.net/npm/meshoptimizer@1.3.0/index.js`), but the minimal build is the same code,
smaller and with one entry fewer.

## When to use npm instead

The CDN builds are ideal for prototypes, demos, CodePen-style experiments and sites without a build step. For larger
apps, [installing from npm](/guide/getting-started) with a bundler such as Vite gives you TypeScript types and
autocompletion, bundles only the code you use, and serves everything from your own domain.

## Troubleshooting

| What you see | Cause and fix |
| --- | --- |
| `Failed to resolve module specifier "three/webgpu"` (or `"three"`, ...) | You use the minimal build without an import map entry for that name, or the import map comes **after** the module script. The import map must be the first script on the page. |
| `Failed to load resource: ... 404` in the console | A typo in a URL or version. Open the URL directly in the browser to check it. |
| `Cannot use import statement outside a module` | The script tag lacks `type="module"`. |
| `The requested module ... does not provide an export named ...` | A misspelled import, or importing `THREE`/`GLTFLoader` from the minimal build (only the all-in-one build exports them). |
| Nothing renders, no errors | Check that `navigator.gpu` exists (WebGPU is available), that `await renderer.init()` runs before rendering, and that three.js is loaded only once. |
| Blocked by CORS policy | The model's server doesn't allow cross-origin requests. Host the model next to the page. |
| Works on `localhost`, not on your server | The server uses `http://`. WebGPU needs HTTPS (or `localhost`). |
| `TypeError` inside three.js | three.js is older than r180, or two copies of three.js are loaded. |

For problems that aren't specific to CDNs, see [Troubleshooting](/guide/troubleshooting).
