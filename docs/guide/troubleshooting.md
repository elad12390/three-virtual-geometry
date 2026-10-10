# Troubleshooting

Start with the browser's developer console (F12, or Cmd+Option+I on a Mac). Almost every problem leaves a message
there: a WebGPU validation error, a failed download, or a warning from three.js. If you load the library from a CDN,
also see the [CDN troubleshooting table](/guide/cdn#troubleshooting).

## WebGPU is not available

three-virtual-geometry needs WebGPU: its level-of-detail selection and culling run in compute shaders.

1. Type `navigator.gpu` in the console. `undefined` means the browser has no WebGPU, or the page isn't a secure
   context: it must be served over `https://`, or from `http://localhost`. Pages opened from `file://` never have it.
2. If `navigator.gpu` exists, run `await navigator.gpu.requestAdapter()`. `null` means the browser has WebGPU, but no
   usable GPU: the GPU or driver is blocklisted, or hardware acceleration is turned off in the browser settings.
3. Supported browsers: Chrome or Edge 113+, Safari 26+, Firefox 141+ on Windows. On Linux and Android, Chrome's
   WebGPU support depends on the GPU and driver; `chrome://gpu` shows its status.

[webgpureport.org](https://webgpureport.org) lists everything a browser reports about its WebGPU support.

`WebGPURenderer` can fall back to WebGL 2 when WebGPU is missing, but virtual geometry can't run there. Check
`renderer.backend.isWebGPUBackend` after `await renderer.init()`, and show a message (or a simpler scene) when it is
not `true`.

## Nothing is drawn

Go through these in order:

- **The renderer.** It must be a `WebGPURenderer` imported from `three/webgpu` (not `three`), and
  `await renderer.init()` must finish before the first render.
- **The scene.** Is the object you passed to `vg.add` (or the mesh from `vg.createMesh`) added to the scene you
  render? With `replace: false`, `vg.add` changes nothing in your scene, and you must add `result.object` yourself.
- **The camera.** Is the model inside the camera's `near` to `far` range, and in front of it? Large worlds often need
  a larger `far`.
- **Compilation.** Meshes appear a moment after the first frame, while their GPU programs compile in the background.
  `await vg.compileAsync(renderer, scene, camera)` waits for them.
- **Visibility.** `mesh.visible` must be `true`, and the mesh's `layers` must match the camera's.
- **The console.** WebGPU validation errors there usually name the problem directly.

Then read the statistics: `await vg.readStats(renderer)`. If `instances` is `0`, nothing was converted (see
[Some meshes were not converted](#some-meshes-were-not-converted)). If `instances` is positive but `visibleInstances`
is `0`, everything was culled: check the camera, `maxDrawDistance` and `minPixelRadius`.

## Some meshes were not converted

`vg.add` returns an object that lists what it left alone and why:

```ts
const result = await vg.add(gltf.scene);
console.table(result.skipped.map((s) => ({ name: s.object.name, reason: s.reason })));
```

These are left as normal three.js objects and keep rendering as before:

- skinned meshes (animated characters) and meshes with morph targets,
- points, lines and sprites,
- meshes whose material has no node equivalent (`ShaderMaterial`, `RawShaderMaterial`). Pass the `material` option of
  `vg.add` to convert them yourself, see [Importing models](/guide/import).

A `filter` function you passed to `vg.add` can also exclude meshes.

## Textures are missing or look wrong

- Use node materials, or let `vg.add` convert classic ones (it does that automatically).
- In your own node materials, sample textures with `vgUv` (or `vgTexture(map)`), not `uv()`: a virtual mesh has no
  `uv` attribute. `texture(map)` without explicit coordinates uses `vgUv` automatically.
- Only the first UV channel is used. Ambient-occlusion or light maps on a second UV channel sample the first one.
- Textures on coarse levels of detail can stretch a little, because simplification keeps UV seams but doesn't weigh
  UV distortion. This shows only at about one pixel of error, so it's rarely visible.

## Shading looks wrong

- **Hard edges became smooth (or the reverse).** `vg.add` keeps a model's own normals, so its hard and smooth edges
  stay as they were. `fromBufferGeometry`, used for your own geometry, computes smooth normals by default; pass
  `fromBufferGeometry(geometry, { normals: 'keep' })` to keep the geometry's normals and its hard edges.
- **Leaves or thin surfaces are dark from one side.** Use `side: THREE.DoubleSide` on the material, as in three.js.
- **Normal maps look inverted.** Check `material.normalScale`, as in three.js. Tangents are computed from screen-space
  derivatives, so no tangent attribute is needed.

## Shadows are missing

Shadows work as in three.js: `renderer.shadowMap.enabled = true`, `light.castShadow = true`, and `castShadow` /
`receiveShadow` on the meshes. `vg.add` copies `castShadow` and `receiveShadow` from the source meshes. Objects outside
the light's shadow camera don't cast shadows, so size `light.shadow.camera` to cover the scene, as in any three.js
scene.

Shadow maps use a coarser level of detail than the camera (`vg.shadowErrorScale`, default `3`). Set
`vg.shadowErrorScale.value = 1` if shadows of fine detail look blocky.

## Detail visibly changes while moving

- Lower `vg.errorThreshold.value` (default `1` pixel). At `0.5`, changes are practically invisible, at a higher cost.
- Keep `vg.lodBlend` at `2` or more (it blends levels over a band instead of switching them), and use temporal
  anti-aliasing with `vg.lodBlendTemporal = true` to turn the blend into a smooth crossfade. See
  [Settings](/guide/settings#smooth-detail-changes).
- Don't set a `triangleBudget` unless you need one: with a fixed threshold, detail changes only with distance, a few
  clusters at a time. A budget changes the threshold for the whole screen at once.
- If `vg.lastStats.capacityUse` is close to `1`, the draw buffers are nearly full and the engine is coarsening detail
  to stay inside them. Raise the capacity: `new VirtualGeometry({ maxTriangles: 16_000_000 })`. See
  [Settings](/guide/settings#draw-capacity).

## Moved objects don't move

- Objects converted by `vg.add` are baked into instances in world space when you call it. After moving, rotating or
  scaling them (or their parents), call `result.syncTransforms()`.
- Meshes from `vg.createMesh`: update with `mesh.setMatrixAt(i, matrix)` and then call `mesh.commitInstances()` once.
  Without `commitInstances()`, nothing is uploaded.
- Don't move, rotate or scale a VirtualMesh object from `vg.createMesh` (or its parents): keep it at the origin and
  position instances only through their matrices. The instance matrices are world-space. A transform on the mesh
  object would shift what is drawn, but culling and level of detail would still use the matrices, so parts would be
  culled in the wrong places.

## It's slow

1. Measure first: `await vg.readStats(renderer)`. `drawnTriangles` is the drawing work, `visibleInstances` the culling
   work.
2. Many drawn triangles: raise `vg.errorThreshold.value` to `1.5` or `2`, or set `vg.triangleBudget`.
3. Many visible instances of small objects: set `maxDrawDistance` on grass, stones and other clutter, or raise
   `vg.minPixelRadius.value`.
4. Shadows: turn `castShadow` off for small clutter. Shadow maps redraw the casters from the light's point of view.
5. High resolution: `renderer.setPixelRatio(Math.min(devicePixelRatio, 2))` avoids rendering at 3x on some phones.

The demo's settings panel has an **uncapped fps** switch that shows how many frames per second the GPU can actually
render, beyond the display's refresh rate. See [Performance](/guide/performance) for what costs what, and the
benchmark numbers.

## The first load is slow

The first time a page sees a model, its cluster hierarchy is built in the browser. The result is cached in IndexedDB,
so later loads are 15 to 25 times faster. To skip the first build too, bake the models ahead of time with
`npx vg-bake` and ship the `.vgeo` files: see [Baking and caching](/guide/caching).

The cache is per browser and per site. Private windows, cleared site data and a different domain all start empty.

## Flickering clusters or holes

These shouldn't happen. If `vg.lastStats.overflow` is `true`, a draw buffer overflowed: raise `maxTriangles`,
`maxShadowTriangles` or `maxMeshlets` in the `VirtualGeometry` options. Otherwise, please report it (see below) with a
screenshot and the model if you can share it.

## Reporting a bug

Open an issue on [GitHub](https://github.com/elad12390/three-virtual-geometry/issues) and include:

- the browser and its version, the operating system, and the GPU (from `chrome://gpu` or webgpureport.org),
- the console output, including any WebGPU validation errors,
- the output of `await vg.readStats(renderer)`,
- the three.js and three-virtual-geometry versions,
- if possible, a minimal page or model that shows the problem.
