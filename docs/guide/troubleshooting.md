# Troubleshooting

## "WebGPU is not available"

three-virtual-geometry needs the WebGPU renderer. Use Chrome or Edge 113+, Safari 26+, or Firefox 141+ on Windows.
On Linux, Chrome may need WebGPU enabled in `chrome://flags`. Check `navigator.gpu` in the console.

## Nothing is drawn

- Is the renderer a `WebGPURenderer` from `three/webgpu`, and did you `await renderer.init()`?
- Is the VirtualMesh (or the object you passed to `vg.add`) added to the scene you render?
- Right after loading, pipelines compile in the background for a moment. Call `await vg.compileAsync(renderer,
  scene, camera)` to wait for them.
- Check the console for WebGPU validation errors.

## Textures are missing or wrong

- Use node materials, or let `vg.add` convert classic ones. In your own nodes, sample with `vgUv`, not `uv()`.
- Only the first UV channel is used.

## Detail visibly changes while moving

Lower `vg.errorThreshold.value` (default 1 px), and avoid a `triangleBudget` if you don't need one: with a fixed
threshold, detail only changes with distance. If `vg.lastStats.capacityUse` is close to 1, raise `maxTriangles`.

## It's slow

- Read `await vg.readStats(renderer)`: `drawnTriangles` and `visibleInstances` show where the work is.
- Set `maxDrawDistance` on small props, raise `errorThreshold`, or set a `triangleBudget`.
- See [Performance](/guide/performance).

## The first load is slow

Building hierarchies takes time the first time. It is cached in IndexedDB afterwards. For production, bake
`.vgeo` files with `npx vg-bake` (see [Baking and caching](/guide/caching)).

## Skinned characters

Skinned and morphing meshes are not converted; `vg.add` leaves them as normal three.js meshes, which keep working.
