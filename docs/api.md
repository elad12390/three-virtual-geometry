# API reference

Everything is exported from `three-virtual-geometry`. Full type information ships with the package (`index.d.ts`).

## VirtualGeometry

One per app. Holds the settings and runs selection and culling for all its meshes.

```ts
const vg = new VirtualGeometry(options?: VirtualGeometryOptions);
```

| Option | Default | |
| --- | --- | --- |
| `maxTriangles` | `10_000_000` | Camera triangles drawn per frame, per pool. |
| `maxShadowTriangles` | `4_000_000` | Shadow-map triangles drawn per frame, per pool. |
| `maxMeshlets` | `1 << 20` | Clusters drawn per frame, per pool. |

| Method | |
| --- | --- |
| `add(object, options?)` | Converts the static meshes under an `Object3D` (e.g. `gltf.scene`) in place. Returns a `VirtualGeometryImport`. See [Importing models](/guide/import). |
| `createMesh(data, material, { matrices, colors? }, options?)` | Creates a `VirtualMesh` from built data. `matrices`: 16 floats per instance. |
| `compileAsync(renderer, scene, camera)` | Compiles every GPU pipeline now, e.g. during a loading screen. |
| `update(renderer, camera)` | Runs selection and culling. Only needed with `autoUpdate = false`. |
| `readStats(renderer)` | Promise of `VirtualGeometryStats`. |
| `dispose()` | Disposes every mesh and GPU buffer. |

| Property | Default | |
| --- | --- | --- |
| `errorThreshold.value` | `1` | Allowed error in pixels. Higher is coarser and faster. |
| `errorInCssPixels` | `true` | Measure the threshold in CSS pixels rather than device pixels. |
| `triangleBudget` | `0` (off) | Target drawn triangles; the threshold adapts slowly to hold it. |
| `thresholdRange` | `[0.5, 3]` | Bounds for the adapted threshold. |
| `minPixelRadius.value` | `0.7` | Skip instances smaller than this on screen (radius, pixels). |
| `frustumCulling.value` | `1` | `0` disables frustum culling. |
| `occlusion.enabled` | `true` | Occlusion culling. |
| `occlusion.auto` | `true` | Keep occlusion only when it is measured to be faster. |
| `occlusion.resolutionScale` | `0.5` | Resolution of the occlusion depth pass. |
| `shadowErrorScale.value` | `3` | Shadow detail is `errorThreshold` times this. |
| `autoShadowSweep` | `true` | Keep casters just outside the view, using the shadow camera's direction. |
| `autoUpdate` | `true` | Update before every render of a scene with virtual meshes. |
| `asyncCompile` | `true` | Compile pipelines in the background. |
| `freeze` | `false` | Keep the current selection. |
| `debugMode.value` | `0` | One of `VG_DEBUG_MODES`. |
| `lastStats` | `null` | Latest stats, refreshed every `statsInterval` (10) frames. |

### VirtualGeometryStats

| Field | |
| --- | --- |
| `drawnTriangles`, `drawnMeshlets` | Drawn by the camera this frame. |
| `fullDetailTriangles` | What drawing every instance at full detail would cost. |
| `instances`, `visibleInstances` | All instances, and those that passed frustum and size culling. |
| `occludedInstances` | Hidden by occlusion culling. |
| `shadowTriangles` | Drawn into shadow maps. |
| `requestedTriangles` | Selected before capacity limits. |
| `capacityUse`, `overflow` | Fullest draw buffer (fraction), and whether one overflowed. |

## VirtualGeometryImport

Returned by `vg.add()`.

| Member | |
| --- | --- |
| `meshes` | The created `VirtualMesh`es. |
| `groups` | One entry per mesh: the source meshes, geometry, material and world matrices. |
| `object` | The object that holds the new meshes. |
| `skipped` | Meshes left as they were (skinned, points, ...), each with a reason. |
| `stats` | `sourceMeshes`, `instances`, `uniqueGeometries`, `buildMs`. |
| `syncTransforms()` | Re-reads world transforms after you move the imported object. |
| `dispose()` | Removes the virtual meshes and shows the originals again. |

`add` options: `build` (build options), `mesh` (`VirtualMeshOptions`, or a function of the group), `filter(mesh)`,
`replace` (default `true`), `onProgress(fraction)`, `builder` (default `buildVirtualMeshCached`).

## VirtualMesh

A three.js `Mesh`. Add it to a scene; `castShadow`, `receiveShadow`, `visible`, `layers` and `renderOrder` work as
usual.

| Member | |
| --- | --- |
| `setMatrixAt(i, matrix)` then `commitInstances()` | Move instances. |
| `maxDrawDistance` | Cull distance in world units (default `Infinity`). Instances shrink away over the last 15%. |
| `minPixelRadius` | Per-mesh screen-size cull; the larger of this and the context's value applies. |
| `occluder` | Draw into the occlusion depth pass (default: opaque, single-sided, no alpha test). |
| `occlusionCulling` | Let occlusion hide this mesh's instances (default `true`). |
| `instanceCount`, `fullDetailTriangles` | Read-only counts. |
| `dispose()` | Stop rendering and free its part of the pool. |

## Building

| Function | |
| --- | --- |
| `fromBufferGeometry(geometry, options?)` | A `VirtualMeshSource` from a `BufferGeometry`: welds by position and UV, computes normals. Options: `uvs`, `colors`, `normals: 'compute' \| 'keep'`, `range`. |
| `mergeSources(parts)` | Merges several sources into one. |
| `buildVirtualMesh(source, options?)` | Builds the cluster hierarchy. Returns `VirtualMeshData`. |
| `buildVirtualMeshCached(source, options?)` | Same, cached in IndexedDB. Extra option `cache: { key?, store? }`. |
| `clearVirtualMeshCache()` | Empties the cache. |

Build options: `prune` (`false`; remove small disconnected pieces, for grass and leaves), `voxelLods` (`true`),
`groupSize` (`12`), `maxLodLevels` (`24`), `minRootTriangles` (`4`), `onProgress(fraction)`.

## Files

| Function | |
| --- | --- |
| `encodeVirtualMesh(data, options?)` | `VirtualMeshData` to `.vgeo` bytes. |
| `decodeVirtualMesh(bytes)` | `.vgeo` bytes to `VirtualMeshData`. |
| `loadVirtualMesh(urlOrBytes)` | Fetches and decodes a `.vgeo` file. |
| `VirtualMeshFormatError` | Thrown for damaged or newer-version files. |

CLI: `npx vg-bake model.glb -o out/` (see [Baking and caching](/guide/caching)).

## Materials

| Export | |
| --- | --- |
| `toNodeMaterial(material)` | Node-material equivalent of a classic material, or `null`. |
| `vgUv` | Texture coordinates of a virtual mesh, for custom nodes (instead of `uv()`). |
| `vgWorldNormal` | World-space normal, for custom nodes. |
| `vgTexture(texture)` | `texture(t, vgUv)`. |

## Debugging

| Export | |
| --- | --- |
| `VG_DEBUG_MODES` | `shaded`, `meshlets`, `lodLevel`, `instances`. |
| `selectCut(data, view, threshold)` | CPU reference of the GPU selection. |
| `verifyCutCoverage(data, cut)` | Leaf clusters a cut fails to cover (empty: no holes). |
