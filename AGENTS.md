# AGENTS.md

Guidance for AI coding agents working on this repository. To **use** the library in another project, read
[docs/public/llms.txt](docs/public/llms.txt) instead.

## What this is

three-virtual-geometry: virtual geometry (a cluster LOD hierarchy with GPU-driven selection and culling) for
three.js WebGPU, written in TypeScript with TSL compute shaders. An independent implementation, not affiliated with
Epic Games. Do not copy code from, or describe this as derived from, any proprietary engine.

## Commands

```bash
npm install
npm run dev          # demo at http://localhost:8090 (?scene=ruins|map|world|import|stress|test)
npm test             # vitest: DAG invariants, partitioner, serialization, import grouping (Node, no GPU)
npm run typecheck    # tsc --noEmit
npm run build        # library (dist/index.js + .d.ts) and the vg-bake CLI (dist/bake.mjs)
npm run build:demo   # demo into dist-demo/
npm run docs:dev     # VitePress docs site; the embedded demo player (/live) loads the demo from `npm run dev`
npm run bench        # uncapped FPS of every demo scene in Chrome (Playwright) -> docs/benchmarks.json
```

GPU code can only be checked in a WebGPU browser. After runtime changes, load the demo scenes and check the console
for WebGPU validation errors and the HUD for `overflow`.

## Layout

| Path | Role |
| --- | --- |
| `src/index.ts` | Public API. Everything users import is re-exported here. |
| `src/source.ts` | `fromBufferGeometry`, `mergeSources`: three.js geometry to a welded `VirtualMeshSource`. |
| `src/core/preprocess/buildVirtualMesh.ts` | Builds the cluster DAG (meshoptimizer clusterize and simplify, with locked borders). |
| `src/core/preprocess/partition.ts` | Groups meshlets by shared edges. |
| `src/core/preprocess/voxelProxy.ts` | Voxel proxy surfaces for coarse levels of aggregate geometry. |
| `src/core/io/serialize.ts` | `.vgeo` binary format. |
| `src/core/io/cache.ts` | IndexedDB build cache. Bump `VG_BUILD_VERSION` when build output changes. |
| `src/core/runtime/VirtualGeometry.ts` | Context: settings, pools, per-frame update, stats, threshold controller. |
| `src/core/runtime/GeometryPool.ts` | Shared buffers for many meshes and the compute passes (cells, instances, meshlets, prefix, expand). |
| `src/core/runtime/VirtualMesh.ts` | A three.js `Mesh` that draws its region of a pool with an indirect draw. |
| `src/core/runtime/OcclusionCulling.ts` | Depth pre-pass, hierarchical depth buffer, automatic on/off tuning. |
| `src/core/runtime/vgMaterial.ts` | `vgUv`, `vgTexture` and material rewiring. |
| `src/core/runtime/cut.ts` | CPU reference of the GPU selection, used by tests. |
| `src/core/import/` | `vg.add()`: grouping an `Object3D` into instanced meshes, material conversion. |
| `scripts/bake.ts` | The `vg-bake` CLI. |
| `examples/` | Demo app (Vite root). Scenes are generated procedurally in `examples/demo/`. |
| `docs/` | VitePress site, deployed with the demo to GitHub Pages. |

## Invariants

- A meshlet is drawn when `projectedError <= threshold < projectedParentError`. Errors must grow monotonically up
  the DAG; `tests/preprocess.test.ts` checks it. Breaking it causes holes or overlaps.
- Meshlets have at most 128 triangles (`MAX_MESHLET_TRIANGLES`); the draw index encodes `slot << 7 | local`.
- Storage-buffer bindings must stay under 128 MB: pools cap their buffers by `maxPoolBytes`.
- In compute shaders, `workgroupBarrier()` must be reached in uniform control flow: no early return before it.
- If the build output changes, bump `VG_BUILD_VERSION` and update the fingerprint in `tests/buildversion.test.ts`.

## Conventions

- TypeScript, 2-space indentation, single quotes. Keep the public API small and stable, and document new options in
  `docs/` and `llms.txt`.
- The engine decides how to render, never what: new optimizations must be automatic with an opt-out setting, not
  something users must call.
- Use three.js and meshoptimizer for what they already do instead of reimplementing it.
